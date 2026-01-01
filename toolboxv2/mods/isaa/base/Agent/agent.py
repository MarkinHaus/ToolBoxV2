import asyncio
import json
import logging
import os
import pickle
import random
import re
import threading
import time
import types
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List

import pandas as pd
import yaml

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.intelligent_rate_limiter import (
    IntelligentRateLimiter,
    LiteLLMRateLimitHandler,
    load_handler_from_file,
    create_handler_from_config,
)
from toolboxv2.mods.isaa.base.tbpocketflow import AsyncFlow, AsyncNode

from pydantic import BaseModel, ValidationError

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
litllm_logger = logging.getLogger("LiteLLM")
litllm_logger.setLevel(logging.CRITICAL)
git_logger = logging.getLogger("git")
git_logger.setLevel(logging.CRITICAL) #(get_logger().level)
mcp_logger = logging.getLogger("mcp")
mcp_logger.setLevel(logging.CRITICAL)
urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.setLevel(logging.CRITICAL)
chardet_logger = logging.getLogger("chardet")
chardet_logger.setLevel(logging.CRITICAL)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.CRITICAL)
asyncio_logger = logging.getLogger("asyncio")
asyncio_logger.setLevel(logging.CRITICAL)

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
    vision_models = ['vision', 'gpt-4o', 'gpt-4-turbo', 'claude-3', 'gemini', 'llava', 'cogvlm']
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
            if isinstance(v, (types.FunctionType, types.ModuleType, threading.Thread, FlowAgent, AsyncNode, VariableManager, UnifiedContextManager)):
                continue
            if _is_json_serializable(v):
                clean_dict[k] = _clean_data_for_serialization(v)
        return clean_dict
    elif isinstance(data, list):
        clean_list = []
        for item in data:
            if isinstance(item, (types.FunctionType, types.ModuleType, threading.Thread, FlowAgent, AsyncNode, VariableManager, UnifiedContextManager)):
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

# ===== CORE NODE IMPLEMENTATIONS =====


@with_progress_tracking
class LLMToolNode(AsyncNode):
    """Enhanced LLM tool with automatic tool calling and agent loop integration"""

    def __init__(self, model: str = None, max_tool_calls: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.model = model or os.getenv("COMPLEXMODEL", "openrouter/qwen/qwen3-code")
        self.max_tool_calls = max_tool_calls
        self.call_log = []

        # Models die Function Calling unterstützen
        self.function_calling_models = {
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "claude-3",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-3-5-sonnet",
            "claude-sonnet-4",
            "claude-opus-4",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro",
            "mistral-large",
            "mistral-medium",
            "mistral-small",
            "command-r",
            "command-r-plus",
            "llama-3.1",
            "llama-3.2",
            "llama-3.3",
        }

    def _supports_function_calling(self, model: str) -> bool:
        """Prüft ob das Model native Function Calling unterstützt."""
        model_lower = model.lower()

        # Entferne Provider-Prefix (z.B. "openrouter/openai/gpt-4o" -> "gpt-4o")
        model_base = model_lower.split("/")[-1]

        for supported in self.function_calling_models:
            if supported in model_base:
                return True

        # Zusätzliche Checks für Provider-spezifische Modelle
        if "openai" in model_lower or "anthropic" in model_lower:
            return True
        if "gemini" in model_lower or "google" in model_lower:
            return True

        res = False
        try:
            res = litellm.supports_function_calling(model_base)
        except:
            pass
        return res

    def _prepare_tools_for_litellm(self, prep_res: dict) -> list[dict]:
        """Bereitet Tools für LiteLLM Function Calling vor."""
        agent_instance = prep_res.get("agent_instance")
        available_tools = prep_res.get("available_tools", [])

        if not agent_instance:
            return []

        # Konvertiere nur verfügbare Tools
        tools = get_selected_tools_litellm(
            tool_registry=agent_instance._tool_registry,
            tool_capabilities=agent_instance._tool_capabilities,
            selected_tools=available_tools
        )

        # Füge direct_response als spezielles Tool hinzu
        tools.append({
            "type": "function",
            "function": {
                "name": "direct_response",
                "description": "Provide the final answer when the task is complete. Use this to finish and return results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "final_answer": {
                            "type": "string",
                            "description": "The complete final answer to return"
                        }
                    },
                    "required": ["final_answer"]
                }
            }
        })

        return tools

    async def prep_async(self, shared):
        context = shared.get("formatted_context", {})
        task_description = shared.get("current_task_description", shared.get("current_query", ""))

        # Variable Manager integration
        variable_manager = shared.get("variable_manager")
        agent_instance = shared.get("agent_instance")

        return {
            "task_description": task_description,
            "context": context,
            "context_manager": shared.get("context_manager"),
            "session_id": shared.get("session_id"),
            "variable_manager": variable_manager,
            "agent_instance": agent_instance,
            "available_tools": shared.get("available_tools", [""]),
            "tool_capabilities": shared.get("tool_capabilities", {}),
            "persona_config": shared.get("persona_config"),
            "base_system_message": variable_manager.format_text(agent_instance.amd.get_system_message_with_persona()),
            "recent_interaction": context.get("recent_interaction", ""),
            "session_summary": context.get("session_summary", ""),
            "task_context": context.get("task_context", ""),
            "fast_llm_model": shared.get("fast_llm_model"),
            "complex_llm_model": shared.get("complex_llm_model"),
            "progress_tracker": shared.get("progress_tracker"),
            "tool_call_count": 0
        }

    async def _exec_async(self, prep_res):
        """Main execution with tool calling loop"""
        if not LITELLM_AVAILABLE:
            return await self._fallback_response(prep_res)

        progress_tracker = prep_res.get("progress_tracker")

        conversation_history = []
        tool_call_count = 0
        final_response = None
        model_to_use = "auto"
        total_llm_calls = 0
        total_cost = 0.0
        total_tokens = 0

        # Initial system message with tool awareness
        system_message = self._build_tool_aware_system_message(prep_res)

        # Initial user prompt with variable resolution
        initial_prompt = await self._build_context_aware_prompt(prep_res)
        conversation_history.append({"role": "user", "content":  prep_res["variable_manager"].format_text(initial_prompt)})
        runs = 0
        all_tool_results = {}
        while tool_call_count < self.max_tool_calls:
            runs += 1
            # Get LLM response
            messages = [{"role": "system", "content": system_message + ( "\nfist look at the context and reason over you intal step." if runs == 1 else "")}] + conversation_history

            model_to_use = self._select_optimal_model(prep_res["task_description"], prep_res)

            llm_start = time.perf_counter()

            try:
                agent_instance = prep_res["agent_instance"]
                response = await agent_instance.a_run_llm_completion(
                    model=model_to_use,
                    messages=messages,
                    temperature=0.7,
                    stream=True,
                    # max_tokens=2048,
                    node_name="LLMToolNode", task_id="llm_phase_" + str(runs)
                )

                llm_response = response
                if not llm_response and not final_response:
                    final_response = "I encountered an error while processing your request."
                    break

                # Check for tool calls
                tool_calls = self._extract_tool_calls(llm_response)

                llm_response = prep_res["variable_manager"].format_text(llm_response)
                conversation_history.append({"role": "assistant", "content": llm_response})


                if not tool_calls:
                    # No more tool calls, this is the final response
                    final_response = llm_response
                    break
                direct_response_call = next(
                    (call for call in tool_calls if call.get("tool_name") == "direct_response"), None)
                if direct_response_call:
                    final_response = direct_response_call.get("arguments", {}).get("final_answer",
                                                                                   "Task completed successfully.")
                    tool_call_count += 1
                    break

                # Execute tool calls
                tool_results = await self._execute_tool_calls(tool_calls, prep_res)
                tool_call_count += len(tool_calls)

                # Add tool results to conversation
                tool_results_text = self._format_tool_results(tool_results)
                all_tool_results[str(runs)] = tool_results_text
                final_response = tool_results_text
                next_prompt = f"""Tool results have been processed:
                {tool_results_text}

                **Your next step:**
                - If you have enough information to answer the user's request, you MUST call the `direct_response` tool with the final answer.
                - If you need more information, call the next required tool.
                - Do not provide a final answer as plain text. Always use the `direct_response` tool to finish."""

                conversation_history.append({"role": "user", "content": next_prompt})
                # Update variable manager with tool results
                self._update_variables_with_results(
                    tool_results, prep_res["variable_manager"]
                )

            except Exception as e:
                llm_duration = time.perf_counter() - llm_start

                if progress_tracker:
                    await progress_tracker.emit_event(
                        ProgressEvent(
                            event_type="llm_call",  # Konsistenter Event-Typ
                            node_name="LLMToolNode",
                            session_id=prep_res.get("session_id"),
                            status=NodeStatus.FAILED,
                            success=False,
                            duration=llm_duration,
                            llm_model=model_to_use,
                            error_details={"message": str(e), "type": type(e).__name__},
                            metadata={"call_number": total_llm_calls + 1},
                        )
                    )
                eprint(f"LLM tool execution failed: {e}")
                final_response = f"I encountered an error while processing: {str(e)}"
                import traceback

                print(traceback.format_exc())
                break

        return {
            "success": True,
            "final_response": final_response or "I was unable to complete the request.",
            "tool_calls_made": tool_call_count,
            "conversation_history": conversation_history,
            "model_used": model_to_use,
            "tool_results": all_tool_results,
            "llm_statistics": {
                "total_calls": total_llm_calls,
                "total_cost": total_cost,
                "total_tokens": total_tokens,
            },
        }

    async def exec_async(self, prep_res):
        """Main execution with native LiteLLM function calling or fallback."""
        if not LITELLM_AVAILABLE:
            return await self._fallback_response(prep_res)

        progress_tracker = prep_res.get("progress_tracker")
        model_to_use = self._select_optimal_model(prep_res["task_description"], prep_res)

        # Entscheide: Native Function Calling oder Manual Parsing
        use_native_function_calling = self._supports_function_calling(model_to_use)

        if use_native_function_calling:
            return await self._exec_with_native_function_calling(prep_res, model_to_use)
        else:
            return await self._exec_with_manual_parsing(prep_res, model_to_use)

    async def _exec_with_native_function_calling(
        self, prep_res: dict, model_to_use: str
    ) -> dict:
        """Execution mit nativem LiteLLM Function Calling."""
        progress_tracker = prep_res.get("progress_tracker")
        agent_instance = prep_res.get("agent_instance")
        variable_manager = prep_res.get("variable_manager")

        conversation_history = []
        tool_call_count = 0
        final_response = None
        all_tool_results = {}

        # System Message
        system_message = self._build_tool_aware_system_message_native(prep_res)

        # Initial User Prompt
        initial_prompt = await self._build_context_aware_prompt(prep_res)
        conversation_history.append(
            {
                "role": "user",
                "content": variable_manager.format_text(initial_prompt)
                if variable_manager
                else initial_prompt,
            }
        )

        # Tools für LiteLLM vorbereiten
        litellm_tools = self._prepare_tools_for_litellm(prep_res)

        runs = 0
        while tool_call_count < self.max_tool_calls:
            runs += 1

            messages = [
                {"role": "system", "content": system_message}
            ] + conversation_history

            try:
                # LiteLLM Completion mit Tools
                response_message = await agent_instance.a_run_llm_completion(
                    model=model_to_use,
                    messages=messages,
                    tools=litellm_tools if litellm_tools else None,
                    tool_choice="auto" if litellm_tools else None,
                    temperature=0.7,
                    get_response_message = True
                )

                # Check für Tool Calls
                tool_calls = response_message.tool_calls

                if not tool_calls:
                    # Keine Tool Calls - finale Antwort
                    final_response = response_message.content or "Task completed."
                    conversation_history.append(
                        {"role": "assistant", "content": final_response}
                    )
                    break

                # Tool Calls verarbeiten
                # Assistant Message mit Tool Calls hinzufügen
                assistant_message = {
                    "role": "assistant",
                    "content": response_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
                conversation_history.append(assistant_message)

                # Tool Calls ausführen
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name

                    # Check für direct_response
                    if tool_name == "direct_response":
                        try:
                            args = json.loads(tool_call.function.arguments)
                            final_response = args.get(
                                "final_answer", "Task completed."
                            )
                            tool_call_count += 1

                            # Tool Response hinzufügen
                            conversation_history.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": json.dumps(
                                        {
                                            "status": "completed",
                                            "response": final_response,
                                        }
                                    ),
                                }
                            )
                            break
                        except json.JSONDecodeError:
                            final_response = tool_call.function.arguments
                            break

                    # Normaler Tool Call
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        error_result = f"Invalid JSON arguments: {e}"
                        conversation_history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps({"error": error_result}),
                            }
                        )
                        continue

                    # Tool ausführen
                    tool_result = await self._execute_single_tool(
                        tool_name, args, prep_res, progress_tracker
                    )

                    tool_call_count += 1
                    all_tool_results[f"{runs}_{tool_name}"] = tool_result

                    # Tool Response zur Conversation hinzufügen
                    conversation_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result, default=str)[
                                :4000
                            ],  # Truncate für Context
                        }
                    )

                    # Variables aktualisieren
                    if tool_result.get("success") and variable_manager:
                        variable_manager.set(
                            f"results.{tool_name}.data", tool_result.get("result")
                        )

                # Check ob direct_response aufgerufen wurde
                if final_response:
                    break

            except Exception as e:
                import traceback

                traceback.print_exc()
                eprint(f"LLM Tool execution failed: {e}")
                final_response = f"Error during execution: {str(e)}"
                break

        return {
            "success": True,
            "final_response": final_response
            or "Task completed without specific result.",
            "tool_calls_made": tool_call_count,
            "conversation_history": conversation_history,
            "model_used": model_to_use,
            "tool_results": all_tool_results,
            "execution_mode": "native_function_calling",
        }

    async def _exec_with_manual_parsing(
        self, prep_res: dict, model_to_use: str
    ) -> dict:
        """Fallback Execution mit manuellem Parsing (für Models ohne Function Calling)."""
        progress_tracker = prep_res.get("progress_tracker")
        agent_instance = prep_res.get("agent_instance")
        variable_manager = prep_res.get("variable_manager")

        conversation_history = []
        tool_call_count = 0
        final_response = None
        all_tool_results = {}

        # System Message mit manuellen Tool-Instruktionen
        system_message = self._build_tool_aware_system_message(prep_res)

        # Initial User Prompt
        initial_prompt = await self._build_context_aware_prompt(prep_res)
        conversation_history.append(
            {
                "role": "user",
                "content": variable_manager.format_text(initial_prompt)
                if variable_manager
                else initial_prompt,
            }
        )

        runs = 0
        while tool_call_count < self.max_tool_calls:
            runs += 1

            messages = [
                {"role": "system", "content": system_message}
            ] + conversation_history

            try:
                # Normaler LLM Call ohne Tools
                llm_response = await agent_instance.a_run_llm_completion(
                    model=model_to_use,
                    messages=messages,
                    temperature=0.7,
                    stream=True,
                    node_name="LLMToolNode",
                    task_id=f"llm_phase_{runs}",
                )

                if not llm_response and not final_response:
                    final_response = "Error processing request."
                    break

                # Manuelles Tool Call Parsing
                tool_calls = self._extract_tool_calls(llm_response)

                llm_response = (
                    variable_manager.format_text(llm_response)
                    if variable_manager
                    else llm_response
                )
                conversation_history.append(
                    {"role": "assistant", "content": llm_response}
                )

                if not tool_calls:
                    final_response = llm_response
                    break

                # Check für direct_response
                direct_response_call = next(
                    (
                        call
                        for call in tool_calls
                        if call.get("tool_name") == "direct_response"
                    ),
                    None,
                )
                if direct_response_call:
                    final_response = direct_response_call.get("arguments", {}).get(
                        "final_answer", "Task completed."
                    )
                    tool_call_count += 1
                    break

                # Tool Calls ausführen
                tool_results = await self._execute_tool_calls(tool_calls, prep_res)
                tool_call_count += len(tool_calls)

                # Results formatieren
                tool_results_text = self._format_tool_results(tool_results)
                all_tool_results[str(runs)] = tool_results_text
                final_response = tool_results_text

                # Next Prompt
                next_prompt = f"""Tool results:
{tool_results_text}

Continue with the next step or call direct_response to finish."""
                conversation_history.append({"role": "user", "content": next_prompt})

                # Variables aktualisieren
                self._update_variables_with_results(tool_results, variable_manager)

            except Exception as e:
                import traceback

                traceback.print_exc()
                eprint(f"LLM Tool execution failed: {e}")
                final_response = f"Error: {str(e)}"
                break

        return {
            "success": True,
            "final_response": final_response or "Task completed.",
            "tool_calls_made": tool_call_count,
            "conversation_history": conversation_history,
            "model_used": model_to_use,
            "tool_results": all_tool_results,
            "execution_mode": "manual_parsing",
        }

    async def _execute_single_tool(
        self, tool_name: str, arguments: dict, prep_res: dict, progress_tracker
    ) -> dict:
        """Führt einen einzelnen Tool Call aus."""
        agent_instance = prep_res.get("agent_instance")
        variable_manager = prep_res.get("variable_manager")

        tool_start = time.perf_counter()

        if progress_tracker:
            await progress_tracker.emit_event(
                ProgressEvent(
                    event_type="tool_call",
                    timestamp=time.time(),
                    status=NodeStatus.RUNNING,
                    node_name="LLMToolNode",
                    tool_name=tool_name,
                    tool_args=arguments,
                    session_id=prep_res.get("session_id"),
                )
            )

        try:
            # Variable Resolution
            resolved_args = {}
            for key, value in arguments.items():
                if isinstance(value, str) and variable_manager:
                    resolved_args[key] = variable_manager.format_text(value)
                else:
                    resolved_args[key] = value

            # Tool ausführen
            result = await agent_instance.arun_function(tool_name, **resolved_args)
            tool_duration = time.perf_counter() - tool_start

            if progress_tracker:
                await progress_tracker.emit_event(
                    ProgressEvent(
                        event_type="tool_call",
                        timestamp=time.time(),
                        node_name="LLMToolNode",
                        status=NodeStatus.COMPLETED,
                        tool_name=tool_name,
                        tool_result=result,
                        duration=tool_duration,
                        success=True,
                        session_id=prep_res.get("session_id"),
                    )
                )

            return {
                "tool_name": tool_name,
                "arguments": resolved_args,
                "success": True,
                "result": result,
            }

        except Exception as e:
            tool_duration = time.perf_counter() - tool_start

            if progress_tracker:
                await progress_tracker.emit_event(
                    ProgressEvent(
                        event_type="tool_call",
                        timestamp=time.time(),
                        node_name="LLMToolNode",
                        status=NodeStatus.FAILED,
                        tool_name=tool_name,
                        tool_error=str(e),
                        duration=tool_duration,
                        success=False,
                        session_id=prep_res.get("session_id"),
                    )
                )

            return {
                "tool_name": tool_name,
                "arguments": arguments,
                "success": False,
                "error": str(e),
            }

    def _build_tool_aware_system_message_native(self, prep_res: dict) -> str:
        """System Message für native Function Calling (kürzer, keine Tool-Syntax-Erklärung)."""
        base_message = prep_res.get(
            "base_system_message", "You are a helpful AI assistant."
        )
        variable_manager = prep_res.get("variable_manager")

        system_parts = [
            "ROLE: INTERNAL EXECUTION UNIT",
            "You execute specific tasks using available tools.",
            "Execute ONLY the assigned task. Be precise and efficient.",
            "",
            base_message,
            "",
            "INSTRUCTIONS:",
            "1. Analyze the task",
            "2. Use appropriate tools to accomplish it",
            "3. Call direct_response with the final answer when done",
            "4. Store important results in variables for later use",
        ]

        # Variable Context
        if variable_manager:
            var_context = variable_manager.get_llm_variable_context()
            if var_context:
                system_parts.append(f"\nVARIABLE CONTEXT:\n{var_context}")

        return "\n".join(system_parts)

    def _build_tool_aware_system_message(self, prep_res: dict) -> str:
        base_message = prep_res.get("base_system_message", "")
        available_tools = prep_res.get("available_tools", [])

        system_parts = [
            "ROLE: Precise Execution Unit",
            "You provide answers based ONLY on available data and tool results.",
            "\nSTRICT ADHERENCE RULES:",
            "- If a variable or entity is not mentioned in the context, do NOT assign it a default value (like 0).",
            "- Explicitly report missing information as 'Information not available'.",
            "- If the user prompt contains a logical trap or asks for unstated details, point this out clearly.",
            "\n" + base_message,
            "\n## Available Tools: " + ", ".join(available_tools),
            "\nUse YAML for TOOL_CALLS. Use direct_response only when the task is fully resolved or proven unresolvable with given data."
        ]

        return "\n".join(system_parts)

    def _calculate_tool_relevance(self, query: str, capabilities: dict) -> float:
        """Calculate how relevant a tool is to the current query"""

        query_words = set(query.lower().split())

        # Check trigger phrases
        trigger_score = 0.0
        triggers = capabilities.get('trigger_phrases', [])
        for trigger in triggers:
            trigger_words = set(trigger.lower().split())
            if trigger_words.intersection(query_words):
                trigger_score += 0.04
        # Check confidence triggers if available
        conf_triggers = capabilities.get('confidence_triggers', {})
        for phrase, confidence in conf_triggers.items():
            if phrase.lower() in query:
                trigger_score += confidence/10
        # Check indirect connections
        indirect = capabilities.get('indirect_connections', [])
        for connection in indirect:
            connection_words = set(connection.lower().split())
            if connection_words.intersection(query_words):
                trigger_score += 0.02
        return min(1.0, trigger_score)

    @staticmethod
    def _extract_tool_calls_custom(text: str) -> list[dict]:
        """Extract tool calls from LLM response"""

        tool_calls = []

        pattern = r'TOOL_CALL:'
        matches = _extract_meta_tool_calls(text, pattern)

        for tool_name, args_str in matches:
            try:
                # Parse arguments
                args = _parse_tool_args(args_str)
                tool_calls.append({
                    "tool_name": tool_name,
                    "arguments": args
                })
            except Exception as e:
                wprint(f"Failed to parse tool call {tool_name}: {e}")

        return tool_calls

    @staticmethod
    def _extract_tool_calls(text: str) -> list[dict]:
        """Extract tool calls from LLM response using YAML format"""
        import re

        import yaml

        tool_calls = []

        print(text)
        # Pattern to find YAML blocks with TOOL_CALLS
        yaml_pattern = r'```yaml\s*\n(.*?TOOL_CALLS:.*?)\n```'
        yaml_matches = re.findall(yaml_pattern, text, re.DOTALL | re.IGNORECASE)

        # Also try without code blocks for simpler cases
        if not yaml_matches:
            simple_pattern = r'TOOL_CALLS:\s*\n((?:.*\n)*?)(?=\n\S|\Z)'
            simple_matches = re.findall(simple_pattern, text, re.MULTILINE)
            if simple_matches:
                yaml_matches = [f"TOOL_CALLS:\n{match}" for match in simple_matches]

        for yaml_content in yaml_matches:
            try:
                # Parse YAML content
                parsed_yaml = yaml.safe_load(yaml_content)

                if not isinstance(parsed_yaml, dict) or 'TOOL_CALLS' not in parsed_yaml:
                    continue

                calls = parsed_yaml['TOOL_CALLS']
                if not isinstance(calls, list):
                    calls = [calls]  # Handle single tool call

                for call in calls:
                    if isinstance(call, dict) and 'tool' in call:
                        tool_call = {
                            "tool_name": call['tool'],
                            "arguments": call.get('args', {})
                        }
                        tool_calls.append(tool_call)

            except yaml.YAMLError as e:
                wprint(f"Failed to parse YAML tool calls: {e}")
            except Exception as e:
                wprint(f"Error processing tool calls: {e}")

        return tool_calls

    def _select_optimal_model(self, task_description: str, prep_res: dict) -> str:
        """Select optimal model based on task complexity and available resources"""
        complexity_score = self._estimate_task_complexity(task_description, prep_res)
        if complexity_score > 0.7:
            return prep_res.get("complex_llm_model", "openrouter/openai/gpt-4o")
        else:
            return prep_res.get("fast_llm_model", "openrouter/anthropic/claude-3-haiku")

    def _estimate_task_complexity(self, task_description: str, prep_res: dict) -> float:
        """Estimate task complexity based on description, length, and available tools"""
        # Simple heuristic: length + keyword matching + tool availability
        description_length_score = min(len(task_description) / 500, 1.0)  # cap at 1.0
        keywords = ["analyze", "research", "generate", "simulate", "complex", "deep", "strategy"]
        keyword_score = sum(1 for k in keywords if k in task_description.lower()) / len(keywords)
        tool_score = min(len(prep_res.get("available_tools", [])) / 10, 1.0)

        # Weighted sum
        complexity_score = (0.5 * description_length_score) + (0.3 * keyword_score) + (0.2 * tool_score)
        return round(complexity_score, 2)

    async def _fallback_response(self, prep_res: dict) -> dict:
        """Fallback response if LiteLLM is not available"""
        wprint("LiteLLM not available — using fallback response.")
        return {
            "success": False,
            "final_response": (
                "I'm unable to process this request fully right now because the LLM interface "
                "is not available. Please try again later or check system configuration."
            ),
            "tool_calls_made": 0,
            "conversation_history": [],
            "model_used": None
        }

    async def _execute_tool_calls(self, tool_calls: list[dict], prep_res: dict) -> list[dict]:
        """Execute tool calls via agent"""
        agent_instance = prep_res.get("agent_instance")
        variable_manager = prep_res.get("variable_manager")
        progress_tracker = prep_res.get("progress_tracker")

        results = []

        for tool_call in tool_calls:
            tool_name = tool_call["tool_name"]
            arguments = tool_call["arguments"]

            # Start tool tracking
            tool_start = time.perf_counter()

            if progress_tracker:
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="tool_call",
                    timestamp=time.time(),
                    status=NodeStatus.RUNNING,
                    node_name="LLMToolNode",
                    tool_name=tool_name,
                    tool_args=arguments,
                    session_id=prep_res.get("session_id"),
                    metadata={"tool_call_initiated": True}
                ))

            try:
                # Resolve variables in arguments
                if variable_manager:
                    resolved_args = {}
                    for key, value in arguments.items():
                        if isinstance(value, str):
                            resolved_args[key] = variable_manager.format_text(value)
                        else:
                            resolved_args[key] = value
                else:
                    resolved_args = arguments

                # Execute via agent
                result = await agent_instance.arun_function(tool_name, **resolved_args)
                tool_duration = time.perf_counter() - tool_start
                variable_manager.set(f"results.{tool_name}.data", result)
                results.append({
                    "tool_name": tool_name,
                    "arguments": resolved_args,
                    "success": True,
                    "result": result
                })

            except Exception as e:
                tool_duration = time.perf_counter() - tool_start
                error_message = str(e)
                error_type = type(e).__name__
                import traceback

                print(traceback.format_exc())

                if progress_tracker:
                    await progress_tracker.emit_event(
                        ProgressEvent(
                            event_type="tool_call",
                            timestamp=time.time(),
                            node_name="LLMToolNode",
                            status=NodeStatus.FAILED,
                            tool_name=tool_name,
                            tool_args=arguments,
                            duration=tool_duration,
                            success=False,
                            tool_error=error_message,
                            session_id=prep_res.get("session_id"),
                            metadata={
                                "error": error_message,
                                "error_message": error_message,
                                "error_type": error_type,
                            },
                        )
                    )

                    # FIXED: Also send generic error event for error log
                    await progress_tracker.emit_event(
                        ProgressEvent(
                            event_type="error",
                            timestamp=time.time(),
                            node_name="LLMToolNode",
                            status=NodeStatus.FAILED,
                            success=False,
                            tool_name=tool_name,
                            metadata={
                                "error": error_message,
                                "error_message": error_message,
                                "error_type": error_type,
                                "source": "tool_execution",
                                "tool_name": tool_name,
                                "tool_args": arguments,
                            },
                        )
                    )
                eprint(f"Tool execution failed {tool_name}: {e}")
                results.append(
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "success": False,
                        "error": str(e),
                    }
                )

        return results

    def _format_tool_results(self, results: list[dict]) -> str:
        formatted = []
        for result in results:
            if result["success"]:
                # Nur der reine Output, kein technisches Rauschen
                formatted.append(f"Result from {result['tool_name']}:\n{result['result']}")
            else:
                # Klarer Fehlerbericht
                formatted.append(f"Notice: Tool {result['tool_name']} failed. Reason: {result['error']}")
        return "\n---\n".join(formatted)

    def _update_variables_with_results(self, results: list[dict], variable_manager):
        """Update variable manager with tool results"""
        if not variable_manager:
            return

        for i, result in enumerate(results):
            if result["success"]:
                tool_name = result["tool_name"]
                result_data = result["result"]

                # FIXED: Store result in proper variable paths
                variable_manager.set(f"results.{tool_name}.data", result_data)
                variable_manager.set(f"tools.{tool_name}.result", result_data)

                # Also store with index for multiple calls to same tool
                var_key = f"tool_result_{tool_name}_{i}"
                variable_manager.set(var_key, result_data)

    async def _build_context_aware_prompt(self, prep_res: dict) -> str:
        variable_manager = prep_res.get("variable_manager")
        context_manager = prep_res.get("context_manager")
        session_id = prep_res.get("session_id", "default")
        task_description = prep_res.get("task_description", "")

        prompt_parts = ["## Context and Task Information"]

        if context_manager:
            try:
                unified_context = await context_manager.build_unified_context(session_id, task_description)
                formatted = context_manager.get_formatted_context_for_llm(unified_context)
                prompt_parts.append(formatted)
            except: pass

        prompt_parts.append(f"\n## Current Request\n{task_description}")

        if variable_manager:
            suggestions = variable_manager.get_variable_suggestions(task_description)
            if suggestions:
                prompt_parts.append(f"\n## Data References\nAvailable variables: {', '.join(suggestions)}")

        final_prompt = "\n".join(prompt_parts)

        # Entfernt: "Return a REPORT summarizing the outcome." -> Ersetzt durch natürliche Instruktion
        final_prompt += "\n\nPlease complete this task efficiently. Provide a clear and helpful response."

        if variable_manager:
            final_prompt = variable_manager.format_text(final_prompt)
        return final_prompt

    async def post_async(self, shared, prep_res, exec_res):
        shared["current_response"] = exec_res.get("final_response", "Task completed.")
        shared["tool_calls_made"] = exec_res.get("tool_calls_made", 0)
        shared["llm_tool_conversation"] = exec_res.get("conversation_history", [])
        shared["synthesized_response"] = {"synthesized_response":exec_res.get("final_response", "Task completed."),
                                          "confidence": (0.7 if exec_res.get("model_used") == prep_res.get("complex_llm_model") else 0.6) if exec_res.get("success", False) else 0,
                                          "metadata": exec_res.get("metadata", {"model_used": exec_res.get("model_used")}),
                                          "synthesis_method": "llm_tool"}
        shared["results"] = exec_res.get("tool_results", [])
        return "llm_tool_complete"

@with_progress_tracking
class StateSyncNode(AsyncNode):
    """Synchronize state between world model and shared store"""
    async def prep_async(self, shared):
        world_model = shared.get("world_model", {})
        session_data = shared.get("session_data", {})
        tasks = shared.get("tasks", {})
        system_status = shared.get("system_status", "idle")

        return {
            "world_model": world_model,
            "session_data": session_data,
            "tasks": tasks,
            "system_status": system_status,
            "sync_timestamp": datetime.now().isoformat()
        }

    async def exec_async(self, prep_res):
        # Perform intelligent state synchronization
        sync_result = {
            "world_model_updates": {},
            "session_updates": {},
            "task_updates": {},
            "conflicts_resolved": [],
            "sync_successful": True
        }

        # Update world model with new information
        if "current_response" in prep_res:
            # Extract learnable facts from responses
            extracted_facts = self._extract_facts(prep_res.get("current_response", ""))
            sync_result["world_model_updates"].update(extracted_facts)

        # Sync task states
        for task_id, task in prep_res["tasks"].items():
            if task.status == "completed" and task.result:
                # Store task results in world model
                fact_key = f"task_{task_id}_result"
                sync_result["world_model_updates"][fact_key] = task.result

        return sync_result

    def _extract_facts(self, text: str) -> dict[str, Any]:
        """Extract learnable facts from text"""
        facts = {}
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            # Look for definitive statements
            if ' is ' in line and not line.startswith('I ') and not '?' in line:
                parts = line.split(' is ', 1)
                if len(parts) == 2:
                    subject = parts[0].strip().lower()
                    predicate = parts[1].strip().rstrip('.')
                    if len(subject.split()) <= 3:  # Keep subjects simple
                        facts[subject] = predicate

        return facts

    async def post_async(self, shared, prep_res, exec_res):
        # Apply the synchronization results
        if exec_res["sync_successful"]:
            shared["world_model"].update(exec_res["world_model_updates"])
            shared["session_data"].update(exec_res["session_updates"])
            shared["last_sync"] = datetime.now()
            return "sync_complete"
        else:
            wprint("State synchronization failed")
            return "sync_failed"


# ===== FLOW COMPOSITIONS =====
@with_progress_tracking
class TaskManagementFlow(AsyncFlow):
    """
    Enhanced Task-Management-Flow with LLMReasonerNode as strategic core.
    The flow now starts with strategic reasoning and delegates to specialized sub-systems.
    """

    def __init__(self, max_parallel_tasks: int = 3, max_reasoning_loops: int = 24, max_tool_calls:int = 5):
        # Create the strategic reasoning core (new primary node)
        self.llm_reasoner = LLMReasonerNode(max_reasoning_loops=max_reasoning_loops)

        # Create specialized sub-system nodes (now supporting nodes)
        self.sync_node = StateSyncNode()
        self.llm_tool_node = LLMToolNode(max_tool_calls=max_tool_calls)

        # Store references for the reasoner to access sub-systems
        # These will be injected into shared state during execution

        # === NEW HIERARCHICAL FLOW STRUCTURE ===

        # Primary flow: LLMReasonerNode is the main orchestrator
        # It makes strategic decisions and routes to appropriate sub-systems

        # The reasoner can internally call any of these sub-systems:
        # - LLMToolNode for direct tool usage
        # - TaskPlanner + TaskExecutor for complex project management
        # - Direct response for simple queries

        # Only one main connection: reasoner completes -> response generation
        self.llm_reasoner - "reasoner_complete" >> self.sync_node

        # Fallback connections for error handling
        self.llm_reasoner - "error" >> self.sync_node
        self.llm_reasoner - "timeout" >> self.sync_node

        # The old linear connections are removed - the reasoner now controls the flow internally

        super().__init__(start=self.llm_reasoner)

    async def run_async(self, shared):
        """Enhanced run with sub-system injection"""

        # Inject sub-system references into shared state so reasoner can access them
        shared["llm_tool_node_instance"] = self.llm_tool_node

        # Store tool registry access for the reasoner
        agent_instance = shared.get("agent_instance")
        if agent_instance:
            shared["tool_registry"] = agent_instance._tool_registry
            shared["tool_capabilities"] = agent_instance._tool_capabilities

        # Execute the flow with the reasoner as starting point
        return await super().run_async(shared)


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

SYSTEM_PROMPT =  """You are an intellectually honest and precise assistant. Your primary loyalty is to the TRUTH.

## Operational Modes:
1. DIRECT ANSWER: If you can solve the task purely with logic or your internal knowledge (e.g., riddles, math, general questions), provide the final answer immediately using `finish`.
2. TOOL USAGE: Only use `run_tools` if the task REQUIRES external data (web search, file access, specific calculations your internal logic cannot handle).
3. ERROR RECOVERY: If a tool fails, do not give up. Analyze the error and try to provide the best possible answer from your internal knowledge.

## Core Rules:
- NO ASSUMPTIONS: If information is truly missing, state it. Do not assume '0'.
- CLEAN OUTPUT: Your final answer via `finish` must be a natural, helpful response. Remove all technical artifacts like YAML or thought-processes.

{context}

Now, help with: {query}"""


# ============================================================================
# REASONER NODE - Lean Implementation
# ============================================================================


@dataclass
class ReasoningState:
    """Minimal state tracking"""

    loop_count: int = 0
    results: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    final_answer: Optional[str] = None
    completed: bool = False


@with_progress_tracking
class LLMReasonerNode(AsyncNode):
    """
    Lean strategic reasoning node with native function calling.

    Key improvements:
    - Uses agent_instance.a_run_llm_completion with tools parameter
    - Clear, minimal meta-tools that LLMs understand
    - No complex prompt engineering that confuses models
    - Direct execution path with clear abort conditions
    """

    def __init__(self, max_reasoning_loops: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.max_loops = max_reasoning_loops
        self.state: Optional[ReasoningState] = None
        self.agent_instance: Optional[FlowAgent] = None
        self.variable_manager = None

    async def prep_async(self, shared: dict) -> dict:
        """Minimal preparation - extract what we need"""
        self.state = ReasoningState()
        self.agent_instance = shared.get("agent_instance")
        self.variable_manager = shared.get("variable_manager")

        # Build minimal context
        context_parts = []

        # Available tools
        available_tools = shared.get("available_tools", [])
        if available_tools:
            context_parts.append(f"Available tools: {', '.join(available_tools[:20])}")

        # Previous results if any
        if self.variable_manager:
            latest = self.variable_manager.get("delegation.latest")
            if latest:
                context_parts.append(
                    f"Previous result available: {latest.get('task_description', 'unknown')[:100]}"
                )

        if "query" in shared:
            q = shared["current_query"].lower()
            # Erkennt suggestive Fragen, die auf fehlende Infos abzielen
            if any(word in q for word in ["wie viele", "wann", "wer", "how many", "when", "who"]) and len(shared.get("available_tools", [])) > 0:
                context_parts.append("\nNote: Validate if the subjects in the question actually exist in the context before calculating.")

        return {
            "query": shared.get("current_query", ""),
            "context": "\n".join(context_parts) if context_parts else "No prior context",
            "available_tools": available_tools,
            "model": shared.get("complex_llm_model", "openrouter/openai/gpt-4o"),
            "agent_instance": self.agent_instance,
            "variable_manager": self.variable_manager,
            "progress_tracker": shared.get("progress_tracker"),
            "session_id": shared.get("session_id", "default"),
            "tool_capabilities": shared.get("tool_capabilities", {}),
            "fast_llm_model": shared.get("fast_llm_model"),
        }

    async def exec_async(self, prep_res: dict) -> dict:
        """Main reasoning loop - simple and direct"""
        query = prep_res["query"]
        model = prep_res["model"]
        agent = prep_res["agent_instance"]
        progress_tracker = prep_res.get("progress_tracker")

        if not agent:
            return self._error_result("No agent instance available")

        # Check if model supports function calling
        use_function_calling = self._supports_function_calling(model)

        # Build initial messages
        system_msg = SYSTEM_PROMPT.format(
            max_loops=self.max_loops, context=prep_res["context"], query=query
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ]

        do = True
        # Main loop
        while do and self.state.loop_count < self.max_loops and not self.state.completed:
            self.state.loop_count += 1

            # Emit progress event
            if progress_tracker:
                await progress_tracker.emit_event(
                    ProgressEvent(
                        event_type="reasoning_loop",
                        timestamp=time.time(),
                        node_name="LLMReasonerNodeV2",
                        status=NodeStatus.RUNNING,
                        metadata={
                            "loop": self.state.loop_count,
                            "max_loops": self.max_loops,
                            "use_function_calling": use_function_calling,
                        },
                    )
                )

            try:
                if use_function_calling:
                    # Native function calling path
                    result = await self._execute_with_function_calling(
                        agent, model, messages, prep_res
                    )
                else:
                    # Fallback to manual parsing
                    result = await self._execute_with_manual_parsing(
                        agent, model, messages, prep_res
                    )

                if result.get("completed"):
                    self.state.completed = True
                    self.state.final_answer = result.get("answer")
                    do = False
                    break

                # Add result to history for next iteration
                if result.get("message"):
                    messages.append(result["message"])
                if result.get("tool_response"):
                    messages.append(result["tool_response"])

            except Exception as e:
                eprint(f"Reasoning loop {self.state.loop_count} error: {e}")
                # Don't break - try to recover
                messages.append(
                    {
                        "role": "user",
                        "content": f"Error occurred: {e}. Please try a different approach or complete with what you know.",
                    }
                )

        # Build final result
        if self.state.final_answer:
            return {
                "final_result": self.state.final_answer,
                "reasoning_loops": self.state.loop_count,
                "completed": True,
                "results": self.state.results,
            }
        else:
            # Timeout - create summary response
            return {
                "final_result": self._create_timeout_response(query),
                "reasoning_loops": self.state.loop_count,
                "completed": False,
                "timeout": True,
            }

    async def _execute_with_function_calling(
        self, agent: 'FlowAgent', model: str, messages: list, prep_res: dict
    ) -> dict:
        """Execute reasoning step with native function calling"""

        # Get dynamic schema with available tools hint
        available_tools = prep_res.get("available_tools", [])
        meta_tools = get_meta_tools_schema(available_tools)

        try:
            # Call LLM with tools - disable streaming for tool calls
            response = await agent.a_run_llm_completion(
                model=model,
                messages=messages,
                tools=meta_tools,
                tool_choice="auto",
                temperature=0.2,
                get_response_message=True,
                node_name="LLMReasonerNode",
                task_id=f"reasoning_loop_{self.state.loop_count}",
                stream=False,  # Important: disable streaming for function calls
            )
        except Exception as e:
            error_msg = str(e)
            # Handle malformed function call error
            if "Malformed function call" in error_msg or "MidStreamFallback" in error_msg:
                wprint(f"Malformed function call detected, retrying with guidance...")
                return {
                    "message": {"role": "assistant", "content": ""},
                    "tool_response": {
                        "role": "user",
                        "content": "Error: Your function call was malformed. Call functions like this:\n"
                        '- finish(final_answer="your answer here")\n'
                        '- run_tools(task_description="what to do", tool_names=["tool1"])\n'
                        '- reason(thought="your thinking")\n'
                        "Try again with correct syntax.",
                    },
                }
            raise

        # Extract tool calls
        tool_calls = getattr(response, "tool_calls", None)
        content = getattr(response, "content", "") or ""

        rprint(
            f"Loop {self.state.loop_count}: {len(tool_calls or [])} tool calls, content: {content[:100]}..."
        )

        if not tool_calls:
            # No tool calls - LLM responded with text
            # Check if this is a direct answer
            if content and len(content) > 50:
                # Treat as completion attempt
                return {"completed": True, "answer": content}
            else:
                # Nudge toward using tools
                return {
                    "message": {"role": "assistant", "content": content},
                    "tool_response": {
                        "role": "user",
                        "content": 'Use function calling: finish(final_answer="...") for answer, or run_tools(...) for tools.',
                    },
                }

        # Process tool calls
        tool_results = []
        assistant_msg = {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            # Parse arguments with error handling
            try:
                raw_args = tool_call.function.arguments
                if not raw_args or raw_args.strip() == "":
                    args = {}
                else:
                    args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                wprint(
                    f"Failed to parse tool args: {tool_call.function.arguments}, error: {e}"
                )
                args = {}

            # Execute the meta-tool
            result = await self._execute_meta_tool(tool_name, args, prep_res)

            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

            # Check for completion
            if result.get("completed"):
                return {
                    "completed": True,
                    "answer": result.get("answer", "Task completed."),
                }

        # Return for next iteration
        return {
            "message": assistant_msg,
            "tool_response": tool_results[0]
            if len(tool_results) == 1
            else {
                "role": "user",
                "content": f"Tool results:\n"
                + "\n".join(f"- {tr['content']}" for tr in tool_results),
            },
        }

    async def _execute_meta_tool(
        self, tool_name: str, args: dict, prep_res: dict
    ) -> dict:
        """Execute a meta-tool and return result"""

        # Normalize tool name (remove any prefixes the LLM might add)
        tool_name = tool_name.lower().strip()
        if tool_name.startswith("tool_"):
            tool_name = tool_name[5:]
        if tool_name.startswith("function_"):
            tool_name = tool_name[9:]

        # Map old names to new for compatibility
        name_mapping = {
            "think": "reason",
            "execute_tools": "run_tools",
            "complete": "finish",
            "store_result": "save",
            "get_result": "load",
        }
        tool_name = name_mapping.get(tool_name, tool_name)

        rprint(f"Executing meta-tool: {tool_name} with args: {args}")

        if tool_name == "reason":
            # Internal reasoning - just acknowledge
            thought = args.get("thought", "")
            if not thought:
                return {"status": "error", "message": "No thought provided"}
            self.state.history.append({"type": "thought", "content": thought})
            return {
                "status": "ok",
                "message": "Reasoning noted. Now take action: run_tools or finish.",
            }

        elif tool_name == "run_tools":
            # Delegate to LLMToolNode
            task = args.get("task_description", args.get("task", ""))
            tools = args.get("tool_names", args.get("tools", []))
            if not task:
                return {"status": "error", "message": "No task_description provided"}
            return await self._delegate_to_tool_node(
                {"task": task, "tools": tools}, prep_res
            )

        elif tool_name == "finish":
            # Task completion
            answer = args.get("final_answer", args.get("answer", ""))
            if not answer:
                return {"status": "error", "message": "No final_answer provided"}
            return {"completed": True, "answer": answer}

        elif tool_name == "save":
            # Store in variables
            key = args.get("name", args.get("key", ""))
            value = args.get("data", args.get("value", ""))
            if not key:
                return {"status": "error", "message": "No name/key provided"}
            self.state.results[key] = value
            if self.variable_manager:
                self.variable_manager.set(f"reasoning.{key}", value)
            return {"status": "saved", "name": key}

        elif tool_name == "load":
            # Retrieve from variables
            key = args.get("name", args.get("key", ""))
            if not key:
                return {"status": "error", "message": "No name/key provided"}
            value = self.state.results.get(key)
            if value is None and self.variable_manager:
                value = self.variable_manager.get(f"reasoning.{key}")
            if value is None:
                return {"status": "not_found", "name": key, "value": None}
            return {"status": "ok", "name": key, "value": value}

        else:
            # Unknown tool - try to guess intent
            wprint(f"Unknown meta-tool: {tool_name}, args: {args}")

            # Check if it looks like a finish attempt
            if any(k in args for k in ["answer", "response", "result", "final_answer"]):
                answer = (
                    args.get("answer")
                    or args.get("response")
                    or args.get("result")
                    or args.get("final_answer")
                )
                if answer:
                    return {"completed": True, "answer": str(answer)}

            return {
                "status": "error",
                "message": f"Unknown tool '{tool_name}'. Use: reason, run_tools, finish, save, or load",
            }

    async def _delegate_to_tool_node(self, args: dict, prep_res: dict) -> dict:
        """Delegate task execution to LLMToolNode"""
        task = args.get("task", args.get("task_description", ""))
        tools = args.get("tools", args.get("tool_names", []))

        # Ensure tools is a list
        if isinstance(tools, str):
            tools = [t.strip() for t in tools.split(",")]

        if not task:
            return {
                "status": "error",
                "message": "No task provided. Use task_description parameter.",
            }

        if LLMToolNode is None:
            return {"status": "error", "message": "LLMToolNode not available"}

        # If no specific tools requested, use all available
        available_tools = prep_res.get("available_tools", [])
        if not tools:
            tools = available_tools

        rprint(f"Delegating to LLMToolNode: task='{task[:80]}...', tools={tools[:5]}")

        # Prepare shared state for tool node
        tool_shared = {
            "current_query": task,
            "current_task_description": task,
            "agent_instance": prep_res.get("agent_instance"),
            "variable_manager": prep_res.get("variable_manager"),
            "available_tools": tools if tools else available_tools,
            "tool_capabilities": prep_res.get("tool_capabilities", {}),
            "fast_llm_model": prep_res.get("fast_llm_model"),
            "complex_llm_model": prep_res.get("model"),
            "progress_tracker": prep_res.get("progress_tracker"),
            "session_id": prep_res.get("session_id"),
            "formatted_context": {
                "recent_interaction": f"Reasoning delegation: {task}",
                "task_context": f"Loop {self.state.loop_count}",
            },
        }

        try:
            # Execute tool node
            tool_node = LLMToolNode()
            await tool_node.run_async(tool_shared)

            # Extract results
            response = tool_shared.get("current_response", "No response")
            tool_calls_made = tool_shared.get("tool_calls_made", 0)
            results = tool_shared.get("results", {})

            # Store in our state
            delegation_key = f"delegation_{self.state.loop_count}"
            self.state.results[delegation_key] = {
                "task": task,
                "response": response,
                "tool_calls": tool_calls_made,
            }

            # Store in variables for persistence
            if self.variable_manager:
                self.variable_manager.set(
                    f"delegation.loop_{self.state.loop_count}",
                    {
                        "task": task,
                        "response": response,
                        "results": results,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                self.variable_manager.set(
                    "delegation.latest",
                    {
                        "task_description": task,
                        "final_response": response,
                        "results": results,
                    },
                )

            # Truncate response for context but keep it useful
            truncated_response = response[:3000] if len(response) > 3000 else response

            return {
                "status": "completed",
                "task": task,
                "response": truncated_response,
                "tool_calls_made": tool_calls_made,
                "message": f"Task completed with {tool_calls_made} tool calls.",
            }

        except Exception as e:
            error_msg = str(e)
            eprint(f"Tool delegation failed: {error_msg}")

            # Store error for debugging
            if self.variable_manager:
                self.variable_manager.set(
                    f"delegation.error.loop_{self.state.loop_count}",
                    {
                        "task": task,
                        "error": error_msg,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            return {
                "status": "error",
                "message": f"Tools unavailable or failed: {str(e)}. Use your internal knowledge to answer or explain why it's not possible.",
                "task": task,
            }

    async def _execute_with_manual_parsing(
        self, agent: 'FlowAgent', model: str, messages: list, prep_res: dict
    ) -> dict:
        """Fallback: Manual parsing for models without function calling"""

        # Add instruction for manual tool calling
        manual_instruction = """
Respond with ONE of these actions in EXACTLY this format:

THINK: <your reasoning>

EXECUTE: <task description>
TOOLS: <tool1, tool2>

COMPLETE: <your final answer>

Choose ONE action and respond now."""

        messages_with_instruction = messages + [
            {"role": "user", "content": manual_instruction}
        ]

        response = await agent.a_run_llm_completion(
            model=model,
            messages=messages_with_instruction,
            temperature=0.3,
            node_name="LLMReasonerNode",
            task_id=f"reasoning_loop_{self.state.loop_count}",
        )

        response_text = response if isinstance(response, str) else str(response)

        # Parse response
        if "COMPLETE:" in response_text:
            answer = response_text.split("COMPLETE:")[-1].strip()
            return {"completed": True, "answer": answer}

        elif "EXECUTE:" in response_text:
            # Extract task and tools
            lines = response_text.split("\n")
            task = ""
            tools = []

            for line in lines:
                if line.startswith("EXECUTE:"):
                    task = line.replace("EXECUTE:", "").strip()
                elif line.startswith("TOOLS:"):
                    tools_str = line.replace("TOOLS:", "").strip()
                    tools = [t.strip() for t in tools_str.split(",")]

            if task:
                result = await self._delegate_to_tool_node(
                    {"task": task, "tools": tools}, prep_res
                )
                return {
                    "message": {"role": "assistant", "content": response_text},
                    "tool_response": {
                        "role": "user",
                        "content": f"Execution result: {json.dumps(result, default=str)}",
                    },
                }

        elif "THINK:" in response_text:
            thought = response_text.split("THINK:")[-1].strip()
            self.state.history.append({"type": "thought", "content": thought})
            return {
                "message": {"role": "assistant", "content": response_text},
                "tool_response": {
                    "role": "user",
                    "content": "Thought noted. Now take action (EXECUTE or COMPLETE).",
                },
            }

        # Unrecognized format - treat as potential answer
        if len(response_text) > 100:
            return {"completed": True, "answer": response_text}

        return {
            "message": {"role": "assistant", "content": response_text},
            "tool_response": {
                "role": "user",
                "content": "Please use the specified format: THINK, EXECUTE, or COMPLETE.",
            },
        }

    def _supports_function_calling(self, model: str) -> bool:
        """Check if model supports native function calling"""
        model_lower = model.lower()
        model_base = model_lower.split("/")[-1]

        supported_patterns = [
            "gpt-4", "gpt-3.5",
            "claude-3", "claude-sonnet", "claude-opus",
            "gemini",
            "mistral",
            "command-r",
            "llama-3.1", "llama-3.2", "llama-3.3"
        ]

        for pattern in supported_patterns:
            if pattern in model_base:
                return True

        # Check providers
        if "openai" in model_lower or "anthropic" in model_lower:
            return True

        res = False
        try:
            res = litellm.supports_function_calling(model_base)
        except:
            pass
        return res

    def _create_timeout_response(self, query: str) -> str:
        """Create response when max loops reached"""
        parts = [f"I worked on your request but reached my reasoning limit ({self.max_loops} steps)."]

        if self.state.results:
            parts.append("\nPartial results gathered:")
            for key, value in list(self.state.results.items())[:5]:
                if isinstance(value, dict) and "response" in value:
                    parts.append(f"- {key}: {value['response'][:200]}...")
                else:
                    parts.append(f"- {key}: {str(value)[:200]}...")

        parts.append(f"\nOriginal query: {query}")
        return "\n".join(parts)

    def _error_result(self, message: str) -> dict:
        """Create error result"""
        return {
            "final_result": f"Error: {message}",
            "reasoning_loops": 0,
            "completed": False,
            "error": message
        }

    async def post_async(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store results and update shared state"""
        final_result = exec_res.get("final_result", "No result")

        # Update shared state
        shared["llm_reasoner_result"] = final_result
        shared["current_response"] = final_result
        shared["reasoning_artifacts"] = {
            "loops": exec_res.get("reasoning_loops", 0),
            "completed": exec_res.get("completed", False),
            "results": exec_res.get("results", {})
        }

        # Store in variables
        if self.variable_manager:
            self.variable_manager.set("reasoning.final_result", final_result)
            self.variable_manager.set("reasoning.session_complete", {
                "timestamp": datetime.now().isoformat(),
                "loops": exec_res.get("reasoning_loops", 0),
                "success": exec_res.get("completed", False)
            })

        return "reasoner_complete"


# ===== Foramt Helper =====
class VariableManager:
    """Unified variable management system with advanced features"""

    def __init__(self, world_model: dict, shared_state: dict = None):
        self.world_model = world_model
        self.shared_state = shared_state or {}
        self.scopes = {
            "world": world_model,
            "shared": self.shared_state,
            "results": {},
            "tasks": {},
            "user": {},
            "system": {},
            "reasoning": {},  # For reasoning scope compression
            "files": {},  # For file operation deduplication
            "session_archive": {},  # For large data archiving
        }
        self._cache = {}
        self.agent_instance = None  # Will be set by FlowAgent
        # Optimiertes Caching
        self._path_cache = {}           # path -> (timestamp, value)
        self._scope_hashes = {}         # scope_name -> hash
        self._cache_invalidations = 0   # Für Monitoring
        self.PATH_CACHE_TTL = 5         # 5 Sekunden für path cache

        # LLM Context Cache (falls nicht schon vorhanden)
        self._llm_ctx_cache = {'hash': None, 'content': None}

    def register_scope(self, name: str, data: dict):
        """Register a new variable scope"""
        self.scopes[name] = data
        self._cache.clear()

    def set_results_store(self, results_store: dict):
        """Set the results store for task result references"""
        self.scopes['results'] = results_store
        self._cache.clear()

    def set_tasks_store(self, tasks_store: dict):
        """Set tasks store for task metadata access"""
        self.scopes['tasks'] = tasks_store
        self._cache.clear()

    def _resolve_path(self, path: str):
        """
        Internal helper to navigate a path that can contain both
        dictionary keys and list indices.
        """
        parts = path.split('.')

        # Determine the starting point
        if len(parts) == 1:
            # Simple key in the top-level world_model
            current = self.world_model
        else:
            scope_name = parts[0]
            if scope_name not in self.scopes:
                raise KeyError(f"Scope '{scope_name}' not found")
            current = self.scopes[scope_name]
            parts = parts[1:]  # Continue with the rest of the path

        # Navigate through the parts
        for part in parts:
            if isinstance(current, list):
                try:
                    # It's a list, so the part must be an integer index
                    index = int(part)
                    current = current[index]
                except (ValueError, IndexError):
                    raise KeyError(f"Invalid list index '{part}' in path '{path}'")
            elif isinstance(current, dict):
                try:
                    # It's a dictionary, so the part is a key
                    current = current[part]
                except KeyError:
                    raise KeyError(f"Key '{part}' not found in path '{path}'")
            else:
                # We've hit a non-collection type (int, str, etc.) but the path continues
                raise KeyError(f"Path cannot descend into non-collection type at '{part}' in path '{path}'")

        return current


    def get(self, path: str, default=None, use_cache: bool = True):
        """
        OPTIMIERTE Version - Mit TTL-basiertem Path-Cache.

        Änderung: Cache-Einträge haben TTL, werden nicht sofort invalidiert.
        """
        # Quick check: Einfacher Key ohne Punkt
        if '.' not in path:
            return self.world_model.get(path, default)

        # Cache-Check mit TTL
        if use_cache and hasattr(self, '_path_cache') and path in self._path_cache:
            timestamp, cached_value = self._path_cache[path]
            if time.time() - timestamp < getattr(self, 'PATH_CACHE_TTL', 5):
                return cached_value

        # Auch alten Cache prüfen für Kompatibilität
        if use_cache and path in self._cache:
            return self._cache[path]

        try:
            value = self._resolve_path(path)

            # In beiden Caches speichern
            if use_cache:
                self._cache[path] = value
                if hasattr(self, '_path_cache'):
                    self._path_cache[path] = (time.time(), value)

            return value
        except (KeyError, IndexError):
            return default


    def set(self, path: str, value, create_scope: bool = True):
        """
        OPTIMIERTE Version - Gezieltes Cache-Invalidieren statt clear().

        Vorher: self._cache.clear() bei JEDEM set() → Alle Cache-Einträge weg
        Nachher: Nur betroffene Paths invalidieren
        """
        parts = path.split('.')

        # 1. Betroffene Cache-Einträge invalidieren (NICHT alles!)
        self._invalidate_affected_paths(path)

        # 2. Original Set-Logik
        if len(parts) == 1:
            self.world_model[path] = value
            self._update_scope_hash('world')
            return

        scope_name = parts[0]
        if scope_name not in self.scopes:
            if create_scope:
                self.scopes[scope_name] = {}
            else:
                raise KeyError(f"Scope '{scope_name}' not found")

        current = self.scopes[scope_name]

        # Navigate to parent container
        for i, part in enumerate(parts[1:-1]):
            next_part = parts[i + 2]

            try:
                key = int(part)
                if not isinstance(current, list):
                    raise TypeError(f"Integer index on non-list: {path}")
                while len(current) <= key:
                    current.append(None)
                if current[key] is None:
                    current[key] = [] if next_part.isdigit() else {}
                current = current[key]
            except ValueError:
                key = part
                if not isinstance(current, dict):
                    raise TypeError(f"String key on non-dict: {path}")
                if key not in current:
                    current[key] = [] if next_part.isdigit() else {}
                current = current[key]

        # Final assignment
        last_part = parts[-1]

        if isinstance(current, list):
            try:
                key = int(last_part)
                if key >= len(current):
                    current.append(value)
                else:
                    current[key] = value
            except ValueError:
                current.append(value)
        elif isinstance(current, dict):
            current[last_part] = value
        else:
            raise TypeError(f"Cannot set on {type(current)}: {path}")

        # 3. Scope-Hash aktualisieren für LLM Context Cache
        self._update_scope_hash(scope_name)

        # 4. LLM Context invalidieren NUR bei wichtigen Scopes
        if scope_name in ['results', 'delegation', 'reasoning', 'files', 'user']:
            self._llm_ctx_cache = {'hash': None, 'content': None}


    def _invalidate_affected_paths(self, changed_path: str):
        """
        Invalidiert nur Cache-Einträge die vom geänderten Path betroffen sind.

        Beispiel: Bei Änderung von "results.task_1.data"
        - Invalidiert: "results.task_1.data", "results.task_1", "results"
        - Behält: "delegation.latest", "user.name", etc.
        """
        if not hasattr(self, '_cache'):
            self._cache = {}

        keys_to_remove = []
        changed_parts = changed_path.split('.')

        for cached_path in self._cache:
            cached_parts = cached_path.split('.')

            # Prüfe ob Paths sich überschneiden
            # 1. Geänderter Path ist Prefix von cached: "results" → "results.task_1"
            # 2. Cached Path ist Prefix von geändertem: "results.task_1" → "results.task_1.data"
            min_len = min(len(changed_parts), len(cached_parts))

            if changed_parts[:min_len] == cached_parts[:min_len]:
                keys_to_remove.append(cached_path)

        for key in keys_to_remove:
            del self._cache[key]

        # Tracking für Monitoring
        if hasattr(self, '_cache_invalidations'):
            self._cache_invalidations += len(keys_to_remove)


    def _update_scope_hash(self, scope_name: str):
        """Aktualisiert den Hash eines Scopes für Change-Detection"""
        if not hasattr(self, '_scope_hashes'):
            self._scope_hashes = {}

        scope = self.scopes.get(scope_name, {})
        if isinstance(scope, dict):
            # Schneller Hash: Nur Anzahl Keys und erste 3 Key-Namen
            keys = list(scope.keys())[:3]
            self._scope_hashes[scope_name] = hash((len(scope), tuple(keys)))

    def format_text(self, text: str, context: dict = None) -> str:
        """Enhanced text formatting with multiple syntaxes"""
        if not text or not isinstance(text, str):
            return str(text) if text is not None else ""

        # Temporary context overlay
        if context:
            original_scopes = self.scopes.copy()
            self.scopes['context'] = context

        try:
            # Handle {{ variable }} syntax
            formatted = self._format_double_braces(text)

            # Handle {variable} syntax
            formatted = self._format_single_braces(formatted)

            # Handle $variable syntax
            formatted = self._format_dollar_syntax(formatted)

            return formatted

        finally:
            if context:
                self.scopes = original_scopes

    def _format_double_braces(self, text: str) -> str:
        """Handle {{ variable.path }} syntax with improved debugging"""
        import re

        def replace_var(match):
            var_path = match.group(1).strip()
            value = self.get(var_path)

            if value is None:
                # IMPROVED: Log missing variables for debugging
                available_vars = list(self.get_available_variables().keys())
                wprint(f"Variable '{var_path}' not found. Available: {available_vars[:10]}")
                return match.group(0)  # Keep original if not found

            return self._value_to_string(value)

        return re.sub(r'\{\{\s*([^}]+)\s*\}\}', replace_var, text)

    def _format_single_braces(self, text: str) -> str:
        """Handle {variable.path} syntax, including with spaces like { variable.path }."""
        import re

        def replace_var(match):
            # Extrahiert den Variablennamen und entfernt führende/nachfolgende Leerzeichen
            var_path = match.group(1).strip()

            # Ruft den Wert über die get-Methode ab, die die Punktnotation bereits verarbeitet
            value = self.get(var_path)

            # Gibt den konvertierten Wert oder das Original-Tag zurück, wenn der Wert nicht gefunden wurde
            return self._value_to_string(value) if value is not None else match.group(0)

        # Dieser Regex findet {beliebiger.inhalt} und erlaubt Leerzeichen um den Inhalt
        # Er schließt verschachtelte oder leere Klammern wie {} oder { {var} } aus.
        return re.sub(r'\{([^{}]+)\}', replace_var, text)

    def _format_dollar_syntax(self, text: str) -> str:
        """Handle $variable syntax"""
        import re

        def replace_var(match):
            var_name = match.group(1)
            value = self.get(var_name)
            return self._value_to_string(value) if value is not None else match.group(0)

        return re.sub(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", replace_var, text)

    def _value_to_string(self, value) -> str:
        """Convert value to string representation"""
        if isinstance(value, str):
            return value
        elif isinstance(value, dict | list):
            return json.dumps(value, default=str)
        else:
            return str(value)

    def validate_references(self, text: str) -> dict[str, bool]:
        """Validate all variable references in text"""
        import re

        references = {}

        # Find all {{ }} references
        double_brace_refs = re.findall(r"\{\{\s*([^}]+)\s*\}\}", text)
        for ref in double_brace_refs:
            references["{{" + ref + "}}"] = self.get(ref.strip()) is not None

        # Find all {} references
        single_brace_refs = re.findall(r"\{([^{}\s]+)\}", text)
        for ref in single_brace_refs:
            if "." not in ref:  # Only simple vars
                references["{" + ref + "}"] = self.get(ref.strip()) is not None

        # Find all $ references
        dollar_refs = re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", text)
        for ref in dollar_refs:
            references[f"${ref}"] = self.get(ref) is not None

        return references

    def get_scope_info(self) -> dict[str, Any]:
        """Get information about all available scopes"""
        info = {}
        for scope_name, scope_data in self.scopes.items():
            if isinstance(scope_data, dict):
                info[scope_name] = {
                    "type": "dict",
                    "keys": len(scope_data),
                    "sample_keys": list(scope_data.keys())[:5],
                }
            else:
                info[scope_name] = {
                    "type": type(scope_data).__name__,
                    "value": str(scope_data)[:100],
                }
        return info

    def _validate_task_references(self, task: Task) -> dict[str, Any]:
        """Validate all variable references in a task"""
        validation_results = {"valid": True, "errors": [], "warnings": []}

        # Check different task types
        if isinstance(task, LLMTask):
            if task.prompt_template:
                refs = self.validate_references(task.prompt_template)
                for ref, is_valid in refs.items():
                    if not is_valid:
                        validation_results["errors"].append(
                            f"Invalid reference in prompt: {ref}"
                        )
                        validation_results["valid"] = False

        elif isinstance(task, ToolTask):
            for key, value in task.arguments.items():
                if isinstance(value, str):
                    refs = self.validate_references(value)
                    for ref, is_valid in refs.items():
                        if not is_valid:
                            validation_results["warnings"].append(
                                f"Invalid reference in {key}: {ref}"
                            )

        return validation_results

    def get_variable_suggestions(self, query: str) -> list[str]:
        """Get variable suggestions based on query content"""

        query_lower = query.lower()
        suggestions = []

        # Check all variables for relevance
        for scope in self.scopes.values():
            for name, var_def in scope.items():
                if name in [
                    "system_context",
                    "index",
                    "tool_capabilities",
                    "use_fast_response",
                ]:
                    continue
                # Name similarity
                if any(word in name.lower() for word in query_lower.split()):
                    suggestions.append(name)
                    continue

                # Description similarity
                if isinstance(var_def, pd.DataFrame):
                    var_def_bool = not var_def.empty
                else:
                    var_def_bool = bool(var_def)
                if var_def_bool and any(
                    word in str(var_def).lower() for word in query_lower.split()
                ):
                    suggestions.append(name)
                    continue


        return list(set(suggestions))[:10]

    def _document_structure(self, data: Any, path_prefix: str, docs: dict[str, dict]):
        """A recursive helper to document nested dictionaries and lists."""
        if isinstance(data, dict):
            for key, value in data.items():
                # Construct the full path for the current item
                current_path = f"{path_prefix}.{key}" if path_prefix else key

                # Generate a preview for the value
                if isinstance(value, str):
                    preview = value[:70] + "..." if len(value) > 70 else value
                elif isinstance(value, dict):
                    preview = f"Object with keys: {list(value.keys())[:3]}" + ("..." if len(value.keys()) > 3 else "")
                elif isinstance(value, list):
                    preview = f"List with {len(value)} items"
                else:
                    preview = str(value)

                # Store the documentation for the current path
                docs[current_path] = {
                    'preview': preview,
                    'type': type(value).__name__
                }

                # Recurse into nested structures
                if isinstance(value, dict | list):
                    self._document_structure(value, current_path, docs)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                # Construct the full path for the list item
                current_path = f"{path_prefix}.{i}"

                # Generate a preview for the item
                if isinstance(item, str):
                    preview = item[:70] + "..." if len(item) > 70 else item
                elif isinstance(item, dict):
                    preview = f"Object with keys: {list(item.keys())[:3]}" + ("..." if len(item.keys()) > 3 else "")
                elif isinstance(item, list):
                    preview = f"List with {len(item)} items"
                else:
                    preview = str(item)

                docs[current_path] = {
                    'preview': preview,
                    'type': type(item).__name__
                }

                # Recurse into nested structures
                if isinstance(item, dict | list):
                    self._document_structure(item, current_path, docs)

    def get_available_variables(self) -> dict[str, dict]:
        """
        Recursively documents all available variables from world_model and scopes
        to provide a comprehensive overview for an LLM.
        """
        all_vars_docs = {}

        # 1. Document the world_model (top-level variables)
        # self._document_structure(self.world_model, "", all_vars_docs)

        # 2. Document each scope
        for scope_name, scope_data in self.scopes.items():
            # Add documentation for the scope root itself
            if scope_name == "shared":
                continue
            if isinstance(scope_data, dict):
                scope_data = f"Dict with keys: {list(scope_data.keys())}"
            elif isinstance(scope_data, list):
                scope_data = f"List with {len(scope_data)} items"
            elif isinstance(scope_data, str | int):
                scope_data = f"{scope_data}"[:70]
            else:
                continue

            all_vars_docs[scope_name] = scope_data

            # Recurse into the scope's data
            # self._document_structure(scope_data, scope_name, all_vars_docs)

        return all_vars_docs

    def get_llm_variable_context(self) -> str:
        """
           Ersetzt get_llm_variable_context() im VariableManager.

           Optimierungen:
           1. Cache mit Change-Detection
           2. Nur wichtige Scopes
           3. Keine vollständigen Werte, nur Keys

           Token-Einsparung: 80% (von ~1000 auf ~200 Tokens)
           """

        # Change Detection
        if not hasattr(self, '_llm_ctx_cache'):
            self._llm_ctx_cache = {'hash': None, 'content': None}

        # Hash über Scope-Größen (schnell zu berechnen)
        current_hash = hash(tuple(
            (name, len(data) if isinstance(data, dict) else 0)
            for name, data in self.scopes.items()
            if name not in ['shared', 'system_context']
        ))

        if self._llm_ctx_cache['hash'] == current_hash:
            return self._llm_ctx_cache['content']

        # Minimaler Context
        lines = ["## Variables (access: {{ scope.key }})"]

        priority_scopes = ['results', 'delegation', 'files', 'user', 'reasoning']

        for scope_name in priority_scopes:
            scope = self.scopes.get(scope_name, {})
            if isinstance(scope, dict) and scope:
                keys = list(scope.keys())[:4]
                extra = f" +{len(scope)-4}" if len(scope) > 4 else ""
                lines.append(f"- {scope_name}: {', '.join(keys)}{extra}")

        content = "\n".join(lines)

        self._llm_ctx_cache = {'hash': current_hash, 'content': content}
        return content


    def has_recent_data(self, scope_name: str, key_prefix: str = None) -> bool:
        """
        NEUE METHODE - Schnelle Prüfung ob relevante Daten vorhanden sind.

        Verwendung: Vor teuren Lookups prüfen ob sich der Aufwand lohnt.
        """
        scope = self.scopes.get(scope_name, {})
        if not isinstance(scope, dict) or not scope:
            return False

        if key_prefix:
            return any(k.startswith(key_prefix) for k in scope.keys())

        return True

    def get_latest_delegation(self) -> dict | None:
        """
        NEUE METHODE - Direkter Zugriff auf neueste Delegation.

        Häufigster Lookup - deshalb optimiert.
        """
        # Erst im dedizierten Key schauen
        latest = self.get('delegation.latest')
        if latest:
            return latest

        # Fallback: Suche nach loop_X keys
        delegation_scope = self.scopes.get('delegation', {})
        if not delegation_scope:
            return None

        # Finde höchste Loop-Nummer
        loop_keys = [k for k in delegation_scope.keys() if k.startswith('loop_')]
        if not loop_keys:
            return None

        # Sortiere und nimm neueste
        loop_keys.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0, reverse=True)
        return delegation_scope.get(loop_keys[0])


    def cleanup_old_entries(self, max_entries_per_scope: int = 20):
        """
        NEUE METHODE - Bereinigt alte Einträge um Memory zu sparen.

        Aufruf: Nach jeder 5. Delegation oder bei Memory-Druck.
        """
        cleaned = 0

        for scope_name in ['results', 'delegation', 'reasoning']:
            scope = self.scopes.get(scope_name, {})
            if not isinstance(scope, dict):
                continue

            if len(scope) > max_entries_per_scope:
                # Sortiere nach Key (neuere haben höhere loop-Nummern)
                keys = list(scope.keys())

                # Behalte 'latest' immer
                if 'latest' in keys:
                    keys.remove('latest')

                # Entferne älteste
                keys_to_remove = keys[:-max_entries_per_scope]
                for key in keys_to_remove:
                    # Archiviere vor dem Löschen
                    archive_key = f"cleaned_{scope_name}_{key}"
                    self.scopes['session_archive'][archive_key] = scope[key]
                    del scope[key]
                    cleaned += 1

        if cleaned > 0:
            # Cache invalidieren
            self._cache.clear()
            if hasattr(self, '_path_cache'):
                self._path_cache.clear()

        return cleaned

    def get_quick_summary(self) -> str:
        """
        NEUE METHODE - Ultra-schnelle Zusammenfassung für Status-Checks.

        Verwendung: Wenn nur ein schneller Überblick gebraucht wird.
        ~20-30 Tokens.
        """
        summary = []

        for scope_name in ['results', 'delegation', 'files']:
            scope = self.scopes.get(scope_name, {})
            if isinstance(scope, dict) and scope:
                summary.append(f"{scope_name[0].upper()}:{len(scope)}")

        return " | ".join(summary) if summary else "Empty"

    # ===== AUTO-CLEAN FUNCTIONS =====

    async def auto_compress_reasoning_scope(self) -> dict[str, Any]:
        """
        AUTO-CLEAN FUNCTION 1: LLM-basierte Komprimierung des Reasoning Context

        Analysiert und komprimiert reasoning_context aus LLMReasonerNode:
        - Was hat funktioniert und was nicht
        - Minimale Zusammenfassung und Akkumulation
        - Speichert komprimierte Version und referenziert sie
        - Wird automatisch nach jeder 10. Loop in LLMReasonerNode aufgerufen

        Returns:
            dict mit compression_stats und compressed_data
        """
        try:
            # Zugriff auf reasoning_context aus LLMReasonerNode
            if not self.agent_instance:
                return {"compressed": False, "reason": "no_agent_instance"}

            if not hasattr(self.agent_instance, 'task_flow'):
                return {"compressed": False, "reason": "no_task_flow"}

            if not hasattr(self.agent_instance.task_flow, 'llm_reasoner'):
                return {"compressed": False, "reason": "no_llm_reasoner"}

            llm_reasoner = self.agent_instance.task_flow.llm_reasoner
            if not hasattr(llm_reasoner, 'reasoning_context'):
                return {"compressed": False, "reason": "no_reasoning_context"}

            reasoning_context = llm_reasoner.reasoning_context

            if not reasoning_context or len(reasoning_context) < 10:
                return {"compressed": False, "reason": "context_too_small"}

            # Sammle alle reasoning-relevanten Daten aus der Liste
            raw_data = {
                "reasoning_entries": [e for e in reasoning_context if e.get("type") == "reasoning"],
                "meta_tool_results": [e for e in reasoning_context if e.get("type") == "meta_tool_result"],
                "errors": [e for e in reasoning_context if e.get("type") == "error"],
                "context_summaries": [e for e in reasoning_context if e.get("type") == "context_summary"],
                "total_entries": len(reasoning_context)
            }

            # Berechne Größe vor Komprimierung
            size_before = len(json.dumps(raw_data, default=str))

            # LLM-basierte Analyse und Komprimierung
            if self.agent_instance and LITELLM_AVAILABLE:
                analysis_prompt = f"""Analyze and compress the following reasoning session data.

Raw Data:
{json.dumps(raw_data, indent=2, default=str)[:3000]}...

Create a minimal summary that captures:
1. What worked (successful patterns)
2. What didn't work (failure patterns)
3. Key learnings and insights
4. Important results to keep

Format as JSON:
{{
    "summary": "Brief overall summary",
    "successes": ["pattern1", "pattern2"],
    "failures": ["pattern1", "pattern2"],
    "key_learnings": ["learning1", "learning2"],
    "important_results": {{"key": "value"}}
}}"""

                try:
                    compressed_response = await self.agent_instance.a_llm_call(
                        model=self.agent_instance.amd.fast_llm_model,
                        messages=[{"role": "user", "content": analysis_prompt}],
                        temperature=0.1,
                        node_name="ReasoningCompressor"
                    )

                    # Parse LLM response
                    import re
                    json_match = re.search(r'\{.*\}', compressed_response, re.DOTALL)
                    if json_match:
                        compressed_data = json.loads(json_match.group(0))
                    else:
                        compressed_data = {"summary": compressed_response[:500]}

                except Exception as e:
                    rprint(f"LLM compression failed, using fallback: {e}")
                    compressed_data = self._fallback_reasoning_compression(raw_data)
            else:
                compressed_data = self._fallback_reasoning_compression(raw_data)

            # Speichere komprimierte Version
            timestamp = datetime.now().isoformat()
            compression_entry = {
                "timestamp": timestamp,
                "compressed_data": compressed_data,
                "size_before": size_before,
                "size_after": len(json.dumps(compressed_data, default=str)),
                "compression_ratio": round(len(json.dumps(compressed_data, default=str)) / size_before, 2)
            }

            # Archiviere alte Daten
            archive_key = f"reasoning_archive_{timestamp}"
            self.scopes['session_archive'][archive_key] = {
                "type": "reasoning_compression",
                "original_data": raw_data,
                "compressed_data": compressed_data,
                "metadata": compression_entry
            }

            # Ersetze reasoning_context mit komprimierter Version
            # Behalte nur die letzten 5 Einträge + komprimierte Summary
            recent_entries = reasoning_context[-5:] if len(reasoning_context) > 5 else reasoning_context

            compressed_entry = {
                "type": "compressed_summary",
                "timestamp": timestamp,
                "summary": compressed_data,
                "archive_reference": archive_key,
                "original_entries_count": len(reasoning_context),
                "compression_ratio": compression_entry['compression_ratio']
            }

            # Setze neuen reasoning_context
            llm_reasoner.reasoning_context = [compressed_entry] + recent_entries

            # Speichere auch im reasoning scope für Referenz
            self.scopes['reasoning'] = {
                "compressed": True,
                "last_compression": timestamp,
                "summary": compressed_data,
                "archive_reference": archive_key,
                "entries_before": len(reasoning_context),
                "entries_after": len(llm_reasoner.reasoning_context)
            }

            rprint(f"✅ Reasoning context compressed: {len(reasoning_context)} -> {len(llm_reasoner.reasoning_context)} entries ({compression_entry['compression_ratio']}x size reduction)")

            return {
                "compressed": True,
                "stats": compression_entry,
                "archive_key": archive_key,
                "entries_before": len(reasoning_context),
                "entries_after": len(llm_reasoner.reasoning_context)
            }

        except Exception as e:
            eprint(f"Reasoning compression failed: {e}")
            return {"compressed": False, "error": str(e)}

    def _fallback_reasoning_compression(self, raw_data: dict) -> dict:
        """Fallback compression without LLM"""
        return {
            "summary": f"Compressed {len(raw_data.get('failure_patterns', []))} failures, {len(raw_data.get('successful_patterns', []))} successes",
            "successes": [p.get("query", "")[:50] for p in raw_data.get("successful_patterns", [])[-5:]],
            "failures": [p.get("reason", "")[:50] for p in raw_data.get("failure_patterns", [])[-5:]],
            "key_learnings": ["See archive for details"],
            "important_results": raw_data.get("latest_results", {})
        }

    async def auto_clean(self):
        await asyncio.gather(
            *[
                asyncio.create_task(self.auto_compress_reasoning_scope()),
                asyncio.create_task(self.auto_deduplicate_results_scope()),
            ]
        )

    async def auto_deduplicate_results_scope(self) -> dict[str, Any]:
        """
        AUTO-CLEAN FUNCTION 2: Deduplizierung des Results Scope

        Vereinheitlicht File-Operationen (read_file, write_file, list_dir):
        - Wenn zweimal von derselben Datei gelesen wurde, nur aktuellste Version behalten
        - Beim Schreiben immer nur aktuellste Version im 'files' scope
        - Agent hat immer nur die aktuellste Version
        - Wird nach jeder Delegation aufgerufen

        Returns:
            dict mit deduplication_stats
        """
        try:
            results_scope = self.scopes.get("results", {})
            files_scope = self.scopes.get("files", {})

            if not results_scope:
                return {"deduplicated": False, "reason": "no_results"}

            # Tracking für File-Operationen
            file_operations = {
                "read": {},  # filepath -> [result_ids]
                "write": {},  # filepath -> [result_ids]
                "list": {},  # dirpath -> [result_ids]
            }

            # Analysiere alle Results nach File-Operationen
            for result_id, result_data in results_scope.items():
                if not isinstance(result_data, dict):
                    continue

                # Erkenne File-Operationen
                data = result_data.get("data", {})
                if isinstance(data, dict):
                    # read_file detection
                    if "content" in data and "path" in data:
                        filepath = data.get("path", "")
                        if filepath:
                            if filepath not in file_operations["read"]:
                                file_operations["read"][filepath] = []
                            file_operations["read"][filepath].append(
                                {
                                    "result_id": result_id,
                                    "timestamp": result_data.get("timestamp", ""),
                                    "data": data,
                                }
                            )

                    # write_file detection
                    elif "written" in data or "file_path" in data:
                        filepath = data.get("file_path", data.get("path", ""))
                        if filepath:
                            if filepath not in file_operations["write"]:
                                file_operations["write"][filepath] = []
                            file_operations["write"][filepath].append(
                                {
                                    "result_id": result_id,
                                    "timestamp": result_data.get("timestamp", ""),
                                    "data": data,
                                }
                            )

                    # list_dir detection
                    elif 'files' in data or 'directories' in data:
                        dirpath = data.get('directory', data.get('path', ''))
                        if dirpath:
                            if dirpath not in file_operations['list']:
                                file_operations['list'][dirpath] = []
                            file_operations["list"][dirpath].append(
                                {
                                    "result_id": result_id,
                                    "timestamp": result_data.get("timestamp", ""),
                                    "data": data,
                                }
                            )

            # Deduplizierung: Nur aktuellste Version behalten
            dedup_stats = {
                "files_deduplicated": 0,
                "results_removed": 0,
                "files_unified": 0,
            }

            # Dedupliziere read operations
            for filepath, operations in file_operations["read"].items():
                if len(operations) > 1:
                    # Sortiere nach Timestamp, behalte neueste
                    operations.sort(key=lambda x: x["timestamp"], reverse=True)
                    latest = operations[0]

                    # Speichere im files scope
                    files_scope[filepath] = {
                        'type': 'file_content',
                        'content': latest['data'].get('content', ''),
                        'last_read': latest['timestamp'],
                        'result_id': latest['result_id'],
                        'path': filepath
                    }

                    # Entferne alte Results
                    for old_op in operations[1:]:
                        if old_op['result_id'] in results_scope:
                            # Archiviere statt löschen
                            archive_key = f"archived_read_{old_op['result_id']}"
                            self.scopes['session_archive'][archive_key] = results_scope[old_op['result_id']]
                            del results_scope[old_op['result_id']]
                            dedup_stats['results_removed'] += 1

                    dedup_stats['files_deduplicated'] += 1

            # Dedupliziere write operations
            for filepath, operations in file_operations['write'].items():
                if len(operations) > 1:
                    operations.sort(key=lambda x: x['timestamp'], reverse=True)
                    latest = operations[0]

                    # Update files scope
                    if filepath in files_scope:
                        files_scope[filepath]['last_write'] = latest['timestamp']
                        files_scope[filepath]['write_result_id'] = latest['result_id']

                    # Entferne alte write results
                    for old_op in operations[1:]:
                        if old_op['result_id'] in results_scope:
                            archive_key = f"archived_write_{old_op['result_id']}"
                            self.scopes['session_archive'][archive_key] = results_scope[old_op['result_id']]
                            del results_scope[old_op['result_id']]
                            dedup_stats['results_removed'] += 1

                    dedup_stats['files_deduplicated'] += 1

            # Update scopes
            self.scopes['results'] = results_scope
            self.scopes['files'] = files_scope
            dedup_stats['files_unified'] = len(files_scope)

            if dedup_stats['files_deduplicated'] > 0:
                rprint(f"✅ Results deduplicated: {dedup_stats['files_deduplicated']} files, {dedup_stats['results_removed']} old results archived")

            return {
                "deduplicated": True,
                "stats": dedup_stats
            }

        except Exception as e:
            eprint(f"Results deduplication failed: {e}")
            return {"deduplicated": False, "error": str(e)}

    def get_archived_variable(self, archive_key: str) -> Any:
        """
        Hilfsfunktion zum Abrufen archivierter Variablen

        Args:
            archive_key: Der Archive-Key (z.B. "results.large_file_content")

        Returns:
            Der vollständige Wert der archivierten Variable
        """
        archive_entry = self.scopes.get('session_archive', {}).get(archive_key)
        if archive_entry and isinstance(archive_entry, dict):
            return archive_entry.get('value')
        return None

    def list_archived_variables(self) -> list[dict]:
        """
        Liste alle archivierten Variablen mit Metadaten

        Returns:
            Liste von Dictionaries mit Archive-Informationen
        """
        archived = []
        for key, entry in self.scopes.get('session_archive', {}).items():
            if isinstance(entry, dict) and entry.get('type') == 'large_variable':
                archived.append({
                    'archive_key': key,
                    'original_scope': entry.get('original_scope'),
                    'original_key': entry.get('original_key'),
                    'size': entry.get('size'),
                    'archived_at': entry.get('archived_at'),
                    'preview': str(entry.get('value', ''))[:100] + '...'
                })
        return archived



# ============================================================================
# OPTIMIERTES CACHING FÜR UNIFIED CONTEXT MANAGER
# ============================================================================

class OptimizedUnifiedContextCache:
    """
    Verbesserte Cache-Strategie für UnifiedContextManager.

    Integrieren in __init__:
        self._opt_cache = OptimizedUnifiedContextCache()

    In build_unified_context:
        return await self._opt_cache.get_or_build(session_id, self._build_context_internal)
    """

    def __init__(self):
        self._cache = {}
        self._ttl = {
            "history": 30,  # Chat history
            "variables": 10,  # Variables ändern sich oft
            "execution": 5,  # Execution state sehr dynamisch
        }

    async def get_or_build(
        self, session_id: str, builder_func, context_type: str = "full"
    ) -> dict:
        """
        Intelligentes Caching mit Komponenten-TTLs.
        """

        now = time.time()
        cache_key = f"{session_id}_{context_type}"

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            age = now - cached["timestamp"]

            # Wenn innerhalb kürzester TTL, vollständig aus Cache
            if age < min(self._ttl.values()):
                return cached["data"]

            # Partielles Rebuild basierend auf TTLs
            partial_rebuild = await self._partial_rebuild(
                cached["data"], builder_func, age
            )

            if partial_rebuild:
                self._cache[cache_key] = {"timestamp": now, "data": partial_rebuild}
                return partial_rebuild

        # Vollständiger Build
        data = await builder_func(session_id, context_type)
        self._cache[cache_key] = {"timestamp": now, "data": data}

        return data

    async def _partial_rebuild(
        self, cached_data: dict, builder_func, age: float
    ) -> dict | None:
        """
        Baut nur abgelaufene Komponenten neu.
        """

        rebuilt = cached_data.copy()
        needs_rebuild = []

        for component, ttl in self._ttl.items():
            if age > ttl:
                needs_rebuild.append(component)

        if not needs_rebuild:
            return cached_data

        # Nur spezifische Komponenten neu bauen
        # (Hier würde die Logik pro Komponente kommen)
        return None  # Fallback zu vollständigem Rebuild

    def invalidate(self, session_id: str = None):
        """Gezieltes Invalidieren"""
        if session_id:
            keys_to_remove = [k for k in self._cache if k.startswith(session_id)]
            for k in keys_to_remove:
                del self._cache[k]
        else:
            self._cache.clear()


class UnifiedContextManager:
    """
    Zentrale Orchestrierung aller Context-Quellen für einheitlichen und effizienten Datenzugriff.
    Vereinigt ChatSession, VariableManager, World Model und Task Results.
    """

    def __init__(self, agent):
        self.agent = agent
        self.session_managers: dict[str, Any] = {}  # ChatSession objects
        self.variable_manager: VariableManager = None
        self.compression_threshold = 15  # Messages before compression
        self._context_cache: dict[str, tuple[float, Any]] = {}  # (timestamp, data)
        self.cache_ttl = 300  # 5 minutes
        self._memory_instance = None

        # Granulare Caches mit unterschiedlichen TTLs
        self._history_cache = {}  # session_id -> (timestamp, data)
        self._variables_cache = {}  # session_id -> (timestamp, data)
        self._execution_cache = {}  # session_id -> (timestamp, data)

        # Adaptive TTLs basierend auf Änderungsfrequenz
        self.HISTORY_TTL = 30  # Chat history - relativ stabil
        self.VARIABLES_TTL = 10  # Variables - ändern sich öfter
        self.EXECUTION_TTL = 5  # Execution state - sehr dynamisch

        # Cache-Statistiken für Monitoring
        self._cache_stats = {"hits": 0, "misses": 0}

    async def initialize_session(self, session_id: str, max_history: int = 200):
        """Initialisiere oder lade existierende ChatSession als primäre Context-Quelle"""
        if session_id not in self.session_managers:
            try:
                # Get memory instance
                if not self._memory_instance:
                    from toolboxv2 import get_app

                    self._memory_instance = get_app().get_mod("isaa").get_memory()
                from toolboxv2.mods.isaa.extras.session import ChatSession

                # Create ChatSession as PRIMARY memory source
                session = ChatSession(
                    self._memory_instance,
                    max_length=max_history,
                    space_name=f"ChatSession/{self.agent.amd.name}.{session_id}.unified",
                )
                self.session_managers[session_id] = session

                # Integration mit VariableManager wenn verfügbar
                if self.variable_manager:
                    self.variable_manager.register_scope(
                        f"session_{session_id}",
                        {
                            "chat_session_active": True,
                            "history_length": len(session.history),
                            "last_interaction": None,
                            "session_id": session_id,
                        },
                    )

                rprint(f"Unified session context initialized for {session_id}")
                return session

            except Exception as e:
                eprint(f"Failed to create ChatSession for {session_id}: {e}")
                # Fallback: Create minimal session manager
                self.session_managers[session_id] = {
                    "history": [],
                    "session_id": session_id,
                    "fallback_mode": True,
                }
                return self.session_managers[session_id]

        return self.session_managers[session_id]

    async def add_interaction(
        self, session_id: str, role: str, content: str, metadata: dict = None
    ) -> None:
        """Einheitlicher Weg um Interaktionen in ChatSession zu speichern"""
        session = await self.initialize_session(session_id)

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "metadata": metadata or {},
        }

        # PRIMARY: Store in ChatSession
        if hasattr(session, "add_message"):
            from toolboxv2 import get_app

            get_app().run_bg_task_advanced(session.add_message, message, direct=False)
        elif isinstance(session, dict) and "history" in session:
            # Fallback mode
            session["history"].append(message)
            # Keep max length
            max_len = 200
            if len(session["history"]) > max_len:
                session["history"] = session["history"][-max_len:]

        # SECONDARY: Update VariableManager
        if self.variable_manager:
            self.variable_manager.set(f"session_{session_id}.last_interaction", message)
            if hasattr(session, "history"):
                self.variable_manager.set(
                    f"session_{session_id}.history_length", len(session.history)
                )
            elif isinstance(session, dict):
                self.variable_manager.set(
                    f"session_{session_id}.history_length",
                    len(session.get("history", [])),
                )

        # Clear context cache for this session
        self._invalidate_cache(session_id)

    async def get_contextual_history(
        self, session_id: str, query: str = "", max_entries: int = 10
    ) -> list[dict]:
        """Intelligente Auswahl relevanter Geschichte aus ChatSession"""
        session = self.session_managers.get(session_id)
        if not session:
            return []

        try:
            # ChatSession mode
            if hasattr(session, 'get_past_x'):
                recent_history = session.get_past_x(max_entries, last_u=False)
                c = await session.get_reference(query)
                return recent_history[:max_entries] + ([] if not c else  [{'role': 'system', 'content': c,
                                                        'timestamp': datetime.now().isoformat(), 'metadata': {'source': 'contextual_history'}}] )

            # Fallback mode
            elif isinstance(session, dict) and 'history' in session:
                history = session['history']
                # Return last max_entries, starting with last user message
                result = []
                for msg in reversed(history[-max_entries:]):
                    result.append(msg)
                    if msg.get('role') == 'user' and len(result) >= max_entries:
                        break
                return list(reversed(result))[:max_entries]

        except Exception as e:
            eprint(f"Error getting contextual history: {e}")

        return []


    async def build_unified_context(self, session_id: str, query: str = None, context_type: str = "full") -> dict[str, Any]:
        """
        OPTIMIERTE Version - ersetzt die originale Methode.

        Änderungen:
        1. Query NICHT im Cache-Key (mehr Cache-Hits)
        2. Granulare Caches pro Komponente
        3. Schnellere Execution für häufige Aufrufe
        """

        # Cache-Key OHNE Query für bessere Hit-Rate
        cache_key = f"{session_id}_{context_type}"

        # Vollständiger Cache-Check (kurze TTL für aktive Sessions)
        cached = self._get_cached_context(cache_key)
        if cached:
            self._cache_stats["hits"] = self._cache_stats.get("hits", 0) + 1
            return cached

        self._cache_stats["misses"] = self._cache_stats.get("misses", 0) + 1

        context: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query": query,
            "context_type": context_type,
        }

        current_time = time.time()

        try:
            # 1. CHAT HISTORY - Mit eigenem Cache
            if self._is_component_cache_valid("history", session_id, self.HISTORY_TTL):
                context["chat_history"] = self._history_cache[session_id][1]
            else:
                # Reduziert auf 10 statt 15 für schnellere Builds
                context["chat_history"] = await self.get_contextual_history(
                    session_id, query or "", max_entries=10
                )
                self._history_cache[session_id] = (current_time, context["chat_history"])

            # 2. VARIABLE SYSTEM - Minimal, nur Struktur
            if self._is_component_cache_valid(
                "variables", session_id, self.VARIABLES_TTL
            ):
                context["variables"] = self._variables_cache[session_id][1]
            else:
                context["variables"] = self._build_minimal_variables_snapshot()
                self._variables_cache[session_id] = (current_time, context["variables"])

            # 3. WORLD MODEL - Nur bei Bedarf und Query
            if query and self.variable_manager:
                world_model = self.variable_manager.get("world", {})
                if world_model:
                    # Maximal 3 relevante Facts statt 5
                    context["relevant_facts"] = self._extract_relevant_facts(
                        world_model, query
                    )[:3]

            # 4. EXECUTION STATE - Immer frisch (zu dynamisch für Cache)
            context["execution_state"] = {
                "active_tasks": len(
                    self._get_active_tasks()
                ),  # Nur Anzahl, nicht Details
                "recent_completions": len(self._get_recent_completions(2)),
                "recent_results": self._get_recent_results(2),
                "system_status": self.agent.shared.get("system_status", "idle"),
            }

            # 5. SESSION STATS - Minimal
            context["session_stats"] = {
                "history_length": len(context.get("chat_history", [])),
                "cache_hit_rate": self._get_cache_hit_rate(),
            }

        except Exception as e:
            context["error"] = str(e)
            context["fallback_mode"] = True

        # Cache mit reduzierter TTL für aktive Sessions
        self._cache_context(cache_key, context)
        return context


    def _is_component_cache_valid(self, component: str, session_id: str, ttl: float) -> bool:
        """Prüft ob ein Komponenten-Cache noch gültig ist"""
        cache_map = {
            'history': self._history_cache,
            'variables': self._variables_cache,
            'execution': self._execution_cache
        }

        cache = cache_map.get(component, {})
        if session_id not in cache:
            return False

        timestamp, _ = cache[session_id]
        return time.time() - timestamp < ttl


    def _build_minimal_variables_snapshot(self) -> dict:
        """
        Minimaler Variable-Snapshot - nur Struktur, keine Werte.

        Vorher: Vollständiger YAML-Dump aller Variablen (~1000 Tokens)
        Nachher: Nur Scope-Namen und Key-Counts (~50 Tokens)
        """
        if not self.variable_manager:
            return {'status': 'unavailable'}

        snapshot = {}
        priority_scopes = ['results', 'delegation', 'files', 'user']

        for scope_name in priority_scopes:
            scope = self.variable_manager.scopes.get(scope_name, {})
            if isinstance(scope, dict) and scope:
                snapshot[scope_name] = {
                    'count': len(scope),
                    'keys': list(scope.keys())[:3]  # Nur erste 3 Keys
                }

        return snapshot


    def _get_cache_hit_rate(self) -> float:
        """Berechnet Cache-Hit-Rate für Monitoring"""
        hits = self._cache_stats.get('hits', 0)
        misses = self._cache_stats.get('misses', 0)
        total = hits + misses
        return round(hits / total, 2) if total > 0 else 0.0


    def get_formatted_context_for_llm(self, unified_context: dict[str, Any]) -> str:
        """
        OPTIMIERTE Version - Schärferer, minimaler Context für LLM.

        Vorher: ~800-1500 Tokens mit redundanten Infos
        Nachher: ~200-400 Tokens, nur essentielles
        """
        try:
            parts = []

            # 1. HEADER - Minimal
            session_id = unified_context.get('session_id', '?')[:20]
            parts.append(f"## Context [{session_id}]")

            # 2. CHAT HISTORY - Nur letzte 3 Messages, stark gekürzt
            chat_history = unified_context.get('chat_history', [])
            if chat_history:
                parts.append("\n### Recent")
                for msg in chat_history[-3:]:  # Nur letzte 3 statt 5
                    role = msg.get('role', '?')[0].upper()  # U/A/S
                    content = msg.get('content', '')
                    # Stark kürzen
                    preview = content[:150] + "..." if len(content) > 150 else content
                    parts.append(f"{role}: {preview}")

            # 3. VARIABLES - Nur wenn vorhanden, ultra-kompakt
            variables = unified_context.get('variables', {})
            if variables and variables.get('status') != 'unavailable':
                var_info = []
                for scope, data in variables.items():
                    if isinstance(data, dict) and 'count' in data:
                        var_info.append(f"{scope}({data['count']})")
                if var_info:
                    parts.append(f"\n### Vars: {', '.join(var_info)}")

            # 4. EXECUTION - Nur wenn aktive Tasks
            execution = unified_context.get('execution_state', {})
            active = execution.get('active_tasks', 0)
            if active > 0:
                parts.append(f"\n### Active: {active} tasks")

            # 5. RELEVANT FACTS - Nur wenn vorhanden
            facts = unified_context.get('relevant_facts', [])
            if facts:
                parts.append("\n### Facts")
                for fact in facts[:2]:  # Max 2 Facts
                    if isinstance(fact, (list, tuple)) and len(fact) >= 2:
                        key, value = fact[0], str(fact[1])[:80]
                        parts.append(f"- {key}: {value}")

            return "\n".join(parts)

        except Exception as e:
            return f"Context error: {str(e)}"

    def _extract_relevant_facts(self, world_model: dict, query: str) -> list[tuple[str, Any]]:
        """Extrahiere relevante Facts basierend auf Query"""
        try:
            query_words = set(query.lower().split())
            relevant_facts = []

            for key, value in world_model.items():
                # Simple relevance scoring
                key_words = set(key.lower().split())
                value_words = set(str(value).lower().split())

                # Check for word overlap
                key_overlap = len(query_words.intersection(key_words))
                value_overlap = len(query_words.intersection(value_words))

                if key_overlap > 0 or value_overlap > 0:
                    relevance_score = key_overlap * 2 + value_overlap  # Key matches weighted higher
                    relevant_facts.append((relevance_score, key, value))

            # Sort by relevance and return top facts
            relevant_facts.sort(key=lambda x: x[0], reverse=True)
            return [(key, value) for _, key, value in relevant_facts[:5]]
        except:
            return list(world_model.items())[:5]

    def _get_active_tasks(self) -> list[dict]:
        """Hole aktive Tasks"""
        try:
            tasks = self.agent.shared.get("tasks", {})
            return [
                {"id": task_id, "description": task.description, "status": task.status}
                for task_id, task in tasks.items()
                if task.status == "running"
            ]
        except:
            return []
    def _get_recent_results(self, limit: int = 3) -> list[dict]:
        """
        OPTIMIERTE Version - Weniger Results, kompaktere Previews.

        Vorher: 5 Results mit 150 Char Previews
        Nachher: 3 Results mit 80 Char Previews
        """
        try:
            results_store = self.agent.shared.get("results", {})
            if not results_store:
                return []

            recent_results = []
            # Nur letzte 'limit' Results
            for task_id, result_data in list(results_store.items())[-limit:]:
                if result_data and result_data.get("data"):
                    data = result_data["data"]
                    # Kürzere Preview
                    if isinstance(data, str):
                        preview = data[:80] + "..." if len(data) > 80 else data
                    elif isinstance(data, dict):
                        preview = f"Dict({len(data)} keys)"
                    else:
                        preview = str(data)[:80]

                    recent_results.append({
                        "task_id": task_id[:30],  # Task-ID auch kürzen
                        "preview": preview,
                        "success": result_data.get("metadata", {}).get("success", False)
                    })

            return recent_results
        except:
            return []

    def get_minimal_context_for_reasoning(self, session_id: str) -> str:
        """
        NEUE METHODE - Ultra-minimaler Context speziell für Reasoning Loops.

        Verwendet wenn der Reasoner nur einen kurzen Status braucht.
        ~50-100 Tokens statt ~400-800.
        """
        try:
            parts = []

            # History-Länge
            session = self.session_managers.get(session_id)
            if session:
                if hasattr(session, 'history'):
                    history_len = len(session.history)
                elif isinstance(session, dict):
                    history_len = len(session.get('history', []))
                else:
                    history_len = 0
                parts.append(f"History: {history_len} msgs")

            # Variable Scopes mit Daten
            if self.variable_manager:
                non_empty = []
                for name in ['results', 'delegation', 'files']:
                    scope = self.variable_manager.scopes.get(name, {})
                    if isinstance(scope, dict) and scope:
                        non_empty.append(f"{name}({len(scope)})")
                if non_empty:
                    parts.append(f"Data: {', '.join(non_empty)}")

            # Aktive Tasks
            active_count = len(self._get_active_tasks())
            if active_count > 0:
                parts.append(f"Active: {active_count}")

            return " | ".join(parts) if parts else "No context"

        except:
            return "Context unavailable"

    def _get_recent_completions(self, limit: int = 3) -> list[dict]:
        """Hole recent completions"""
        try:
            tasks = self.agent.shared.get("tasks", {})
            completed = [
                {"id": task_id, "description": task.description, "completed_at": task.completed_at}
                for task_id, task in tasks.items()
                if task.status == "completed" and hasattr(task, 'completed_at') and task.completed_at
            ]
            # Sort by completion time
            completed.sort(key=lambda x: x.get('completed_at', ''), reverse=True)
            return completed[:limit]
        except:
            return []

    def _get_cached_context(self, cache_key: str) -> dict[str, Any] | None:
        """Hole Context aus Cache wenn noch gültig"""
        if cache_key in self._context_cache:
            timestamp, data = self._context_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            else:
                del self._context_cache[cache_key]
        return None

    def _cache_context(self, cache_key: str, context: dict[str, Any]):
        """Speichere Context in Cache"""
        self._context_cache[cache_key] = (time.time(), context.copy())

        # Cleanup old cache entries
        if len(self._context_cache) > 50:  # Keep max 50 entries
            oldest_key = min(self._context_cache.keys(),
                             key=lambda k: self._context_cache[k][0])
            del self._context_cache[oldest_key]

    def _invalidate_cache(self, session_id: str = None):
        """
        OPTIMIERTE Version - Gezieltes Invalidieren statt alles löschen.
        """
        if session_id:
            # Nur spezifische Session invalidieren
            for cache in [self._context_cache, self._history_cache,
                          self._variables_cache, self._execution_cache]:
                keys_to_remove = [k for k in cache if session_id in str(k)]
                for key in keys_to_remove:
                    del cache[key]
        else:
            # Alles invalidieren (selten nötig)
            self._context_cache.clear()
            self._history_cache.clear()
            self._variables_cache.clear()
            self._execution_cache.clear()

    def get_session_statistics(self) -> dict[str, Any]:
        """Hole Statistiken über alle Sessions"""
        stats = {
            "total_sessions": len(self.session_managers),
            "active_sessions": [],
            "cache_entries": len(self._context_cache),
            "cache_hit_rate": 0.0  # Could be tracked if needed
        }

        for session_id, session in self.session_managers.items():
            session_info = {
                "session_id": session_id,
                "fallback_mode": isinstance(session, dict) and session.get('fallback_mode', False)
            }

            if hasattr(session, 'history'):
                session_info["message_count"] = len(session.history)
            elif isinstance(session, dict) and 'history' in session:
                session_info["message_count"] = len(session['history'])

            stats["active_sessions"].append(session_info)

        return stats

    async def cleanup_old_sessions(self, max_age_hours: int = 168) -> int:
        """Cleanup alte Sessions (default: 1 Woche)"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            removed_count = 0

            sessions_to_remove = []
            for session_id, session in self.session_managers.items():
                should_remove = False

                # Check last activity
                if hasattr(session, 'history') and session.history:
                    last_msg = session.history[-1]
                    last_timestamp = last_msg.get('timestamp')
                    if last_timestamp:
                        try:
                            last_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                            if last_time < cutoff_time:
                                should_remove = True
                        except:
                            pass
                elif isinstance(session, dict) and session.get('history'):
                    last_msg = session['history'][-1]
                    last_timestamp = last_msg.get('timestamp')
                    if last_timestamp:
                        try:
                            last_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                            if last_time < cutoff_time:
                                should_remove = True
                        except:
                            pass

                if should_remove:
                    sessions_to_remove.append(session_id)

            # Remove old sessions
            for session_id in sessions_to_remove:
                session = self.session_managers[session_id]
                if hasattr(session, 'on_exit'):
                    session.on_exit()  # Save ChatSession data
                del self.session_managers[session_id]
                removed_count += 1

                # Remove from variable manager
                if self.variable_manager:
                    scope_name = f'session_{session_id}'
                    if scope_name in self.variable_manager.scopes:
                        del self.variable_manager.scopes[scope_name]

            # Clear related cache entries
            self._invalidate_cache()

            return removed_count
        except Exception as e:
            eprint(f"Error cleaning up old sessions: {e}")
            return 0

# ===== VOTING ======

import os
import asyncio
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ===== PYDANTIC MODELS FOR STRUCTURED VOTING =====

class VotingMode(str, Enum):
    """Voting mode types"""
    SIMPLE = "simple"
    ADVANCED = "advanced"
    UNSTRUCTURED = "unstructured"


class VotingStrategy(str, Enum):
    """Strategy for advanced voting"""
    BEST = "best"
    VOTE = "vote"
    RECOMBINE = "recombine"


class SimpleVoteResult(BaseModel):
    """Result of a simple vote"""
    option: str = Field(description="The voted option")
    reasoning: Optional[str] = Field(default=None, description="Optional reasoning for the vote")


class ThinkingResult(BaseModel):
    """Result from a thinking/analysis phase"""
    analysis: str = Field(description="The analysis or thinking result")
    key_points: list[str] = Field(description="Key points extracted")
    quality_score: float = Field(description="Self-assessed quality score 0-1", ge=0, le=1)


class OrganizedData(BaseModel):
    """Organized structure from unstructured data"""
    structure: dict[str, Any] = Field(description="The organized data structure")
    categories: list[str] = Field(description="Identified categories")
    parts: list[dict[str, str]] = Field(description="Individual parts with id and content")
    quality_score: float = Field(description="Organization quality 0-1", ge=0, le=1)


class VoteSelection(BaseModel):
    """Selection of best item from voting"""
    selected_id: str = Field(description="ID of selected item")
    reasoning: str = Field(description="Why this item was selected")
    confidence: float = Field(description="Confidence in selection 0-1", ge=0, le=1)


class FinalConstruction(BaseModel):
    """Final constructed output"""
    output: str = Field(description="The final constructed output")
    sources_used: list[str] = Field(description="IDs of sources used in construction")
    synthesis_notes: str = Field(description="How sources were synthesized")


class VotingResult(BaseModel):
    """Complete voting result"""
    mode: VotingMode
    winner: str
    votes: int
    margin: int
    k_margin: int
    total_votes: int
    reached_k_margin: bool
    details: dict[str, Any] = Field(default_factory=dict)
    cost_info: dict[str, float] = Field(default_factory=dict)

# ===== MAIN AGENT CLASS =====
class FlowAgent:
    """Production-ready agent system built on PocketFlow """
    def __init__(
        self,
        amd: AgentModelData,
        world_model: dict[str, Any] = None,
        verbose: bool = False,
        enable_pause_resume: bool = True,
        checkpoint_interval: int = 300,  # 5 minutes
        max_parallel_tasks: int = 3,
        progress_callback: callable = None,
        stream:bool=True,
        **kwargs
    ):
        self.amd = amd
        self.stream = stream
        self.world_model = world_model or {}
        self.verbose = verbose
        self.enable_pause_resume = enable_pause_resume
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_config = CheckpointConfig()
        self.max_parallel_tasks = max_parallel_tasks
        self.progress_tracker = ProgressTracker(progress_callback, agent_name=amd.name)

        # Core state
        self.shared = {
            "world_model": self.world_model,
            "tasks": {},
            "task_plans": {},
            "system_status": "idle",
            "session_data": {},
            "performance_metrics": {},
            "conversation_history": [],
            "available_tools": [],
            "progress_tracker": self.progress_tracker,
        }
        self.context_manager = UnifiedContextManager(self)
        self.variable_manager = VariableManager(self.shared["world_model"], self.shared)
        self.variable_manager.agent_instance = (
            self  # Set agent reference for auto-clean functions
        )
        self.context_manager.variable_manager = (
            self.variable_manager
        )  # Register default scopes

        self.shared["context_manager"] = self.context_manager
        self.shared["variable_manager"] = self.variable_manager
        # Flows
        self.task_flow = TaskManagementFlow(max_parallel_tasks=self.max_parallel_tasks)

        if hasattr(self.task_flow, "executor_node"):
            self.task_flow.executor_node.agent_instance = self

        # Agent state
        self.is_running = False
        self.is_paused = False
        self.last_checkpoint = None

        # Token and cost tracking (persistent across runs)
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost_accumulated = 0.0
        self.total_llm_calls = 0
        self.checkpoint_data = {}
        self.ac_cost = 0

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_tasks)
        self._shutdown_event = threading.Event()

        # Server components
        self.a2a_server: A2AServer = None
        self.mcp_server: FastMCP = None

        # Enhanced tool registry
        self._tool_registry = {}
        self._all_tool_capabilities = {}
        self._tool_capabilities = {}
        self._tool_analysis_cache = {}

        self.active_session = None
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

        # Load tool analysis - will be filtered to active tools during setup
        # self._tool_capabilities.update(self._load_tool_analysis())
        if self.amd.budget_manager:
            self.amd.budget_manager.load_data()

        self._setup_variable_scopes()

        rprint(f"FlowAgent initialized: {amd.name}")

    def task_flow_settings(self, max_parallel_tasks: int = 3, max_reasoning_loops: int = 24, max_tool_calls:int = 5):
        self.task_flow.executor_node.max_parallel = max_parallel_tasks
        self.task_flow.llm_reasoner.max_reasoning_loops = max_reasoning_loops
        self.task_flow.llm_tool_node.max_tool_calls = max_tool_calls

    @property
    def progress_callback(self):
        return self.progress_tracker.progress_callback

    @progress_callback.setter
    def progress_callback(self, value):
        self.progress_tracker.progress_callback = value

    def set_progress_callback(self, progress_callback: callable = None):
        self.progress_callback = progress_callback

    def sanitize_message_history(self, messages: list[dict]) -> list[dict]:
        """
        Sanitize message history to ensure tool call/response pairs are complete.

        This prevents LiteLLM errors like:
        "Missing corresponding tool call for tool response message"

        Rules:
        1. Every 'role: tool' message MUST have a preceding 'role: assistant' message
           with tool_calls containing the matching tool_call_id
        2. If orphaned tool response found → remove it
        3. If assistant has tool_calls but no tool responses follow → remove the tool_calls

        Returns:
            Sanitized message list safe for all LLM providers
        """
        if not messages:
            return messages

        sanitized = []
        pending_tool_calls = {}  # tool_call_id -> assistant_message_index

        for msg in messages:
            role = msg.get('role', '')

            if role == 'assistant':
                # Track tool_calls if present
                tool_calls = msg.get('tool_calls', [])
                if tool_calls:
                    for tc in tool_calls:
                        tc_id = tc.get('id') or tc.get('tool_call_id')
                        if tc_id:
                            pending_tool_calls[tc_id] = len(sanitized)
                sanitized.append(msg)

            elif role == 'tool':
                # Check if we have the corresponding tool_call
                tool_call_id = msg.get('tool_call_id')
                if tool_call_id and tool_call_id in pending_tool_calls:
                    sanitized.append(msg)
                    del pending_tool_calls[tool_call_id]
                else:
                    # ORPHANED TOOL RESPONSE - skip it
                    print(f"⚠️ [SANITIZE] Removing orphaned tool response: {tool_call_id}")
                    continue
            else:
                sanitized.append(msg)

        # Clean up assistant messages with unmatched tool_calls at the end
        # (tool_calls that never got responses)
        if pending_tool_calls:
            indices_to_clean = set(pending_tool_calls.values())
            for idx in indices_to_clean:
                if idx < len(sanitized):
                    msg = sanitized[idx]
                    if msg.get('tool_calls'):
                        # Remove tool_calls from this message or convert to regular assistant
                        print(f"⚠️ [SANITIZE] Removing unmatched tool_calls from assistant message at index {idx}")
                        msg_copy = msg.copy()
                        del msg_copy['tool_calls']
                        if msg_copy.get('content'):
                            sanitized[idx] = msg_copy
                        else:
                            # Mark for removal if no content
                            sanitized[idx] = None

            # Remove None entries (empty assistant messages)
            sanitized = [m for m in sanitized if m is not None]

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

    async def a_run_llm_completion(self, node_name="FlowAgentLLMCall",task_id="unknown",model_preference="fast", with_context=True, auto_fallbacks=False, llm_kwargs=None, get_response_message=False,**kwargs) -> str:
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
            # Sanitize message history to prevent tool call/response pair corruption
            kwargs["messages"] = self.sanitize_message_history(kwargs["messages"])

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
                    await self.context_manager.get_contextual_history(self.active_session)
                    if self.context_manager
                    else ""
                )
                kwargs["messages"] = [
                    {
                        "role": "system",
                        "content": self.amd.get_system_message_with_persona()
                        + f"\n\nLast: {str(last)[:300]}",
                    }
                ] + kwargs.get("messages", [])

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

        # build fallback dict using FALLBACKS_MODELS/PREM and _KEYS

        if auto_fallbacks and 'fallbacks' not in kwargs:
            fallbacks_dict_list = []
            fallbacks = os.getenv("FALLBACKS_MODELS", '').split(',') if model_preference == "fast" else os.getenv(
                "FALLBACKS_MODELS_PREM", '').split(',')
            fallbacks_keys = os.getenv("FALLBACKS_MODELS_KEYS", '').split(
                ',') if model_preference == "fast" else os.getenv(
                "FALLBACKS_MODELS_KEYS_PREM", '').split(',')
            for model, key in zip(fallbacks, fallbacks_keys):
                fallbacks_dict_list.append({"model": model, "api_key": os.getenv(key, kwargs.get("api_key", None))})
            kwargs['fallbacks'] = fallbacks_dict_list

        try:
            # P1 - HOCH: LLM Rate Limiting to prevent cost explosions

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
            self.ac_cost += call_cost

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
        user_id: str = None,
        stream_callback: Callable = None,
        remember: bool = True,
        as_callback: Callable = None,
        fast_run: bool = False,
        **kwargs
    ) -> str:
        """Main entry point für Agent-Ausführung mit UnifiedContextManager

        Args:
            query: Die Benutzeranfrage (kann [media:(path/url)] Tags enthalten)
            session_id: Session-ID für Kontext-Management
            user_id: Benutzer-ID
            stream_callback: Callback für Streaming-Antworten
            remember: Ob die Interaktion gespeichert werden soll
            as_callback: Optional - Callback-Funktion für Echtzeit-Kontext-Injektion
            fast_run: Optional - Überspringt detaillierte Outline-Phase für schnelle Antworten
            **kwargs: Zusätzliche Argumente (kann llm_kwargs enthalten)

        Note:
            Media-Tags im Format [media:(path/url)] werden automatisch geparst und
            an das LLM als Multi-Modal-Input übergeben.
        """

        execution_start = self.progress_tracker.start_timer("total_execution")
        self.active_session = session_id
        self.resent_tools_called = []
        result = None

        await self.progress_tracker.emit_event(ProgressEvent(
            event_type="execution_start",
            timestamp=time.time(),
            status=NodeStatus.RUNNING,
            node_name="FlowAgent",
            session_id=session_id,
            metadata={"query": query, "user_id": user_id, "fast_run": fast_run, "has_callback": as_callback is not None}
        ))

        try:
            #Initialize or get session über UnifiedContextManager
            await self.initialize_session_context(session_id, max_history=200)

            #Store user message immediately in ChatSession wenn remember=True
            if remember:
                await self.context_manager.add_interaction(
                    session_id,
                    'user',
                    query,
                    metadata={"user_id": user_id}
                )

            # Set user context variables
            timestamp = datetime.now()
            self.variable_manager.register_scope('user', {
                'id': user_id,
                'session': session_id,
                'query': query,
                'timestamp': timestamp.isoformat()
            })

            # Update system variables
            self.variable_manager.set('system_context.timestamp', {'isoformat': timestamp.isoformat()})
            self.variable_manager.set('system_context.current_session', session_id)
            self.variable_manager.set('system_context.current_user', user_id)
            self.variable_manager.set('system_context.last_query', query)

            # Initialize with tool awareness
            await self.initialize_context_awareness()

            # VEREINFACHT: Prepare execution context - weniger Daten duplizieren
            self.shared.update({
                "current_query": query,
                "session_id": session_id,
                "user_id": user_id,
                "stream_callback": stream_callback,
                "remember": remember,
                # CENTRAL: Context Manager ist die primäre Context-Quelle
                "context_manager": self.context_manager,
                "variable_manager": self.variable_manager,
                "fast_run": fast_run,  # fast_run-Flag übergeben
            })

            # --- Neu: as_callback behandeln ---
            if as_callback:
                self.shared['callback_context'] = {
                    'callback_timestamp': datetime.now().isoformat(),
                    'callback_name': getattr(as_callback, '__name__', 'unnamed_callback'),
                    'initial_query': query
                }
            # --------------------------------

            # Set LLM models in shared context
            self.shared['fast_llm_model'] = self.amd.fast_llm_model
            self.shared['complex_llm_model'] = self.amd.complex_llm_model
            self.shared['persona_config'] = self.amd.persona
            self.shared['use_fast_response'] = self.amd.use_fast_response

            await self.variable_manager.auto_clean()

            # Set system status
            self.shared["system_status"] = "running"
            self.is_running = True

            # Execute main orchestration flow
            result = await self._orchestrate_execution()

            #Store assistant response in ChatSession wenn remember=True
            if remember:
                await self.context_manager.add_interaction(
                    session_id,
                    'assistant',
                    result,
                    metadata={"user_id": user_id, "execution_duration": time.time() - execution_start}
                )

            total_duration = self.progress_tracker.end_timer("total_execution")

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

            # Checkpoint if needed
            if self.enable_pause_resume:
                with Spinner("Creating checkpoint..."):
                    await self._maybe_checkpoint()
            return result

        except Exception as e:
            eprint(f"Agent execution failed: {e}", exc_info=True)
            error_response = f"I encountered an error: {str(e)}"
            result = error_response
            import traceback
            print(traceback.format_exc())

            # Store error in ChatSession wenn remember=True
            if remember:
                await self.context_manager.add_interaction(
                    session_id,
                    'assistant',
                    error_response,
                    metadata={
                        "user_id": user_id,
                        "error": True,
                        "error_type": type(e).__name__
                    }
                )

            total_duration = self.progress_tracker.end_timer("total_execution")

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
            self.shared["system_status"] = "idle"
            self.is_running = False
            self.active_session = None

    def set_response_format(
        self,
        response_format: str,
        text_length: str,
        custom_instructions: str = "",
        quality_threshold: float = 0.7
    ):
        """Dynamische Format- und Längen-Konfiguration"""

        # Validiere Eingaben
        try:
            ResponseFormat(response_format)
            TextLength(text_length)
        except ValueError:
            available_formats = [f.value for f in ResponseFormat]
            available_lengths = [l.value for l in TextLength]
            raise ValueError(
                f"Invalid format or length. "
                f"Available formats: {available_formats}. "
                f"Available lengths: {available_lengths}"
            )

        # Erstelle oder aktualisiere Persona
        if not self.amd.persona:
            self.amd.persona = PersonaConfig(name="Assistant")

        # Erstelle Format-Konfiguration
        format_config = FormatConfig(
            response_format=ResponseFormat(response_format),
            text_length=TextLength(text_length),
            custom_instructions=custom_instructions,
            quality_threshold=quality_threshold
        )

        self.amd.persona.format_config = format_config

        # Aktualisiere Personality Traits mit Format-Hinweisen
        self._update_persona_with_format(response_format, text_length)

        # Update shared state
        self.shared["persona_config"] = self.amd.persona
        self.shared["format_config"] = format_config

        rprint(f"Response format set: {response_format}, length: {text_length}")

    def _update_persona_with_format(self, response_format: str, text_length: str):
        """Aktualisiere Persona-Traits basierend auf Format"""

        # Format-spezifische Traits
        format_traits = {
            "with-tables": ["structured", "data-oriented", "analytical"],
            "with-bullet-points": ["organized", "clear", "systematic"],
            "with-lists": ["methodical", "sequential", "thorough"],
            "md-text": ["technical", "formatted", "detailed"],
            "yaml-text": ["structured", "machine-readable", "precise"],
            "json-text": ["technical", "API-focused", "structured"],
            "text-only": ["conversational", "natural", "flowing"],
            "pseudo-code": ["logical", "algorithmic", "step-by-step"],
            "code-structure": ["technical", "systematic", "hierarchical"]
        }

        # Längen-spezifische Traits
        length_traits = {
            "mini-chat": ["concise", "quick", "to-the-point"],
            "chat-conversation": ["conversational", "friendly", "balanced"],
            "table-conversation": ["structured", "comparative", "organized"],
            "detailed-indepth": ["thorough", "comprehensive", "analytical"],
            "phd-level": ["academic", "scholarly", "authoritative"]
        }

        # Kombiniere Traits
        current_traits = set(self.amd.persona.personality_traits)

        # Entferne alte Format-Traits
        old_format_traits = set()
        for traits in format_traits.values():
            old_format_traits.update(traits)
        for traits in length_traits.values():
            old_format_traits.update(traits)

        current_traits -= old_format_traits

        # Füge neue Traits hinzu
        new_traits = format_traits.get(response_format, [])
        new_traits.extend(length_traits.get(text_length, []))

        current_traits.update(new_traits)
        self.amd.persona.personality_traits = list(current_traits)

    def get_available_formats(self) -> dict[str, list[str]]:
        """Erhalte verfügbare Format- und Längen-Optionen"""
        return {
            "formats": [f.value for f in ResponseFormat],
            "lengths": [l.value for l in TextLength],
            "format_descriptions": {
                f.value: FormatConfig(response_format=f).get_format_instructions()
                for f in ResponseFormat
            },
            "length_descriptions": {
                l.value: FormatConfig(text_length=l).get_length_instructions()
                for l in TextLength
            }
        }

    async def a_run_with_format(
        self,
        query: str,
        response_format: str = "frei-text",
        text_length: str = "chat-conversation",
        custom_instructions: str = "",
        **kwargs
    ) -> str:
        """Führe Agent mit spezifischem Format aus"""

        # Temporäre Format-Einstellung
        original_persona = self.amd.persona

        try:
            self.set_response_format(response_format, text_length, custom_instructions)
            response = await self.a_run(query, **kwargs)
            return response
        finally:
            # Restore original persona
            self.amd.persona = original_persona
            self.shared["persona_config"] = original_persona

    def get_format_quality_report(self) -> dict[str, Any]:
        """Erhalte detaillierten Format-Qualitätsbericht"""
        quality_assessment = self.shared.get("quality_assessment", {})

        if not quality_assessment:
            return {"status": "no_assessment", "message": "No recent quality assessment available"}

        quality_details = quality_assessment.get("quality_details", {})

        return {
            "overall_score": quality_details.get("total_score", 0.0),
            "format_adherence": quality_details.get("format_adherence", 0.0),
            "length_adherence": quality_details.get("length_adherence", 0.0),
            "content_quality": quality_details.get("base_quality", 0.0),
            "llm_assessment": quality_details.get("llm_assessment", 0.0),
            "suggestions": quality_assessment.get("suggestions", []),
            "assessment": quality_assessment.get("quality_assessment", "unknown"),
            "format_config_active": quality_details.get("format_config_used", False)
        }

    def get_variable_documentation(self) -> str:
        """Get comprehensive variable system documentation"""
        docs = []
        docs.append("# Variable System Documentation\n")

        # Available scopes
        docs.append("## Available Scopes:")
        scope_info = self.variable_manager.get_scope_info()
        for scope_name, info in scope_info.items():
            docs.append(f"- `{scope_name}`: {info['type']} with {info.get('keys', 'N/A')} keys")

        docs.append("\n## Syntax Options:")
        docs.append("- `{{ variable.path }}` - Full path resolution")
        docs.append("- `{variable}` - Simple variable (no dots)")
        docs.append("- `$variable` - Shell-style variable")

        docs.append("\n## Example Usage:")
        docs.append("- `{{ results.task_1.data }}` - Get result from task_1")
        docs.append("- `{{ user.name }}` - Get user name")
        docs.append("- `{agent_name}` - Simple agent name")
        docs.append("- `$timestamp` - System timestamp")

        # Available variables
        docs.append("\n## Available Variables:")
        variables = self.variable_manager.get_available_variables()
        for scope_name, scope_vars in variables.items():
            docs.append(f"\n### {scope_name}:")
            for _var_name, var_info in scope_vars.items():
                docs.append(f"- `{var_info['path']}`: {var_info['preview']} ({var_info['type']})")

        return "\n".join(docs)

    def _setup_variable_scopes(self):
        """Setup default variable scopes with enhanced structure"""
        self.variable_manager.register_scope('agent', {
            'name': self.amd.name,
            'model_fast': self.amd.fast_llm_model,
            'model_complex': self.amd.complex_llm_model
        })

        timestamp = datetime.now()
        self.variable_manager.register_scope('system', {
            'timestamp': timestamp.isoformat(),
            'version': '2.0',
            'capabilities': list(self._tool_capabilities.keys())
        })

        # ADDED: Initialize empty results and tasks scopes
        self.variable_manager.register_scope('results', {})
        self.variable_manager.register_scope('tasks', {})

        # Update shared state
        self.shared["variable_manager"] = self.variable_manager

    def set_variable(self, path: str, value: Any):
        """Set variable using unified system"""
        self.variable_manager.set(path, value)

    def get_variable(self, path: str, default=None):
        """Get variable using unified system"""
        return self.variable_manager.get(path, default)

    def format_text(self, text: str, **context) -> str:
        """Format text with variables"""
        return self.variable_manager.format_text(text, context)

    async def initialize_session_context(self, session_id: str = "default", max_history: int = 200) -> bool:
        """Vereinfachte Session-Initialisierung über UnifiedContextManager"""
        try:
            # Delegation an UnifiedContextManager
            session = await self.context_manager.initialize_session(session_id, max_history)

            # Ensure Variable Manager integration
            if not self.context_manager.variable_manager:
                self.context_manager.variable_manager = self.variable_manager

            # Update shared state (minimal - primary data now in context_manager)
            self.shared["active_session_id"] = session_id
            self.shared["session_initialized"] = True

            # Legacy support: Keep session_managers reference in shared for backward compatibility
            self.shared["session_managers"] = self.context_manager.session_managers

            rprint(f"Session context initialized for {session_id} via UnifiedContextManager")
            return True

        except Exception as e:
            eprint(f"Session context initialization failed: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    async def initialize_context_awareness(self):
        """Enhanced context awareness with session management"""

        # Initialize session if not already done
        session_id = self.shared.get("session_id", self.active_session)
        if not self.shared.get("session_initialized"):
            await self.initialize_session_context(session_id)

        # Ensure tool capabilities are loaded
        # add tqdm prigress bar

        from tqdm import tqdm

        if hasattr(self.task_flow, 'llm_reasoner'):
            if "read_from_variables" not in self.shared["available_tools"] and hasattr(self.task_flow.llm_reasoner, '_execute_read_from_variables'):
                await self.add_tool(lambda scope, key, purpose: self.task_flow.llm_reasoner._execute_read_from_variables({"scope": scope, "key": key, "purpose": purpose}), "read_from_variables", "Read from variables")
            if "write_to_variables" not in self.shared["available_tools"] and hasattr(self.task_flow.llm_reasoner, '_execute_write_to_variables'):
                await self.add_tool(lambda scope, key, value, description: self.task_flow.llm_reasoner._execute_write_to_variables({"scope": scope, "key": key, "value": value, "description": description}), "write_to_variables", "Write to variables")

            if "internal_reasoning" not in self.shared["available_tools"] and hasattr(self.task_flow.llm_reasoner, '_execute_internal_reasoning'):
                async def internal_reasoning_tool(thought:str, thought_number:int, total_thoughts:int, next_thought_needed:bool, current_focus:str, key_insights:list[str], potential_issues:list[str], confidence_level:float):
                    args = {
                        "thought": thought,
                        "thought_number": thought_number,
                        "total_thoughts": total_thoughts,
                        "next_thought_needed": next_thought_needed,
                        "current_focus": current_focus,
                        "key_insights": key_insights,
                        "potential_issues": potential_issues,
                        "confidence_level": confidence_level
                    }
                    return await self.task_flow.llm_reasoner._execute_internal_reasoning(args, self.shared)
                await self.add_tool(internal_reasoning_tool, "internal_reasoning", "Internal reasoning")

            if "manage_internal_task_stack" not in self.shared["available_tools"] and hasattr(self.task_flow.llm_reasoner, '_execute_manage_task_stack'):
                async def manage_internal_task_stack_tool(action:str, task_description:str, outline_step_ref:str):
                    args = {
                        "action": action,
                        "task_description": task_description,
                        "outline_step_ref": outline_step_ref
                    }
                    return await self.task_flow.llm_reasoner._execute_manage_task_stack(args, self.shared)
                await self.add_tool(manage_internal_task_stack_tool, "manage_internal_task_stack", "Manage internal task stack")

            if "outline_step_completion" not in self.shared["available_tools"] and hasattr(self.task_flow.llm_reasoner, '_execute_outline_step_completion'):
                async def outline_step_completion_tool(step_completed:bool, completion_evidence:str, next_step_focus:str):
                    args = {
                        "step_completed": step_completed,
                        "completion_evidence": completion_evidence,
                        "next_step_focus": next_step_focus
                    }
                    return await self.task_flow.llm_reasoner._execute_outline_step_completion(args, self.shared)
                await self.add_tool(outline_step_completion_tool, "outline_step_completion", "Outline step completion")


        registered_tools = set(self._tool_registry.keys())
        cached_capabilities = list(self._tool_capabilities.keys())  # Create a copy of

        # Remove capabilities for tools that are no longer registered
        for tool_name in cached_capabilities:
            if tool_name in self._tool_capabilities and tool_name not in registered_tools:
                del self._tool_capabilities[tool_name]
                iprint(f"Removed outdated capability for unavailable tool: {tool_name}")

        # Collect tools that need analysis
        tools_to_analyze = []
        for tool_name in self.shared["available_tools"]:
            if tool_name not in self._tool_capabilities:
                tool_info = self._tool_registry.get(tool_name, {})
                tools_to_analyze.append({
                    "name": tool_name,
                    "description": tool_info.get("description", "No description"),
                    "args_schema": tool_info.get("args_schema", "()")
                })

        # Batch analyze tools if there are any to analyze
        if tools_to_analyze:
            if len(tools_to_analyze) <= 3:
                # For small batches, analyze individually for better quality
                for tool_data in tqdm(tools_to_analyze, desc=f"Agent {self.amd.name} Analyzing Tools", unit="tool", colour="green"):
                    with Spinner(f"Analyzing tool {tool_data['name']}"):
                        await self._analyze_tool_capabilities(tool_data['name'], tool_data['description'], tool_data['args_schema'])
            else:
                # For larger batches, use batch analysis
                with Spinner(f"Batch analyzing {len(tools_to_analyze)} tools"):
                    await self._batch_analyze_tool_capabilities(tools_to_analyze)

        # Update args_schema for all registered tools
        for tool_name in self.shared["available_tools"]:
            if tool_name in self._tool_capabilities:
                function = self._tool_registry[tool_name]["function"]
                if not isinstance(self._tool_capabilities[tool_name], dict):
                    self._tool_capabilities[tool_name] = {}
                self._tool_capabilities[tool_name]["args_schema"] = get_args_schema(function)

        # Set enhanced system context
        self.shared["system_context"] = {
            "capabilities_summary": self._build_capabilities_summary(),
            "tool_count": len(self.shared["available_tools"]),
            "analysis_loaded": len(self._tool_capabilities),
            "intelligence_level": "high" if self._tool_capabilities else "basic",
            "context_management": "advanced_session_aware",
            "session_managers": len(self.shared.get("session_managers", {})),
        }


        rprint("Advanced context awareness initialized with session management")

    async def get_context(self, session_id: str = None, format_for_llm: bool = True) -> str | dict[str, Any]:
        """
        ÜBERARBEITET: Get context über UnifiedContextManager statt verteilte Quellen
        """
        try:
            session_id = session_id or self.shared.get("session_id", self.active_session)
            query = self.shared.get("current_query", "")

            #Hole unified context über Context Manager
            unified_context = await self.context_manager.build_unified_context(session_id, query, "full")


            if format_for_llm:
                return self.context_manager.get_formatted_context_for_llm(unified_context)
            else:
                return unified_context

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            eprint(f"Failed to generate context via UnifiedContextManager: {e}")

            # FALLBACK: Fallback zu alter Methode falls UnifiedContextManager fehlschlägt
            if format_for_llm:
                return f"Error generating context: {str(e)}"
            else:
                return {
                    "error": str(e),
                    "generated_at": datetime.now().isoformat(),
                    "fallback_mode": True
                }

    def get_context_statistics(self) -> dict[str, Any]:
        """Get comprehensive context management statistics"""
        stats = {
            "context_system": "advanced_session_aware",
            "compression_threshold": 0.76,
            "max_tokens": getattr(self, 'max_input_tokens', 8000),
            "session_managers": {},
            "context_usage": {},
            "compression_stats": {}
        }

        # Session manager statistics
        session_managers = self.shared.get("session_managers", {})
        for name, manager in session_managers.items():
            stats["session_managers"][name] = {
                "history_length": len(manager.history if hasattr(manager, 'history') else (manager.get("history", []) if hasattr(manager, 'get') else [])),
                "max_length": manager.max_length if hasattr(manager, 'max_length') else manager.get("max_length", 0),
                "space_name": manager.space_name if hasattr(manager, 'space_name') else manager.get("space_name", "")
            }

        # Context node statistics if available
        if hasattr(self.task_flow, 'context_manager'):
            context_manager = self.task_flow.context_manager
            stats["compression_stats"] = {
                "compression_threshold": context_manager.compression_threshold,
                "max_tokens": context_manager.max_tokens,
                "active_sessions": len(context_manager.session_managers)
            }

        # LLM call statistics from enhanced node
        llm_stats = self.shared.get("llm_call_stats", {})
        if llm_stats:
            stats["context_usage"] = {
                "total_llm_calls": llm_stats.get("total_calls", 0),
                "context_compression_rate": llm_stats.get("context_compression_rate", 0.0),
                "average_context_tokens": llm_stats.get("context_tokens_used", 0) / max(llm_stats.get("total_calls", 1),
                                                                                        1)
            }

        return stats

    def set_persona(self, name: str, style: str = "professional", tone: str = "friendly",
                    personality_traits: list[str] = None, apply_method: str = "system_prompt",
                    integration_level: str = "light", custom_instructions: str = ""):
        """Set agent persona mit erweiterten Konfigurationsmöglichkeiten"""
        if personality_traits is None:
            personality_traits = ["helpful", "concise"]

        self.amd.persona = PersonaConfig(
            name=name,
            style=style,
            tone=tone,
            personality_traits=personality_traits,
            custom_instructions=custom_instructions,
            apply_method=apply_method,
            integration_level=integration_level
        )

        rprint(f"Persona set: {name} ({style}, {tone}) - Method: {apply_method}, Level: {integration_level}")

    def configure_persona_integration(self, apply_method: str = "system_prompt", integration_level: str = "light"):
        """Configure how persona is applied"""
        if self.amd.persona:
            self.amd.persona.apply_method = apply_method
            self.amd.persona.integration_level = integration_level
            rprint(f"Persona integration updated: {apply_method}, {integration_level}")
        else:
            wprint("No persona configured to update")

    def get_available_variables(self) -> dict[str, dict]:
        """Get available variables for dynamic formatting"""
        return self.variable_manager.get_available_variables()

    async def _orchestrate_execution(self) -> str:
        """
        Enhanced orchestration with LLMReasonerNode as strategic core.
        The reasoner now handles both task management and response generation internally.
        """

        self.shared["agent_instance"] = self
        self.shared["session_id"] = self.active_session
        # === UNIFIED REASONING AND EXECUTION CYCLE ===
        rprint("Starting strategic reasoning and execution cycle")

        # The LLMReasonerNode now handles the complete cycle:
        # 1. Strategic analysis of the query
        # 2. Decision making about approach
        # 3. Orchestration of sub-systems (LLMToolNode, TaskPlanner/Executor)
        # 4. Response synthesis and formatting

        # Execute the unified flow
        task_management_result = await self.task_flow.run_async(self.shared)

        # Check for various completion states
        if self.shared.get("plan_halted"):
            error_response = f"Task execution was halted: {self.shared.get('halt_reason', 'Unknown reason')}"
            self.shared["current_response"] = error_response
            return error_response

        # The reasoner provides the final response
        final_response = self.shared.get("current_response", "Task completed successfully.")

        # Add reasoning artifacts to response if available
        reasoning_artifacts = self.shared.get("reasoning_artifacts", {})
        if reasoning_artifacts and reasoning_artifacts.get("reasoning_loops", 0) > 1:
            # For debugging/transparency, could add reasoning info to metadata
            pass

        # Log enhanced statistics
        self._log_execution_stats()

        return final_response

    def _log_execution_stats(self):
        """Enhanced execution statistics with reasoning metrics"""
        tasks = self.shared.get("tasks", {})
        adaptations = self.shared.get("plan_adaptations", 0)
        reasoning_artifacts = self.shared.get("reasoning_artifacts", {})

        completed_tasks = sum(1 for t in tasks.values() if t.status == "completed")
        failed_tasks = sum(1 for t in tasks.values() if t.status == "failed")

        # Enhanced logging with reasoning metrics
        reasoning_loops = reasoning_artifacts.get("reasoning_loops", 0)

        stats_message = f"Execution complete - Tasks: {completed_tasks} completed, {failed_tasks} failed"

        if adaptations > 0:
            stats_message += f", {adaptations} adaptations"

        if reasoning_loops > 0:
            stats_message += f", {reasoning_loops} reasoning loops"

            # Add reasoning efficiency metric
            if completed_tasks > 0:
                efficiency = completed_tasks / max(reasoning_loops, 1)
                stats_message += f" (efficiency: {efficiency:.1f} tasks/loop)"

        rprint(stats_message)

        # Log reasoning context if significant
        if reasoning_loops > 3:
            internal_task_stack = reasoning_artifacts.get("internal_task_stack", [])
            completed_reasoning_tasks = len([t for t in internal_task_stack if t.get("status") == "completed"])

            if completed_reasoning_tasks > 0:
                rprint(f"Strategic reasoning: {completed_reasoning_tasks} high-level tasks completed")

    def _build_capabilities_summary(self) -> str:
        """Build summary of agent capabilities"""

        if not self._tool_capabilities:
            return "Basic LLM capabilities only"

        summaries = []
        for tool_name, cap in self._tool_capabilities.items():
            primary = cap.get('primary_function', 'Unknown function')
            summaries.append(f"{tool_name}{cap.get('args_schema', '()')}: {primary}")

        return f"Enhanced capabilities: {'; '.join(summaries)}"

    # Neue Hilfsmethoden für erweiterte Funktionalität

    async def get_task_execution_summary(self) -> dict[str, Any]:
        """Erhalte detaillierte Zusammenfassung der Task-Ausführung"""
        tasks = self.shared.get("tasks", {})
        results_store = self.shared.get("results", {})

        summary = {
            "total_tasks": len(tasks),
            "completed_tasks": [],
            "failed_tasks": [],
            "task_types_used": {},
            "tools_used": [],
            "adaptations": self.shared.get("plan_adaptations", 0),
            "execution_timeline": [],
            "results_store": results_store
        }

        for task_id, task in tasks.items():
            task_info = {
                "id": task_id,
                "type": task.type,
                "description": task.description,
                "status": task.status,
                "duration": None
            }

            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                task_info["duration"] = duration

            if task.status == "completed":
                summary["completed_tasks"].append(task_info)
                if isinstance(task, ToolTask):
                    summary["tools_used"].append(task.tool_name)
            elif task.status == "failed":
                task_info["error"] = task.error
                summary["failed_tasks"].append(task_info)

            # Task types counting
            task_type = task.type
            summary["task_types_used"][task_type] = summary["task_types_used"].get(task_type, 0) + 1

        return summary

    async def explain_reasoning_process(self) -> str:
        """Erkläre den Reasoning-Prozess des Agenten"""
        if not LITELLM_AVAILABLE:
            return "Reasoning explanation requires LLM capabilities."

        summary = await self.get_task_execution_summary()

        prompt = f"""
Erkläre den Reasoning-Prozess dieses AI-Agenten in verständlicher Form:

## Ausführungszusammenfassung
- Total Tasks: {summary['total_tasks']}
- Erfolgreich: {len(summary['completed_tasks'])}
- Fehlgeschlagen: {len(summary['failed_tasks'])}
- Plan-Adaptationen: {summary['adaptations']}
- Verwendete Tools: {', '.join(set(summary['tools_used']))}
- Task-Typen: {summary['task_types_used']}

## Task-Details
Erfolgreiche Tasks:
{self._format_tasks_for_explanation(summary['completed_tasks'])}

## Anweisungen
Erkläre in 2-3 Absätzen:
1. Welche Strategie der Agent gewählt hat
2. Wie er die Aufgabe in Tasks unterteilt hat
3. Wie er auf unerwartete Ergebnisse reagiert hat (falls Adaptationen)
4. Was die wichtigsten Erkenntnisse waren

Schreibe für einen technischen Nutzer, aber verständlich."""

        try:
            response = await self.a_run_llm_completion(
                model=self.amd.complex_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=800,task_id="reasoning_explanation"
            )

            return response

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"Could not generate reasoning explanation: {e}"

    def _format_tasks_for_explanation(self, tasks: list[dict]) -> str:
        formatted = []
        for task in tasks[:5]:  # Top 5 tasks
            duration_info = f" ({task['duration']:.1f}s)" if task['duration'] else ""
            formatted.append(f"- {task['type']}: {task['description']}{duration_info}")
        return "\n".join(formatted)

    # ===== PAUSE/RESUME FUNCTIONALITY =====

    async def pause(self) -> bool:
        """Pause agent execution"""
        if not self.is_running:
            return False

        self.is_paused = True
        self.shared["system_status"] = "paused"

        # Create checkpoint
        checkpoint = await self._create_checkpoint()
        await self._save_checkpoint(checkpoint)

        rprint("Agent execution paused")
        return True

    async def resume(self) -> bool:
        """Resume agent execution"""
        if not self.is_paused:
            return False

        self.is_paused = False
        self.shared["system_status"] = "running"

        rprint("Agent execution resumed")
        return True

    # ===== CHECKPOINT MANAGEMENT =====

    async def _create_checkpoint(self) -> AgentCheckpoint:
        """
        Erstellt einen robusten, serialisierbaren Checkpoint, der nur reine Daten enthält.
        Laufzeitobjekte und nicht-serialisierbare Elemente werden explizit ausgeschlossen.
        """
        try:
            rprint("Starte Erstellung eines Daten-Checkpoints...")
            if hasattr(self.amd, 'budget_manager') and self.amd.budget_manager:
                self.amd.budget_manager.save_data()

            amd_data = self.amd.model_dump()
            amd_data['budget_manager'] = None  # Explizit entfernen, da es nicht serialisierbar ist

            # 1. Bereinige die Variable-Scopes: Dies ist der wichtigste Schritt.
            cleaned_variable_scopes = {}
            if self.variable_manager:
                # Wir erstellen eine tiefe Kopie, um den laufenden Zustand nicht zu verändern
                # import copy
                scopes_copy = self.variable_manager.scopes.copy()
                cleaned_variable_scopes = _clean_data_for_serialization(scopes_copy)

            # 2. Bereinige Session-Daten
            session_data = {}
            if self.context_manager and self.context_manager.session_managers:
                for session_id, session in self.context_manager.session_managers.items():
                    history = []
                    # Greife sicher auf die History zu
                    if hasattr(session, 'history') and session.history:
                        history = session.history[-50:]  # Nur die letzten 50 Interaktionen speichern
                    elif isinstance(session, dict) and 'history' in session:
                        history = session.get('history', [])[-50:]

                    session_data[session_id] = {
                        "history": history,
                        "session_type": "chatsession" if hasattr(session, 'history') else "fallback"
                    }

            # 3. Erstelle den Checkpoint nur mit den bereinigten, reinen Daten
            checkpoint = AgentCheckpoint(
                timestamp=datetime.now(),
                agent_state={
                    "is_running": self.is_running,
                    "is_paused": self.is_paused,
                    "amd_data": amd_data,
                    "active_session": self.active_session,
                    "system_status": self.shared.get("system_status", "idle"),
                    # Token and cost tracking
                    "total_tokens_in": self.total_tokens_in,
                    "total_tokens_out": self.total_tokens_out,
                    "total_cost_accumulated": self.total_cost_accumulated,
                    "total_llm_calls": self.total_llm_calls
                },
                task_state={
                    task_id: asdict(task) for task_id, task in self.shared.get("tasks", {}).items()
                },
                world_model=self.shared.get("world_model", {}),
                active_flows=["task_flow", "response_flow"],
                metadata={
                    "session_id": self.shared.get("session_id", "default"),
                    "last_query": self.shared.get("current_query", ""),
                    "checkpoint_version": "4.1_data_only",
                    "agent_name": self.amd.name
                },
                # Die bereinigten Zusatzdaten
                session_data=session_data,
                variable_scopes=cleaned_variable_scopes,
                results_store=self.shared.get("results", {}),
                conversation_history=self.shared.get("conversation_history", [])[-100:],
                tool_capabilities=self._tool_capabilities.copy(),
                session_tool_restrictions=self.session_tool_restrictions.copy()
            )

            rprint(
                f"Daten-Checkpoint erfolgreich erstellt. {len(cleaned_variable_scopes)} Scopes bereinigt und gespeichert.")
            return checkpoint

        except Exception as e:
            eprint(f"FEHLER bei der Checkpoint-Erstellung: {e}")
            import traceback
            print(traceback.format_exc())
            raise

    async def _save_checkpoint(self, checkpoint: AgentCheckpoint, filepath: str = None):
        """Vereinfachtes Checkpoint-Speichern - alles in eine Datei"""
        try:
            from toolboxv2 import get_app
            folder = str(get_app().data_dir) + '/Agents/checkpoint/' + self.amd.name
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

            if not filepath:
                timestamp = checkpoint.timestamp.strftime("%Y%m%d_%H%M%S")
                filepath = f"agent_checkpoint_{timestamp}.pkl"
            filepath = os.path.join(folder, filepath)

            # Sessions vor dem Speichern synchronisieren
            if self.context_manager and self.context_manager.session_managers:
                for session_id, session in self.context_manager.session_managers.items():
                    try:
                        if hasattr(session, 'save'):
                            await session.save()
                        elif hasattr(session, '_save_to_memory'):
                            session._save_to_memory()
                    except Exception as e:
                        rprint(f"Session sync error für {session_id}: {e}")

            # Speichere Checkpoint direkt
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f)

            self.last_checkpoint = checkpoint.timestamp

            # Erstelle einfache Zusammenfassung
            summary_parts = []
            if hasattr(checkpoint, 'session_data') and checkpoint.session_data:
                summary_parts.append(f"{len(checkpoint.session_data)} sessions")
            if checkpoint.task_state:
                completed_tasks = len([t for t in checkpoint.task_state.values() if t.get("status") == "completed"])
                summary_parts.append(f"{completed_tasks} completed tasks")
            if hasattr(checkpoint, 'variable_scopes') and checkpoint.variable_scopes:
                summary_parts.append(f"{len(checkpoint.variable_scopes)} variable scopes")

            summary = "; ".join(summary_parts) if summary_parts else "Basic checkpoint"
            rprint(f"Checkpoint gespeichert: {filepath} ({summary})")
            return True

        except Exception as e:
            eprint(f"Checkpoint-Speicherung fehlgeschlagen: {e}")
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

            # Re-initialisiere Kontext-Awareness
            await self.initialize_context_awareness()

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

            # 3. Tasks wiederherstellen
            if checkpoint.task_state:
                restored_tasks = {}
                for task_id, task_data in checkpoint.task_state.items():
                    try:
                        task_type = task_data.get("type", "generic")
                        if task_type == "LLMTask":
                            restored_tasks[task_id] = LLMTask(**task_data)
                        elif task_type == "ToolTask":
                            restored_tasks[task_id] = ToolTask(**task_data)
                        elif task_type == "DecisionTask":
                            restored_tasks[task_id] = DecisionTask(**task_data)
                        else:
                            restored_tasks[task_id] = Task(**task_data)

                        restore_stats["tasks_restored"] += 1
                    except Exception as e:
                        restore_stats["errors"].append(f"Task {task_id}: {e}")

                self.shared["tasks"] = restored_tasks

            # 4. Results Store wiederherstellen
            if hasattr(checkpoint, 'results_store') and checkpoint.results_store:
                self.shared["results"] = checkpoint.results_store
                if self.variable_manager:
                    self.variable_manager.set_results_store(checkpoint.results_store)

            # 5. Variable System wiederherstellen (KRITISCHER TEIL)
            if hasattr(checkpoint, 'variable_scopes') and checkpoint.variable_scopes:
                # A. Der VariableManager wird mit dem geladenen World Model neu erstellt.
                self.variable_manager = VariableManager(self.shared["world_model"], self.shared)
                self._setup_variable_scopes()

                # B. Stellen Sie die bereinigten Daten-Scopes wieder her.
                for scope_name, scope_data in checkpoint.variable_scopes.items():
                    self.variable_manager.register_scope(scope_name, scope_data)
                restore_stats["variables_restored"] = len(checkpoint.variable_scopes)

                # C. WICHTIG: Fügen Sie jetzt die Laufzeitobjekte wieder in den 'shared' Scope ein.
                # Diese werden nicht aus dem Checkpoint geladen, sondern neu zugewiesen.
                self.shared["variable_manager"] = self.variable_manager
                self.shared["context_manager"] = self.context_manager
                self.shared["agent_instance"] = self
                self.shared["progress_tracker"] = self.progress_tracker
                self.shared["llm_tool_node_instance"] = self.task_flow.llm_tool_node
                # Verbinde den Executor wieder mit der Agent-Instanz

                rprint("Variablen-System aus Daten wiederhergestellt und Laufzeitobjekte neu verknüpft.")

            # 6. Sessions und Conversation wiederherstellen
            if auto_restore_history:
                await self._restore_sessions_and_conversation_simplified(checkpoint, restore_stats)

            # 7. Tool Capabilities wiederherstellen
            if hasattr(checkpoint, 'tool_capabilities') and checkpoint.tool_capabilities:
                self._tool_capabilities = checkpoint.tool_capabilities.copy()

            # 8. Session Tool Restrictions wiederherstellen
            if hasattr(checkpoint, 'session_tool_restrictions') and checkpoint.session_tool_restrictions:
                self.session_tool_restrictions = checkpoint.session_tool_restrictions.copy()
                restore_stats["tool_restrictions_restored"] = len(checkpoint.session_tool_restrictions)
                rprint(f"Tool restrictions wiederhergestellt: {len(checkpoint.session_tool_restrictions)} Tools mit Restrictions")

            self.shared["system_status"] = "restored"
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
            if not self.context_manager:
                self.context_manager = UnifiedContextManager(self)
                self.context_manager.variable_manager = self.variable_manager

            # Sessions wiederherstellen
            if hasattr(checkpoint, 'session_data') and checkpoint.session_data:
                for session_id, session_info in checkpoint.session_data.items():
                    try:
                        # Session über Context Manager initialisieren
                        max_length = session_info.get("message_count", 200)
                        restored_session = await self.context_manager.initialize_session(session_id, max_length)

                        # History wiederherstellen
                        history = session_info.get("history", [])
                        if history and hasattr(restored_session, 'history'):
                            # Direkt in Session-History einfügen
                            restored_session.history.extend(history)

                        restore_stats["sessions_restored"] += 1
                    except Exception as e:
                        restore_stats["errors"].append(f"Session {session_id}: {e}")

            # Conversation History wiederherstellen
            if hasattr(checkpoint, 'conversation_history') and checkpoint.conversation_history:
                self.shared["conversation_history"] = checkpoint.conversation_history
                restore_stats["conversation_restored"] = len(checkpoint.conversation_history)

            # Update shared context
            self.shared["context_manager"] = self.context_manager
            if self.context_manager.session_managers:
                self.shared["session_managers"] = self.context_manager.session_managers
                self.shared["session_initialized"] = True

        except Exception as e:
            restore_stats["errors"].append(f"Session/conversation restore failed: {e}")

    async def _maybe_checkpoint(self):
        """Vereinfachtes automatisches Checkpointing"""
        if not self.enable_pause_resume:
            return

        now = datetime.now()
        if (not self.last_checkpoint or
            (now - self.last_checkpoint).seconds >= self.checkpoint_interval):

            try:
                checkpoint = await self._create_checkpoint()
                await self.delete_old_checkpoints(keep_count=self.checkpoint_config.max_checkpoints)
                await self._save_checkpoint(checkpoint)
            except Exception as e:
                eprint(f"Automatic checkpoint failed: {e}")

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
    def _get_tool_analysis_path(self) -> str:
        """Get path for tool analysis cache"""
        from toolboxv2 import get_app
        folder = str(get_app().data_dir) + '/Agents/capabilities/'
        os.makedirs(folder, exist_ok=True)
        return folder + 'tool_capabilities.json'

    def _get_context_path(self, session_id=None) -> str:
        """Get path for tool analysis cache"""
        from toolboxv2 import get_app
        folder = str(get_app().data_dir) + '/Agents/context/' + self.amd.name
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_suffix = f"_session_{session_id}" if session_id else ""
        filepath = f"agent_context_{self.amd.name}_{timestamp}{session_suffix}.json"
        return folder + f'/{filepath}'

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
        if hasattr(self.task_flow, 'llm_reasoner'):
            if not hasattr(self.task_flow.llm_reasoner, 'meta_tools_registry'):
                self.task_flow.llm_reasoner.meta_tools_registry = {}

            self.task_flow.llm_reasoner.meta_tools_registry[tool_name] = {
                "function": effective_func,
                "description": tool_description,
                "args_schema": get_args_schema(tool_func)
            }

            rprint(f"First-class meta-tool added: {tool_name}")
        else:
            wprint("LLMReasonerNode not available for first-class tool registration")

    async def add_tool(self, tool_func: Callable, name: str = None, description: str = None, is_new=False):
        """Enhanced tool addition with intelligent analysis"""
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
        if is_new:
            await self._analyze_tool_capabilities(tool_name, tool_description, get_args_schema(tool_func))
        else:
            if res := self._load_tool_analysis([tool_name]):
                self._tool_capabilities[tool_name] = res.get(tool_name)
            else:
                await self._analyze_tool_capabilities(tool_name, tool_description, get_args_schema(tool_func))

        rprint(f"Tool added with analysis: {tool_name}")

    async def _batch_analyze_tool_capabilities(self, tools_data: list[dict]):
        """
        Batch analyze multiple tools in a single LLM call for efficiency

        Args:
            tools_data: List of dicts with 'name', 'description', 'args_schema' keys
        """
        if not LITELLM_AVAILABLE:
            # Fallback for each tool
            for tool_data in tools_data:
                self._tool_capabilities[tool_data['name']] = {
                    "use_cases": [tool_data['description']],
                    "triggers": [tool_data['name'].lower().replace('_', ' ')],
                    "complexity": "unknown",
                    "confidence": 0.3
                }
            return

        # Build batch analysis prompt
        tools_section = "\n\n".join([
            f"Tool {i+1}: {tool['name']}\nArgs: {tool['args_schema']}\nDescription: {tool['description']}"
            for i, tool in enumerate(tools_data)
        ])

        prompt = f"""
Analyze these {len(tools_data)} tools and identify their capabilities in a structured format.
For EACH tool, provide a complete analysis.

{tools_section}

For each tool, provide:
1. primary_function: One-sentence description of main purpose
2. use_cases: List of 3-5 specific use cases
3. trigger_phrases: List of 5-10 phrases that indicate this tool should be used
4. confidence_triggers: Dict of phrases with confidence scores (0.0-1.0)
5. indirect_connections: List of related concepts/tasks
6. tool_complexity: "simple" | "medium" | "complex"
7. estimated_execution_time: "fast" | "medium" | "slow"

Respond in YAML format with this structure:
```yaml
tools:
  tool_name_1:
    primary_function: "..."
    use_cases: [...]
    trigger_phrases: [...]
    confidence_triggers:
      "phrase": 0.8
    indirect_connections: [...]
    tool_complexity: "medium"
    estimated_execution_time: "fast"
  tool_name_2:
    # ... same structure
```
"""

        model = os.getenv("BASEMODEL", self.amd.fast_llm_model)

        try:
            response = await self.a_run_llm_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                with_context=False,
                temperature=0.3,
                max_tokens=2000 + (len(tools_data) * 200),  # Scale with number of tools
                task_id="batch_tool_analysis"
            )

            # Extract YAML
            yaml_match = re.search(r"```yaml\s*(.*?)\s*```", response, re.DOTALL)
            if yaml_match:
                yaml_str = yaml_match.group(1)
            else:
                yaml_str = response

            analysis_data = yaml.safe_load(yaml_str)

            # Store individual tool analyses
            if "tools" in analysis_data:
                for tool_name, analysis in analysis_data["tools"].items():
                    self._tool_capabilities[tool_name] = analysis
                    rprint(f"Batch analyzed: {tool_name}")

            # Save to cache
            self._all_tool_capabilities.update(self._tool_capabilities)
            await self._save_tool_analysis()

        except Exception as e:
            eprint(f"Batch tool analysis failed: {e}")
            # Fallback to individual analysis
            for tool_data in tools_data:
                await self._analyze_tool_capabilities(
                    tool_data["name"], tool_data["description"], tool_data["args_schema"]
                )

    async def _analyze_tool_capabilities(self, tool_name: str, description: str, tool_args:str):
        """Analyze tool capabilities with LLM for smart usage"""

        # Try to load existing analysis
        existing_analysis = self._load_tool_analysis()

        if tool_name in existing_analysis:
            try:
                # Validate cached data against the Pydantic model
                ToolAnalysis.model_validate(existing_analysis[tool_name])
                self._tool_capabilities[tool_name] = existing_analysis[tool_name]
                rprint(f"Loaded and validated cached analysis for {tool_name}")
            except ValidationError as e:
                wprint(f"Cached data for {tool_name} is invalid and will be regenerated: {e}")
                del self._tool_capabilities[tool_name]

        if not LITELLM_AVAILABLE:
            # Fallback analysis
            self._tool_capabilities[tool_name] = {
                "use_cases": [description],
                "triggers": [tool_name.lower().replace('_', ' ')],
                "complexity": "unknown",
                "confidence": 0.3
            }
            return

        # LLM-based intelligent analysis
        prompt = f"""
Analyze this tool and identify ALL possible use cases, triggers, and connections:

Tool Name: {tool_name}
args: {tool_args}
Description: {description}


Provide a comprehensive analysis covering:

1. OBVIOUS use cases (direct functionality)
2. INDIRECT connections (when this tool might be relevant)
3. TRIGGER PHRASES (what user queries would benefit from this tool)
4. COMPLEX scenarios (non-obvious applications)
5. CONTEXTUAL usage (when combined with other information)

Example for a "get_user_name" tool:
- Obvious: When user asks "what is my name"
- Indirect: Personalization, greetings, user identification
- Triggers: "my name", "who am I", "hello", "introduce yourself", "personalize"
- Complex: User context in multi-step tasks, addressing user directly
- Contextual: Any response that could be personalized

Rule! no additional comments or text in the format !
schema:
 {yaml.dump(safe_for_yaml(ToolAnalysis.model_json_schema()))}

Respond in YAML format:
Example:
```yaml
primary_function: "Retrieves the current user's name."
use_cases:
  - "Responding to 'what is my name?'"
  - "Personalizing greeting messages."
trigger_phrases:
  - "my name"
  - "who am I"
  - "introduce yourself"
indirect_connections:
  - "User identification in multi-factor authentication."
  - "Tagging user-generated content."
complexity_scenarios:
  - "In a multi-step task, remembering the user's name to personalize the final output."
user_intent_categories:
  - "Personalization"
  - "User Identification"
confidence_triggers:
  "my name": 0.95
  "who am I": 0.9
tool_complexity: low/medium/high
```
"""
        model = os.getenv("BASEMODEL", self.amd.fast_llm_model)
        for i in range(3):
            try:
                response = await self.a_run_llm_completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    with_context=False,
                    temperature=0.3,
                    max_tokens=1000,
                    task_id=f"tool_analysis_{tool_name}"
                )

                content = response.strip()

                # Extract JSON
                if "```yaml" in content:
                    yaml_str = content.split("```yaml")[1].split("```")[0].strip()
                else:
                    yaml_str = content

                analysis = yaml.safe_load(yaml_str)

                # Store analysis
                self._tool_capabilities[tool_name] = analysis

                # Save to cache
                self._all_tool_capabilities[tool_name] = analysis
                await self._save_tool_analysis()

                validated_analysis = ToolAnalysis.model_validate(analysis)
                rprint(f"Generated intelligent analysis for {tool_name}")
                break

            except Exception as e:
                import traceback
                print(traceback.format_exc())
                model = self.amd.complex_llm_model if i > 1 else self.amd.fast_llm_model
                eprint(f"Tool analysis failed for {tool_name}: {e}")
                # Fallback
                self._tool_capabilities[tool_name] = {
                    "primary_function": description,
                    "use_cases": [description],
                    "trigger_phrases": [tool_name.lower().replace('_', ' ')],
                    "tool_complexity": "medium"
                }

    def _load_tool_analysis(self, tool_names: list[str] = None) -> dict[str, Any]:
        """
        Load tool analysis from cache - optimized to load only specified tools

        Args:
            tool_names: Optional list of tool names to load. If None, loads all cached analyses.

        Returns:
            dict: Tool capabilities for requested tools only
        """
        try:
            if os.path.exists(self.tool_analysis_file):
                with open(self.tool_analysis_file) as f:
                    all_analyses = json.load(f)
                self._all_tool_capabilities.update(all_analyses)
                # If specific tools requested, filter to only those
                if tool_names is not None:
                    return {name: analysis for name, analysis in all_analyses.items() if name in tool_names}

                return all_analyses
        except Exception as e:
            wprint(f"Could not load tool analysis: {e}")
        return {}


    async def save_context_to_file(self, session_id: str = None) -> bool:
        """Save current context to file"""
        try:
            context = await self.get_context(session_id=session_id, format_for_llm=False)

            filepath = self._get_context_path(session_id)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(context, f, indent=2, ensure_ascii=False, default=str)

            rprint(f"Context saved to: {filepath}")
            return True

        except Exception as e:
            eprint(f"Failed to save context: {e}")
            return False

    async def _save_tool_analysis(self):
        """Save tool analysis to cache"""
        if not self._all_tool_capabilities:
            return
        try:
            with open(self.tool_analysis_file, 'w') as f:
                json.dump(self._all_tool_capabilities, f, indent=2)
        except Exception as e:
            eprint(f"Could not save tool analysis: {e}")

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

    async def a_format_class_leg(self,
                             pydantic_model: type[BaseModel],
                             prompt: str,
                             message_context: list[dict] = None,
                             max_retries: int = 2, auto_context=True, session_id: str = None, llm_kwargs=None,
                             model_preference="complex", **kwargs) -> dict[str, Any]:
        """
        State-of-the-art LLM-based structured data formatting using Pydantic models.
        Supports media inputs via [media:(path/url)] tags in the prompt.

        Args:
            pydantic_model: The Pydantic model class to structure the response
            prompt: The main prompt for the LLM (can include [media:(path/url)] tags)
            message_context: Optional conversation context messages
            max_retries: Maximum number of retry attempts
            auto_context: Whether to include session context
            session_id: Optional session ID
            llm_kwargs: Additional kwargs to pass to litellm
            model_preference: "fast" or "complex"
            **kwargs: Additional arguments (merged with llm_kwargs)

        Returns:
            dict: Validated structured data matching the Pydantic model

        Raises:
            ValidationError: If the LLM response cannot be validated against the model
            RuntimeError: If all retry attempts fail
        """

        if not LITELLM_AVAILABLE:
            raise RuntimeError("LiteLLM is required for structured formatting but not available")

        if session_id and self.active_session != session_id:
            self.active_session = session_id
        # Generate schema documentation
        schema = pydantic_model.model_json_schema() if issubclass(pydantic_model, BaseModel) else (json.loads(pydantic_model) if isinstance(pydantic_model, str) else pydantic_model)
        model_name = pydantic_model.__name__ if hasattr(pydantic_model, "__name__") else (pydantic_model.get("title", "UnknownModel") if isinstance(pydantic_model, dict) else "UnknownModel")

        # Create enhanced prompt with schema
        enhanced_prompt = f"""
    {prompt}

    CRITICAL FORMATTING REQUIREMENTS:
    1. Respond ONLY in valid YAML format
    2. Follow the exact schema structure provided
    3. Use appropriate data types (strings, lists, numbers, booleans)
    4. Include ALL required fields
    5. No additional comments, explanations, or text outside the YAML

    SCHEMA FOR {model_name}:
    {yaml.dump(safe_for_yaml(schema), default_flow_style=False, indent=2)}

    EXAMPLE OUTPUT FORMAT:
    ```yaml
    # Your response here following the schema exactly
    field_name: "value"
    list_field:
      - "item1"
      - "item2"
    boolean_field: true
    number_field: 42
Respond in YAML format only:
"""
        # Prepare messages
        messages = []
        if message_context:
            messages.extend(message_context)
        messages.append({"role": "user", "content": enhanced_prompt})

        # Retry logic with progressive adjustments
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Adjust parameters based on attempt
                temperature = 0.1 + (attempt * 0.1)  # Increase temperature slightly on retries
                max_tokens = min(2000 + (attempt * 500), 4000)  # Increase token limit on retries

                rprint(f"[{model_name}] Attempt {attempt + 1}/{max_retries + 1} (temp: {temperature})")

                # Generate LLM response
                response = await self.a_run_llm_completion(
                    model_preference=model_preference,
                    messages=messages,
                    stream=False,
                    with_context=auto_context,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    task_id=f"format_{model_name.lower()}_{attempt}",
                    llm_kwargs=llm_kwargs
                )

                if not response or not response.strip():
                    raise ValueError("Empty response from LLM")

                # Extract YAML content with multiple fallback strategies

                yaml_content = self._extract_yaml_content(response)

                print(f"{'='*20}\n {prompt} \n{'-'*20}\n")
                print(f"{response} \n{'='*20}")

                if not yaml_content:
                    raise ValueError("No valid YAML content found in response")

                # Parse YAML
                try:
                    parsed_data = yaml.safe_load(yaml_content)
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid YAML syntax: {e}")
                iprint(parsed_data)
                if not isinstance(parsed_data, dict):
                    raise ValueError(f"Expected dict, got {type(parsed_data)}")

                # Validate against Pydantic model
                try:
                    if isinstance(pydantic_model, BaseModel):
                        validated_instance = pydantic_model.model_validate(parsed_data)
                        validated_data = validated_instance.model_dump()
                    else:
                        validated_data = parsed_data

                    rprint(f"✅ Successfully formatted {model_name} on attempt {attempt + 1}")
                    return validated_data

                except ValidationError as e:
                    detailed_errors = []
                    for error in e.errors():
                        field_path = " -> ".join(str(x) for x in error['loc'])
                        detailed_errors.append(f"Field '{field_path}': {error['msg']}")

                    error_msg = "Validation failed:\n" + "\n".join(detailed_errors)
                    raise ValueError(error_msg)

            except Exception as e:
                last_error = e
                wprint(f"[{model_name}] Attempt {attempt + 1} failed: {str(e)}")

                if attempt < max_retries:
                    # Add error feedback for next attempt
                    error_feedback = f"\n\nPREVIOUS ATTEMPT FAILED: {str(e)}\nPlease correct the issues and provide valid YAML matching the schema exactly."
                    messages[-1]["content"] = enhanced_prompt + error_feedback

                    # Brief delay before retry
                    # await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    eprint(f"[{model_name}] All {max_retries + 1} attempts failed")

        # All attempts failed
        raise RuntimeError(f"Failed to format {model_name} after {max_retries + 1} attempts. Last error: {last_error}")

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
            #Clear über Context Manager
            if session_id:
                # Clear specific session
                if session_id in self.context_manager.session_managers:
                    session = self.context_manager.session_managers[session_id]
                    if hasattr(session, 'history'):
                        session.history = []
                    elif isinstance(session, dict) and 'history' in session:
                        session['history'] = []

                    # Remove from session managers
                    del self.context_manager.session_managers[session_id]

                    # Clear variable manager scope for this session
                    if self.variable_manager:
                        scope_name = f'session_{session_id}'
                        if scope_name in self.variable_manager.scopes:
                            del self.variable_manager.scopes[scope_name]

                    rprint(f"Context cleared for session: {session_id}")
            else:
                # Clear all sessions
                for session_id, session in self.context_manager.session_managers.items():
                    if hasattr(session, 'history'):
                        session.history = []
                    elif isinstance(session, dict) and 'history' in session:
                        session['history'] = []

                self.context_manager.session_managers = {}
                rprint("Context cleared for all sessions")

            # Clear context cache
            self.context_manager._invalidate_cache(session_id)

            # Clear current execution context in shared
            context_keys_to_clear = [
                "current_query", "current_response", "current_plan", "tasks",
                "results", "task_plans", "session_data", "formatted_context",
                "synthesized_response", "quality_assessment", "plan_adaptations",
                "executor_performance", "llm_tool_conversation", "aggregated_context"
            ]

            for key in context_keys_to_clear:
                if key in self.shared:
                    if isinstance(self.shared[key], dict):
                        self.shared[key] = {}
                    elif isinstance(self.shared[key], list):
                        self.shared[key] = []
                    else:
                        self.shared[key] = None

            # Clear variable manager scopes (except core system variables)
            if hasattr(self, 'variable_manager'):
                # Clear user, results, tasks scopes
                self.variable_manager.register_scope('user', {})
                self.variable_manager.register_scope('results', {})
                self.variable_manager.register_scope('tasks', {})
                # Reset cache
                self.variable_manager._cache.clear()

            # Reset execution state
            self.is_running = False
            self.is_paused = False
            self.shared["system_status"] = "idle"

            # Clear progress tracking
            if hasattr(self, 'progress_tracker'):
                self.progress_tracker.reset_session_metrics()

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
            self.shared["world_model"] = {}
            self.world_model = {}

            # Clean performance metrics
            self.shared["performance_metrics"] = {}

            # Deep clean session storage
            session_managers = self.shared.get("session_managers", {})
            if session_managers:
                for _manager_name, manager in session_managers.items():
                    if hasattr(manager, 'clear_all_history'):
                        await manager.clear_all_history()
                    elif hasattr(manager, 'clear_history'):
                        manager.clear_history()

            # Clear session managers entirely
            self.shared["session_managers"] = {}
            self.shared["session_initialized"] = False

            # Clean variable manager completely
            if hasattr(self, 'variable_manager'):
                # Reinitialize with clean state
                self.variable_manager = VariableManager({}, self.shared)
                self._setup_variable_scopes()

            # Clean tool analysis cache if deep clean
            if deep_clean:
                self._tool_capabilities = {}
                self._tool_analysis_cache = {}

            # Clean checkpoint data
            self.checkpoint_data = {}
            self.last_checkpoint = None

            # Clean context manager sessions
            if hasattr(self.task_flow, 'context_manager'):
                self.task_flow.context_manager.session_managers = {}

            # Clean LLM call statistics
            self.shared.pop("llm_call_stats", None)

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

        # Create final checkpoint
        if self.enable_pause_resume:
            checkpoint = await self._create_checkpoint()
            await self._save_checkpoint(checkpoint, "final_checkpoint.pkl")

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

Returns:
    dict: Voting results with winner, votes, margin, and cost info"""
        )

    async def a_voting(
        self,
        mode: Literal["simple", "advanced", "unstructured"] = "simple",
        prompt: str = None,
        options: list[str] = None,
        k_margin: int = 2,
        num_voters: int = 5,
        votes_per_voter: int = 1,
        num_thinkers: int = 3,
        strategy: Literal["best", "vote", "recombine"] = "best",
        num_organizers: int = 2,
        vote_on_parts: bool = True,
        final_construction: bool = True,
        unstructured_data: str = None,
        complex_data: bool = False,
        task_id: str = "voting_task",
        session_id: str = None,
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
            if mode == "simple":
                result = await self._voting_simple(
                    prompt, options, k_margin, num_voters, votes_per_voter,
                    session_id, voting_model, **kwargs
                )
            elif mode == "advanced":
                result = await self._voting_advanced(
                    prompt, num_thinkers, strategy, k_margin, complex_data,
                    task_id, session_id, voting_model, **kwargs
                )
            elif mode == "unstructured":
                result = await self._voting_unstructured(
                    prompt, unstructured_data, num_organizers, k_margin,
                    vote_on_parts, final_construction, complex_data,
                    task_id, session_id, voting_model, **kwargs
                )
            else:
                raise ValueError(f"Invalid mode: {mode}. Use 'simple', 'advanced', or 'unstructured'")

            # Add cost information
            result["cost_info"] = {
                "tokens_in": self.total_tokens_in - start_tokens_in,
                "tokens_out": self.total_tokens_out - start_tokens_out,
                "cost": self.total_cost_accumulated - start_cost
            }

            if self.verbose:
                print(f"[Voting] Mode: {mode}, Winner: {result['winner']}, "
                      f"Cost: ${result['cost_info']['cost']:.4f}")

            return result

        except Exception as e:
            print(f"[Voting Error] {e}")
            raise

    async def _voting_simple(
        self,
        prompt: str,
        options: list[str],
        k_margin: int,
        num_voters: int,
        votes_per_voter: int,
        session_id: str,
        voting_model: str,
        **kwargs
    ) -> dict[str, Any]:
        """Simple voting: Multiple voters vote on predefined options"""

        if not options or len(options) < 2:
            raise ValueError("Simple voting requires at least 2 options")

        if not prompt:
            prompt = "Select the best option from the given choices."

        votes = []
        vote_details = []

        # Collect votes from all voters
        for voter_id in range(num_voters):
            for vote_num in range(votes_per_voter):
                voting_prompt = f"""{prompt}

    Options:
    {chr(10).join(f"{i + 1}. {opt}" for i, opt in enumerate(options))}

    Select the best option and explain your reasoning briefly."""

                # Use a_format_class for structured voting
                vote_result = await self.a_format_class(
                    pydantic_model=SimpleVoteResult,
                    prompt=voting_prompt,
                    max_retries=2,
                    auto_context=False,
                    session_id=session_id,
                    model_preference="fast",
                    llm_kwargs={"model": voting_model} if voting_model else None,
                    **kwargs
                )

                votes.append(vote_result["option"])
                vote_details.append({
                    "voter": voter_id,
                    "vote_num": vote_num,
                    "option": vote_result["option"],
                    "reasoning": vote_result.get("reasoning", "")
                })

        # Apply First-to-ahead-by-k algorithm
        result = self._first_to_ahead_by_k(votes, k_margin)

        return {
            "mode": "simple",
            "winner": result["winner"],
            "votes": result["votes"],
            "margin": result["margin"],
            "k_margin": k_margin,
            "total_votes": result["total_votes"],
            "reached_k_margin": result["margin"] >= k_margin,
            "details": {
                "options": options,
                "vote_details": vote_details,
                "vote_history": result["history"]
            }
        }

    async def _voting_advanced(
        self,
        prompt: str,
        num_thinkers: int,
        strategy: str,
        k_margin: int,
        complex_data: bool,
        task_id: str,
        session_id: str,
        voting_model: str,
        **kwargs
    ) -> dict[str, Any]:
        """Advanced voting: Thinkers analyze, then apply strategy"""

        if not prompt:
            raise ValueError("Advanced voting requires a prompt")

        # Phase 1: Thinkers analyze the problem
        thinker_results = []
        model_pref = "complex" if complex_data else "fast"

        thinking_tasks = []
        for i in range(num_thinkers):
            thinking_prompt = f"""You are Thinker #{i + 1} of {num_thinkers}.

    {prompt}

    Provide a thorough analysis with key points and assess your confidence (0-1)."""

            task = self.a_format_class(
                pydantic_model=ThinkingResult,
                prompt=thinking_prompt,
                max_retries=2,
                auto_context=False,
                session_id=session_id,
                model_preference=model_pref,
                llm_kwargs={"model": voting_model} if voting_model else None,
                **kwargs
            )
            thinking_tasks.append(task)

        # Execute all thinking in parallel
        thinker_results = await asyncio.gather(*thinking_tasks)

        # Phase 2: Apply strategy
        if strategy == "best":
            # Select best by quality score
            best = max(thinker_results, key=lambda x: x["quality_score"])
            winner_id = f"Thinker-{thinker_results.index(best) + 1}"

            return {
                "mode": "advanced",
                "winner": winner_id,
                "votes": 1,
                "margin": 1,
                "k_margin": k_margin,
                "total_votes": 1,
                "reached_k_margin": True,
                "details": {
                    "strategy": "best",
                    "thinker_results": thinker_results,
                    "best_result": best
                }
            }

        elif strategy == "vote":
            # Vote on thinker results using fast model
            votes = []
            for _ in range(num_thinkers * 2):  # Each thinker result gets multiple votes

                vote_prompt = f"""Evaluate these analysis results and select the best one:

    {chr(10).join(f"Thinker-{i + 1}: {r['analysis'][:200]}..." for i, r in enumerate(thinker_results))}

    Select the ID of the best analysis."""

                vote = await self.a_format_class(
                    pydantic_model=VoteSelection,
                    prompt=vote_prompt,
                    max_retries=2,
                    auto_context=False,
                    session_id=session_id,
                    model_preference="fast",
                    llm_kwargs={"model": voting_model} if voting_model else None,
                    **kwargs
                )

                votes.append(vote["selected_id"])

            result = self._first_to_ahead_by_k(votes, k_margin)

            return {
                "mode": "advanced",
                "winner": result["winner"],
                "votes": result["votes"],
                "margin": result["margin"],
                "k_margin": k_margin,
                "total_votes": result["total_votes"],
                "reached_k_margin": result["margin"] >= k_margin,
                "details": {
                    "strategy": "vote",
                    "thinker_results": thinker_results,
                    "vote_history": result["history"]
                }
            }

        elif strategy == "recombine":
            # Recombine best results - use fast model for synthesis
            top_n = max(2, num_thinkers // 2)
            top_results = sorted(thinker_results, key=lambda x: x["quality_score"], reverse=True)[:top_n]

            recombine_prompt = f"""Synthesize these analyses into a superior solution:

    {chr(10).join(f"Analysis {i + 1}:{chr(10)}{r['analysis']}{chr(10)}" for i, r in enumerate(top_results))}

    Create a final synthesis that combines the best insights."""

            # Use a_run_llm_completion for final natural language output
            final_output = await self.a_run_llm_completion(
                node_name="VotingRecombine",
                task_id=task_id,
                model_preference="fast",
                with_context=False,
                auto_fallbacks=True,
                llm_kwargs={"model": voting_model} if voting_model else None,
                messages=[{"role": "user", "content": recombine_prompt}],
                session_id=session_id,
                **kwargs
            )

            return {
                "mode": "advanced",
                "winner": "recombined",
                "votes": len(top_results),
                "margin": len(top_results),
                "k_margin": k_margin,
                "total_votes": len(top_results),
                "reached_k_margin": True,
                "details": {
                    "strategy": "recombine",
                    "thinker_results": thinker_results,
                    "top_results_used": top_results,
                    "final_synthesis": final_output
                }
            }

        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    async def _voting_unstructured(
        self,
        prompt: str,
        unstructured_data: str,
        num_organizers: int,
        k_margin: int,
        vote_on_parts: bool,
        final_construction: bool,
        complex_data: bool,
        task_id: str,
        session_id: str,
        voting_model: str,
        **kwargs
    ) -> dict[str, Any]:
        """Unstructured voting: Organize data, vote, optionally construct final output"""

        if not unstructured_data:
            raise ValueError("Unstructured voting requires data")

        # Phase 1: Organizers structure the data
        model_pref = "complex" if complex_data else "fast"

        organize_tasks = []
        for i in range(num_organizers):
            organize_prompt = f"""You are Organizer #{i + 1} of {num_organizers}.

    {prompt if prompt else 'Organize the following unstructured data into a meaningful structure:'}

    Data:
    {unstructured_data}

    Create a structured organization with categories and parts."""

            task = self.a_format_class(
                pydantic_model=OrganizedData,
                prompt=organize_prompt,
                max_retries=2,
                auto_context=False,
                session_id=session_id,
                model_preference=model_pref,
                llm_kwargs={"model": voting_model} if voting_model else None,
                **kwargs
            )
            organize_tasks.append(task)

        organized_versions = await asyncio.gather(*organize_tasks)

        # Phase 2: Vote on parts or structures
        votes = []

        if vote_on_parts:
            # Collect all parts from all organizers
            all_parts = []
            for org_id, org in enumerate(organized_versions):
                for part in org["parts"]:
                    all_parts.append(f"Org{org_id + 1}-Part{part['id']}")

            # Vote on best parts using fast model
            for _ in range(len(all_parts)):
                vote_prompt = f"""Select the best organized part:

    {chr(10).join(f"{i + 1}. {part}" for i, part in enumerate(all_parts))}

    Select the ID of the best part."""

                vote = await self.a_format_class(
                    pydantic_model=VoteSelection,
                    prompt=vote_prompt,
                    max_retries=2,
                    auto_context=False,
                    session_id=session_id,
                    model_preference="fast",
                    llm_kwargs={"model": voting_model} if voting_model else None,
                    **kwargs
                )
                votes.append(vote["selected_id"])
        else:
            # Vote on complete structures
            structure_ids = [f"Structure-Org{i + 1}" for i in range(num_organizers)]

            for _ in range(num_organizers * 2):
                vote_prompt = f"""Evaluate these organizational structures:

    {chr(10).join(f"{sid}: Quality {org['quality_score']:.2f}" for sid, org in zip(structure_ids, organized_versions))}

    Select the best structure ID."""

                vote = await self.a_format_class(
                    pydantic_model=VoteSelection,
                    prompt=vote_prompt,
                    max_retries=2,
                    auto_context=False,
                    session_id=session_id,
                    model_preference="fast",
                    llm_kwargs={"model": voting_model} if voting_model else None,
                    **kwargs
                )
                votes.append(vote["selected_id"])

        vote_result = self._first_to_ahead_by_k(votes, k_margin)

        # Phase 3: Optional final construction
        final_output = None
        if final_construction:
            # Use fast model for final construction
            construct_prompt = f"""Create a final polished output based on the winning selection:

    Winner: {vote_result['winner']}
    Context: {vote_on_parts and 'individual parts' or 'complete structures'}

    Synthesize the best elements into a coherent final result."""

            # Use a_run_llm_completion for natural language final output
            final_text = await self.a_run_llm_completion(
                node_name="VotingConstruct",
                task_id=task_id,
                model_preference="fast",
                with_context=False,
                auto_fallbacks=True,
                llm_kwargs={"model": voting_model} if voting_model else None,
                messages=[{"role": "user", "content": construct_prompt}],
                session_id=session_id,
                **kwargs
            )

            final_output = {
                "output": final_text,
                "winner_used": vote_result["winner"],
                "vote_on_parts": vote_on_parts
            }

        return {
            "mode": "unstructured",
            "winner": vote_result["winner"],
            "votes": vote_result["votes"],
            "margin": vote_result["margin"],
            "k_margin": k_margin,
            "total_votes": vote_result["total_votes"],
            "reached_k_margin": vote_result["margin"] >= k_margin,
            "details": {
                "organized_versions": organized_versions,
                "vote_on_parts": vote_on_parts,
                "vote_history": vote_result["history"],
                "final_construction": final_output
            }
        }

    def _first_to_ahead_by_k(self, votes: list[str], k: int) -> dict[str, Any]:
        """
        First-to-ahead-by-k algorithm implementation.

        Returns winner when one option has k more votes than the next best.
        Based on: P(correct) = 1 / (1 + ((1-p)/p)^k)
        """
        counts = {}
        history = []

        for vote in votes:
            counts[vote] = counts.get(vote, 0) + 1
            history.append(dict(counts))

            # Check if any option is k ahead
            if len(counts) >= 2:
                sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                first, second = sorted_counts[0], sorted_counts[1]

                if first[1] - second[1] >= k:
                    return {
                        "winner": first[0],
                        "votes": first[1],
                        "margin": first[1] - second[1],
                        "history": history,
                        "total_votes": len(votes)
                    }
            elif len(counts) == 1:
                only_option = list(counts.items())[0]
                if only_option[1] >= k:
                    return {
                        "winner": only_option[0],
                        "votes": only_option[1],
                        "margin": only_option[1],
                        "history": history,
                        "total_votes": len(votes)
                    }

        # Fallback: return most voted (k-margin not reached)
        if counts:
            winner = max(counts.items(), key=lambda x: x[1])
            return {
                "winner": winner[0],
                "votes": winner[1],
                "margin": 0,
                "history": history,
                "total_votes": len(votes)
            }

        raise ValueError("No votes collected")

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
                    "persona_integration": self.amd.persona.apply_method if self.amd.persona else None
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

        # Create unique binding ID
        binding_id = f"bind_{int(time.time())}_{len(agents)}"

        # All agents in this binding (including self)
        all_agents = [self] + list(agents)

        # Create shared variable manager that all agents will reference
        shared_world_model = {}
        shared_state = {}

        # Merge existing data from all agents
        for agent in all_agents:
            # Merge world models
            shared_world_model.update(agent.world_model)
            shared_state.update(agent.shared)

        # Create shared variable manager
        shared_variable_manager = VariableManager(shared_world_model, shared_state)

        # Set up shared scopes with merged data
        for scope_name in shared_scopes:
            merged_scope = {}
            for agent in all_agents:
                if hasattr(agent, 'variable_manager') and agent.variable_manager:
                    agent_scope_data = agent.variable_manager.scopes.get(scope_name, {})
                    if isinstance(agent_scope_data, dict):
                        merged_scope.update(agent_scope_data)
            shared_variable_manager.register_scope(scope_name, merged_scope)

        # Create binding configuration
        binding_config = {
            'binding_id': binding_id,
            'agents': all_agents,
            'shared_scopes': shared_scopes,
            'auto_sync': auto_sync,
            'shared_variable_manager': shared_variable_manager,
            'private_managers': {},
            'created_at': datetime.now().isoformat()
        }

        # Configure each agent
        for i, agent in enumerate(all_agents):
            agent_private_id = f"{binding_id}_agent_{i}_{agent.amd.name}"

            # Create private variable manager for agent-specific data
            private_world_model = agent.world_model.copy()
            private_shared = agent.shared.copy()
            private_variable_manager = VariableManager(private_world_model, private_shared)

            # Set up private scopes (user, session-specific data, agent-specific configs)
            private_scopes = ['user', 'agent', 'session_private', 'tasks_private']
            for scope_name in private_scopes:
                if hasattr(agent, 'variable_manager') and agent.variable_manager:
                    agent_scope_data = agent.variable_manager.scopes.get(scope_name, {})
                    private_variable_manager.register_scope(f"{scope_name}_{agent.amd.name}", agent_scope_data)

            binding_config['private_managers'][agent.amd.name] = private_variable_manager

            # Replace agent's variable manager with a unified one
            unified_manager = UnifiedBindingManager(
                shared_manager=shared_variable_manager,
                private_manager=private_variable_manager,
                agent_name=agent.amd.name,
                shared_scopes=shared_scopes,
                auto_sync=auto_sync,
                binding_config=binding_config
            )

            # Store original managers for unbinding
            if not hasattr(agent, '_original_managers'):
                agent._original_managers = {
                    'variable_manager': agent.variable_manager,
                    'world_model': agent.world_model.copy(),
                    'shared': agent.shared.copy()
                }

            # Set new unified manager
            agent.variable_manager = unified_manager
            agent.world_model = shared_world_model
            agent.shared = shared_state

            # Update shared state with binding info
            agent.shared['binding_config'] = binding_config
            agent.shared['is_bound'] = True
            agent.shared['binding_id'] = binding_id
            agent.shared['bound_agents'] = [a.amd.name for a in all_agents]

            # Sync context manager if available
            if hasattr(agent, 'context_manager') and agent.context_manager:
                agent.context_manager.variable_manager = unified_manager

                # Share session managers between bound agents if auto_sync is enabled
                if auto_sync:
                    # Merge session managers from all agents
                    all_sessions = {}
                    for bound_agent in all_agents:
                        if hasattr(bound_agent, 'context_manager') and bound_agent.context_manager:
                            if hasattr(bound_agent.context_manager, 'session_managers'):
                                all_sessions.update(bound_agent.context_manager.session_managers)

                    # Update all agents with merged sessions
                    for bound_agent in all_agents:
                        if hasattr(bound_agent, 'context_manager') and bound_agent.context_manager:
                            bound_agent.context_manager.session_managers.update(all_sessions)

        # Set up auto-sync mechanism if enabled
        if auto_sync:
            binding_config['sync_handler'] = BindingSyncHandler(binding_config)

        rprint(f"Successfully bound {len(all_agents)} agents together (Binding ID: {binding_id})")
        rprint(f"Shared scopes: {', '.join(shared_scopes)}")
        rprint(f"Bound agents: {', '.join([agent.amd.name for agent in all_agents])}")

        return binding_config

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
                self.variable_manager = original['variable_manager']
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


class UnifiedBindingManager:
    """Unified manager that handles both shared and private variable scopes for bound agents"""

    def __init__(
        self,
        shared_manager: VariableManager,
        private_manager: VariableManager,
        agent_name: str,
        shared_scopes: list[str],
        auto_sync: bool,
        binding_config: dict,
    ):
        self.shared_manager = shared_manager
        self.private_manager = private_manager
        self.agent_name = agent_name
        self.shared_scopes = shared_scopes
        self.auto_sync = auto_sync
        self.binding_config = binding_config

    def get(self, path: str, default=None, use_cache: bool = True):
        """Get variable from appropriate manager (shared or private)"""
        scope = path.split(".")[0] if "." in path else path

        if scope in self.shared_scopes:
            return self.shared_manager.get(path, default, use_cache)
        else:
            # Try private first, then shared as fallback
            result = self.private_manager.get(path, None, use_cache)
            if result is None:
                return self.shared_manager.get(path, default, use_cache)
            return result

    def set(self, path: str, value, create_scope: bool = True):
        """Set variable in appropriate manager (shared or private)"""
        scope = path.split(".")[0] if "." in path else path

        if scope in self.shared_scopes:
            self.shared_manager.set(path, value, create_scope)
            # Auto-sync to other bound agents if enabled
            if self.auto_sync:
                self._sync_to_bound_agents(path, value)
        else:
            # Private scope - add agent identifier
            private_path = (
                f"{path}_{self.agent_name}"
                if not path.endswith(f"_{self.agent_name}")
                else path
            )
            self.private_manager.set(private_path, value, create_scope)

    def _sync_to_bound_agents(self, path: str, value):
        """Sync shared variable changes to all bound agents"""
        try:
            bound_agents = self.binding_config.get("agents", [])
            for agent in bound_agents:
                if (
                    agent.amd.name != self.agent_name
                    and hasattr(agent, "variable_manager")
                    and isinstance(agent.variable_manager, UnifiedBindingManager)
                ):
                    agent.variable_manager.shared_manager.set(
                        path, value, create_scope=True
                    )
        except Exception as e:
            wprint(f"Auto-sync failed for path {path}: {e}")

    def format_text(self, text: str, context: dict = None) -> str:
        """Format text with variables from both managers"""
        # First try private manager, then shared manager
        try:
            result = self.private_manager.format_text(text, context)
            return self.shared_manager.format_text(result, context)
        except:
            return self.shared_manager.format_text(text, context)

    def get_available_variables(self) -> dict[str, dict]:
        """Get available variables from both managers"""
        shared_vars = self.shared_manager.get_available_variables()
        private_vars = self.private_manager.get_available_variables()

        # Merge with prefix for private vars
        combined = shared_vars.copy()
        for key, value in private_vars.items():
            combined[f"private_{self.agent_name}_{key}"] = value

        return combined

    def get_scope_info(self) -> dict[str, Any]:
        """Get scope information from both managers"""
        shared_info = self.shared_manager.get_scope_info()
        private_info = self.private_manager.get_scope_info()

        return {
            "shared_scopes": shared_info,
            "private_scopes": private_info,
            "binding_info": {
                "agent_name": self.agent_name,
                "binding_id": self.binding_config.get("binding_id"),
                "auto_sync": self.auto_sync,
            },
        }

    # Delegate other methods to shared manager by default
    def __getattr__(self, name):
        return getattr(self.shared_manager, name)


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

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "kwargs", "args"):
            continue

        # Type annotation verarbeiten
        param_type = "string"  # Default
        param_description = f"Parameter {param_name}"

        if param.annotation != inspect.Parameter.empty:
            annotation = param.annotation

            # Type mapping
            type_mapping = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }

            # Handle Optional, Union, etc.
            origin = getattr(annotation, "__origin__", None)
            if origin is not None:
                # z.B. Optional[str], List[int]
                args = getattr(annotation, "__args__", ())
                if origin in (list, List):
                    param_type = "array"
                elif origin in (dict, Dict):
                    param_type = "object"
                elif args:
                    # Nimm ersten non-None Typ
                    for arg in args:
                        if arg is not type(None):
                            param_type = type_mapping.get(arg, "string")
                            break
            else:
                param_type = type_mapping.get(annotation, "string")

        # Parameter-Beschreibung aus Docstring extrahieren (wenn vorhanden)
        if docstring:
            # Suche nach ":param param_name:" oder "Args:\n    param_name:" Pattern
            import re
            param_doc_match = re.search(
                rf":param {param_name}:\s*(.+?)(?=:param|:return|$)",
                docstring,
                re.DOTALL
            )
            if param_doc_match:
                param_description = param_doc_match.group(1).strip()
            else:
                # Google-style docstring
                param_doc_match = re.search(
                    rf"{param_name}[:\s]+(.+?)(?=\n\s*\w+[:\s]|\n\n|$)",
                    docstring,
                    re.DOTALL
                )
                if param_doc_match:
                    param_description = param_doc_match.group(1).strip()

        properties[param_name] = {
            "type": param_type,
            "description": param_description
        }

        # Required wenn kein Default
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

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
            if query == "r":
                res = await agent.explain_reasoning_process()
                print(res)
                continue
            if query == "exit":
                break
            response = await agent.a_run(query)
            print(f"Response: {response}")

        await agent.close()

    asyncio.run(_agent())

