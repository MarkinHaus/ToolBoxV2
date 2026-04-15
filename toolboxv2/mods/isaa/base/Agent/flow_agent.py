"""
FlowAgent V2 - Production-ready Agent System

Refactored architecture:
- SessionManager: Session lifecycle with ChatSession integration
- ToolManager: Unified tool registry (local, MCP, A2A)
- CheckpointManager: Full state persistence
- BindManager: Agent-to-agent binding
- ExecutionEngine: MAKER/RLM inspired orchestration with Pause/Continue

Author: FlowAgent V2
"""
import asyncio
import contextlib
import copy
import io
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Generator, Union, Optional

import yaml
from pydantic import BaseModel

from toolboxv2 import Style, get_logger, get_app
from toolboxv2.mods.isaa.base.Agent.chain import Chain, ConditionalChain
from toolboxv2.mods.isaa.base.Agent.types import (
    AgentModelData,
)
from toolboxv2.mods.isaa.base.patch.power_vfs import grep_vfs, search_vfs, find_files
from toolboxv2.mods.isaa.base.audio_io.audioIo import (
    setup_isaa_audio, LocalPlayer, WebPlayer, NullPlayer, AudioStreamPlayer
)
from toolboxv2.mods.isaa.base.audio_io.Tts import TTSConfig, TTSBackend
AGENT_VERBOSE = os.environ.get("AGENT_VERBOSE", "false").lower() == "true"
# Framework imports
try:
    #import litellm
    ## Unterdrückt die störenden LiteLLM Konsolen-Logs
    #logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    #logging.getLogger("litellm").setLevel(logging.WARNING)
    #litellm.suppress_debug_info = not AGENT_VERBOSE
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
try:
    # from python_a2a import A2AServer, AgentCard
    class A2AServer:
        pass

    class AgentCard:
        pass
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False

try:
    #from mcp.server.fastmcp import FastMCP
    class FastMCP:
        pass
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

try:
    gmail_toolkit = None
    calendar_toolkit = None
    if os.getenv("WITH_GOOGLE_TOOLS", "false") == "true":
        from toolboxv2.mods.isaa.extras.toolkit.google_calendar_toolkit import (
            CalendarToolkit,
        )
        from toolboxv2.mods.isaa.extras.toolkit.google_gmail_toolkit import GmailToolkit
        google_token_dir = os.getenv("GOOGLE_TOKEN_DIR", "token")
        if not os.path.exists(google_token_dir):
            os.makedirs(google_token_dir, exist_ok=True)
        gmail_toolkit = GmailToolkit(
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            token_dir=google_token_dir,
        )
        calendar_toolkit = CalendarToolkit(
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            token_dir=google_token_dir,
        )
except ImportError as e:
    gmail_toolkit = None
    calendar_toolkit = None
    if os.getenv("WITH_GOOGLE_TOOLS", "false") == "true":
        print(f"⚠️ Google tools not available: {e}")

logger = get_logger()

MAX_CONTINUATIONS = os.environ.get("AGENT_INTERN_MAX_CONTINUATIONS", 5)


# ===== MEDIA PARSING UTILITIES =====
def parse_media_from_query(query: str) -> tuple[str, list[dict]]:
    """
    Parse [media:(path/url)] tags from query and convert to litellm vision format

    Args:
        query: Text query that may contain [media:(path/url)] tags

    Returns:
        tuple: (cleaned_query, media_list)
            - cleaned_query: Query with media tags removed
            - media_list: List of dicts in litellm vision format

    Examples:
        >>> parse_media_from_query("Analyze [media:image.jpg] this image")
        ("Analyze  this image", [{"type": "image_url", "image_url": {"url": "image.jpg", "format": "image/jpeg"}}])

    Note:
        litellm uses the OpenAI vision format: {"type": "image_url", "image_url": {"url": "...", "format": "..."}}
        The "format" field is optional but recommended for explicit MIME type specification.
    """
    media_pattern = r"\[media:([^\]]+)\]"
    media_matches = re.findall(media_pattern, query)

    media_list = []
    for media_path in media_matches:
        media_path = media_path.strip()

        # Determine media type from extension or URL
        media_type = _detect_media_type(media_path)

        # litellm uses image_url format for vision models
        # Format: {"type": "image_url", "image_url": {"url": "...", "format": "image/jpeg"}}
        if media_type == "image":
            # Detect image format for explicit MIME type
            mime_type = _get_image_mime_type(media_path)
            image_obj = {"url": media_path}
            if mime_type:
                image_obj["format"] = mime_type

            media_list.append({"type": "image_url", "image_url": image_obj})
        elif media_type in ["audio", "video", "pdf"]:
            # For non-image media, some models may support them
            # but we use image_url as the standard format
            # The model will handle or reject based on its capabilities
            if AGENT_VERBOSE:
                print(
                    f"Warning: Media type '{media_type}' detected. Not all models support non-image media."
                )
            media_list.append({"type": "image_url", "image_url": {"url": media_path}})
        else:
            # Unknown type - try as image
            media_list.append({"type": "image_url", "image_url": {"url": media_path}})

    # Remove media tags from query
    cleaned_query = re.sub(media_pattern, "", query).strip()
    return cleaned_query, media_list


def _detect_media_type(path: str) -> str:
    """Detect media type from file extension or URL"""
    path_lower = path.lower()

    # Image extensions
    if any(
        path_lower.endswith(ext)
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"]
    ):
        return "image"

    # Audio extensions
    if any(
        path_lower.endswith(ext)
        for ext in [".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac"]
    ):
        return "audio"

    # Video extensions
    if any(
        path_lower.endswith(ext)
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]
    ):
        return "video"

    # PDF
    if path_lower.endswith(".pdf"):
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
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".ico": "image/x-icon",
    }

    for ext, mime in mime_map.items():
        if path_lower.endswith(ext):
            return mime

    return ""


# Muti media Extention
MEDIA_ERROR_PATTERNS = [
    r"unsupported.*(?:media|image|file).*type",
    r"(?:media|image|file).*type.*(?:not supported|unsupported)",
    r"cannot process.*(?:media|image|file)",
    r"invalid.*(?:media|image|file).*format",
    r"(?:media|image).*(?:could not be|cannot be).*processed",
    r"unable to.*(?:read|process|decode).*(?:media|image|file)",
    r"(?:400|422).*(?:media|image)",
    r"content.*type.*not.*(?:allowed|supported|valid)",
    r"(?:pdf|audio|video).*not.*supported",
]


def _is_media_error(error: Exception) -> bool:
    """Prüft ob ein Fehler ein Media-Verarbeitungsfehler ist"""
    error_str = str(error).lower()
    return any(re.search(p, error_str, re.IGNORECASE) for p in MEDIA_ERROR_PATTERNS)


def _extract_failed_media_type(error: Exception) -> str | None:
    """Extrahiert den fehlgeschlagenen Medientyp aus der Fehlermeldung"""
    error_str = str(error).lower()
    for media_type in ["pdf", "audio", "video", "mp3", "wav", "mp4", "avi"]:
        if media_type in error_str:
            return media_type
    return None


def _remove_media_by_type(
    messages: list[dict], types_to_remove: list[str]
) -> tuple[list[dict], list[dict]]:
    """Entfernt bestimmte Medientypen aus den Messages"""
    cleaned = []
    removed = []

    type_extensions = {
        "pdf": [".pdf"],
        "audio": [".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac"],
        "video": [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"],
        "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"],
    }

    def should_remove(url: str) -> tuple[bool, str]:
        url_lower = url.lower()
        for media_type, extensions in type_extensions.items():
            if media_type in types_to_remove:
                if any(url_lower.endswith(ext) for ext in extensions):
                    return True, media_type
        return False, ""

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            cleaned.append(msg)
            continue

        new_content = []
        for part in content:
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                remove, media_type = should_remove(url)
                if remove:
                    removed.append({"path": url, "media_type": media_type})
                else:
                    new_content.append(part)
            else:
                new_content.append(part)

        if new_content:
            if len(new_content) == 1 and new_content[0].get("type") == "text":
                cleaned.append({"role": msg["role"], "content": new_content[0]["text"]})
            else:
                cleaned.append({"role": msg["role"], "content": new_content})

    return cleaned, removed


def _inject_media_notice(messages: list[dict], removed: list[dict]) -> list[dict]:
    """Fügt einen Hinweis über entfernte Medien ein"""
    if not removed:
        return messages

    notice_items = [f"  - {m['path']} (Typ: {m['media_type']})" for m in removed]
    notice = (
        f"\n\n[System-Hinweis: Folgende Medientypen werden nicht automatisch unterstützt "
        f"und wurden entfernt:\n{chr(10).join(notice_items)}\n"
        f"Für diese Dateien sind zusätzliche Tools/Schritte erforderlich "
        f"(z.B. PDF-Extraktion, Audio-Transkription, Video-Frame-Analyse).]"
    )

    result = copy.deepcopy(messages)
    for i in range(len(result) - 1, -1, -1):
        if result[i]["role"] == "user":
            content = result[i]["content"]
            if isinstance(content, str):
                result[i]["content"] = content + notice
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        part["text"] += notice
                        break
            break
    return result


class FlowAgent:
    """Production-ready autonomous agent with session isolation."""

    def __init__(
        self,
        amd: AgentModelData,
        verbose: bool = False,
        max_parallel_tasks: int = 3,
        auto_load_checkpoint: bool = True,
        rule_config_path: str | None = None,
        progress_callback: Callable | None = None,
        stream: bool = True,
        **kwargs,
    ):
        self._dreamer = None
        self._vison = {"fast":None, "coplex":None}
        self.amd : AgentModelData= amd
        self.verbose = verbose
        self.stream = stream
        self._rule_config_path = rule_config_path

        self.is_running = False
        self.active_session: str | None = None
        self.active_execution_id: str | None = None

        # Statistics
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost_accumulated = 0.0
        self.total_llm_calls = 0

        # Servers
        self.a2a_server: A2AServer | None = None
        self.mcp_server: FastMCP | None = None

        # Execution engine instance (lazy loaded)
        self._execution_engine_cache = None

        self._init_managers(auto_load_checkpoint)
        self._init_rate_limiter()

        logger.info(f"FlowAgent '{amd.name}' initialized")

    def _init_managers(self, auto_load_checkpoint: bool):
        from toolboxv2.mods.isaa.base.Agent.bind_manager import BindManager
        from toolboxv2.mods.isaa.base.Agent.checkpoint_manager import CheckpointManager
        from toolboxv2.mods.isaa.base.Agent.docker_vfs import DockerConfig
        from toolboxv2.mods.isaa.base.Agent.session_manager import SessionManager
        from toolboxv2.mods.isaa.base.Agent.tool_manager import ToolManager

        self.session_manager = SessionManager(
            agent_name=self.amd.name,
            default_max_history=os.getenv("DEFAULT_MAX_HISTORY_LENGTH", 100),
            vfs_max_window_lines=self.amd.vfs_max_window_lines,
            rule_config_path=self._rule_config_path,
            summarizer=self._create_summarizer(),
            enable_lsp=self.amd.enable_lsp,
            enable_docker=self.amd.enable_docker,
            docker_config=self.amd.docker_config
            or DockerConfig(memory_limit="4g", timeout_seconds=600),
            toolboxv2_wheel_path=os.getenv(
                "TOOLBV2_WHEEL_PATH",
                "C:/Users/Markin/Workspace/ToolBoxV2/dist/toolboxv2-0.1.24-py2.py3-none-any.whl",
            ),
        )

        self.tool_manager = ToolManager()

        self.checkpoint_manager = CheckpointManager(
            agent=self, auto_load=auto_load_checkpoint
        )

        self.bind_manager = BindManager(agent=self)

    def _init_rate_limiter(self):
        from toolboxv2.mods.isaa.base.IntelligentRateLimiter.intelligent_rate_limiter import (
            LiteLLMRateLimitHandler,
            create_handler_from_config,
            load_handler_from_file,
        )

        if isinstance(self.amd.handler_path_or_dict, dict):
            self.llm_handler = create_handler_from_config(self.amd.handler_path_or_dict)
        elif isinstance(self.amd.handler_path_or_dict, str) and os.path.exists(
            self.amd.handler_path_or_dict
        ):
            self.llm_handler = load_handler_from_file(self.amd.handler_path_or_dict)
        else:
            self.llm_handler = LiteLLMRateLimitHandler(max_retries=3)

    def _create_summarizer(self) -> Callable:
        async def summarize(content: str) -> str:
            try:
                result = await self.a_run_llm_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Summarize in 1-2 sentences:\n\n{content[:8000]}",
                        }
                    ],
                    max_tokens=100,
                    temperature=0.3,
                    with_context=False,
                    model_preference="fast",
                    task_id="vfs_summarize",
                )
                return result.strip()
            except Exception:
                return f"[{len(content)} chars]"

        return summarize

    # =========================================================================
    # CORE: a_run_llm_completion
    # =========================================================================

    @staticmethod
    def _process_media_in_messages(messages: list[dict]) -> list[dict]:
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
            if "[media:" in content:
                cleaned_content, media_list = parse_media_from_query(content)

                if media_list:
                    # Convert to multi-modal message format for litellm
                    # Format: content becomes a list with text and media items
                    content_parts = []

                    # Add text part if there's any text left
                    if cleaned_content.strip():
                        content_parts.append({"type": "text", "text": cleaned_content})

                    # Add media parts
                    content_parts.extend(media_list)

                    processed_messages.append(
                        {"role": msg["role"], "content": content_parts}
                    )
                else:
                    # No valid media found, keep original
                    processed_messages.append(msg)
            else:
                # No media tags, keep original
                processed_messages.append(msg)
        return processed_messages

    async def save_supports_vision(self, messages:list|None=None,model_preference="fast"):
        if hasattr(self, "_vison"):
            self._vison[model_preference] = self._vison.get(model_preference)
            if isinstance(self._vison[model_preference], bool):
                return self._vison[model_preference]
        model = self.amd.fast_llm_model if model_preference == "fast" else self.amd.complex_llm_model
        provider = model.split('/')[-2]
        try:
            from litellm import supports_vision
            self._vison[model_preference] = supports_vision(model)
            return self._vison[model_preference]
        except:
            try:
                from litellm import supports_vision
                self._vison[model_preference] = supports_vision(model, provider)
                return self._vison[model_preference]
            except:
                import litellm
                has_media = False
                if not messages:
                    return False
                for msg in messages:
                    if not isinstance(msg.get("content"), str):
                        continue

                    content = msg["content"]

                    if not content:
                        continue

                    # Check if content contains media tags
                    if "[media:" in content:
                        has_media = True
                        break
                if not has_media:
                    return False

                processed_messages = self._process_media_in_messages(messages.copy())
                try:
                    response = await self.llm_handler.completion_with_rate_limiting(
                        litellm, model=model, messages=processed_messages, max_tokens=1)
                    self._vison[model_preference] = True
                    return self._vison[model_preference]
                except:
                    self._vison[model_preference] = False
                    return self._vison[model_preference]



    async def a_run_llm_completion(
        self,
        messages: list[dict],
        model_preference: str = "fast",
        with_context: bool = True,
        stream: bool | None = None,
        get_response_message: bool = False,
        task_id: str = "unknown",
        session_id: str | None = None,
        do_tool_execution: bool = False,
        # NEU: Parameter für Media-Retry
        _media_retry: int = 0,
        _removed_types: list[str] | None = None,
        **kwargs,
    ) -> str | Any:
        if not LITELLM_AVAILABLE:
            raise RuntimeError("LiteLLM required")

        _removed_types = _removed_types or []

        model = kwargs.pop("model", None) or (
            self.amd.fast_llm_model
            if model_preference == "fast"
            else self.amd.complex_llm_model
        )
        use_stream = stream if stream is not None else self.stream

        # NEU: Verarbeite Media in Messages
        if await self.save_supports_vision(messages, model_preference):
            processed_messages = self._process_media_in_messages(messages.copy())
        else:
            processed_messages = messages.copy()

        # NEU: Entferne bereits bekannte problematische Typen
        if _removed_types:
            processed_messages, newly_removed = _remove_media_by_type(
                processed_messages, _removed_types
            )
            if newly_removed:
                processed_messages = _inject_media_notice(
                    processed_messages, newly_removed
                )

        true_stream = kwargs.pop("true_stream", False)

        llm_kwargs = {
            "model": model,
            "messages": processed_messages,
            "stream": use_stream,
            "request_timeout": 300,
            **kwargs,
        }

        # Ollama ignoriert tool_choice='auto' bei einigen Modellen
        if model.startswith("ollama") and llm_kwargs.get("tool_choice") == "auto":
            llm_kwargs.pop("tool_choice", None)

        session_id = session_id or self.active_session
        system_msg = self.amd.get_system_message()
        session = None
        if session_id:
            session = self.session_manager.get(session_id)
            if session:
                await session.initialize()
                system_msg += "\n\n" + session.build_vfs_context()
        if with_context:
            if session:
                sysmsg = [{"role": "system", "content": f"{system_msg}"}]
                full_history = session.get_history(kwargs.get("history_size", 6))
                current_msg = llm_kwargs["messages"]
                for msg in full_history:
                    if not current_msg:
                        break
                    if msg["role"] != "user":
                        continue
                    content = msg["content"]
                    if (
                        current_msg[0]["role"] == "user"
                        and current_msg[0]["content"] == content
                    ):
                        current_msg = current_msg[1:]
                        break
                    if (
                        len(current_msg) > 1
                        and current_msg[-1]["role"] == "user"
                        and current_msg[-1]["content"] == content
                    ):
                        current_msg = current_msg[:-1]
                        break
                llm_kwargs["messages"] = sysmsg + full_history + current_msg
            else:
                llm_kwargs["messages"] = [
                    {"role": "system", "content": f"{system_msg}"}
                ] + llm_kwargs["messages"]

        # NEU: Prompt Hash für Audit-Log
        import hashlib
        import json
        import time
        def safe_serializer(obj):
            # If OpenAI tool call object
            if hasattr(obj, "model_dump"):
                return obj.model_dump()

            # Fallback: convert object to dict if possible
            if hasattr(obj, "__dict__"):
                return obj.__dict__

            # Final fallback
            return str(obj)

        prompt_content = json.dumps(llm_kwargs["messages"], sort_keys=True, default=safe_serializer)
        prompt_hash = hashlib.sha256(prompt_content.encode()).hexdigest()[:16]
        start_time = time.time()

        # Get user_id from session if available
        user_id = "anonymous"
        if session_id:
            session = self.session_manager.get(session_id)
            if session and hasattr(session, 'user_id'):
                user_id = session.user_id
            elif session and hasattr(session, 'session_id'):
                user_id = session.session_id

        try:
            from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message
            import litellm
            litellm.drop_params = True
            litellm.suppress_debug_info = True
            original_messages = llm_kwargs["messages"].copy()
            original_tools = llm_kwargs.get("tools")

            continuation_count = 0

            final_text_content = ""
            final_tool_calls_raw = []
            accumulated_usage = None

            active_cut_off_tool = None

            # Edge Case für True-Streaming (wird vom UI-Streaming direkt an Nutzer geleitet)
            if use_stream and true_stream:
                if use_stream:
                    llm_kwargs["stream_options"] = {"include_usage": True}
                return await self.llm_handler.completion_with_rate_limiting(litellm, **llm_kwargs)

            # --- AUTO-RESUME SCHLEIFE (bis zu 100x Output Limit!) ---
            while continuation_count < MAX_CONTINUATIONS:

                if use_stream:
                    llm_kwargs["stream_options"] = {"include_usage": True}

                response = await self.llm_handler.completion_with_rate_limiting(
                    litellm, **llm_kwargs
                )

                chunk_text = ""
                chunk_tool_calls = []
                finish_reason = None
                current_usage = None

                # Datenextrahierung abhängig davon, ob es ein Stream ist oder nicht
                if use_stream:
                    result_obj, current_usage, finish_reason = await self._process_streaming_response(
                        response, task_id, model, get_response_message=True
                    )
                    chunk_text = result_obj.content or ""
                    chunk_tool_calls = getattr(result_obj, "tool_calls", []) or []
                else:
                    msg_obj = response.choices[0].message
                    chunk_text = msg_obj.content or ""
                    chunk_tool_calls = msg_obj.tool_calls or []
                    finish_reason = response.choices[0].finish_reason
                    current_usage = response.usage

                # --- 1. USAGE MERGEN ---
                if current_usage:
                    if accumulated_usage is None:
                        accumulated_usage = current_usage
                    else:
                        if hasattr(current_usage, "prompt_tokens"):
                            accumulated_usage.prompt_tokens += current_usage.prompt_tokens
                            accumulated_usage.completion_tokens += current_usage.completion_tokens
                            accumulated_usage.total_tokens += current_usage.total_tokens

                # --- 2. CONTENT & TOOL CALLS MERGEN ---
                if active_cut_off_tool is not None:
                    # Das LLM versucht gerade, einen Tool-Call (als reinen Text-Output) fortzusetzen.
                    active_cut_off_tool["args"] += chunk_text

                    # Manchmal schieben Modelle es trotzdem in den nativen tool_calls Parameter. Das fangen wir ab:
                    for tc in chunk_tool_calls:
                        if tc.function and tc.function.arguments:
                            active_cut_off_tool["args"] += tc.function.arguments

                    # Prüfen ob der fortgesetzte JSON-String nun valide ist
                    try:
                        json.loads(active_cut_off_tool["args"])
                        # Success! Er ist nun vollständig generiert.
                        final_tool_calls_raw.append(active_cut_off_tool)
                        active_cut_off_tool = None
                    except json.JSONDecodeError:
                        pass  # Er ist weiterhin unvollständig, wir brauchen noch eine Iteration.
                else:
                    final_text_content += chunk_text

                    for idx, tc in enumerate(chunk_tool_calls):
                        tc_dict = {
                            "id": getattr(tc, "id", f"call_{uuid.uuid4().hex[:8]}"),
                            "name": tc.function.name,
                            "args": tc.function.arguments
                        }
                        # Wenn wir am Token-Limit sind und dies der LETZTE Tool-Call im Chunk ist
                        if finish_reason in ["length", "max_tokens"] and idx == len(chunk_tool_calls) - 1:
                            try:
                                json.loads(tc_dict["args"])
                                final_tool_calls_raw.append(tc_dict)  # War zum Glück vollständig
                            except json.JSONDecodeError:
                                active_cut_off_tool = tc_dict  # Ist abgerissen! Speichern für die nächste Iteration.
                        else:
                            final_tool_calls_raw.append(tc_dict)

                # --- 3. STOPP ODER FORTSETZUNG? ---
                if finish_reason not in ["length", "max_tokens"]:
                    break  # Generierung freiwillig beendet (stop, tool_calls, etc.)

                # Token Limit erreicht - wir zwingen das LLM zur exakten Fortsetzung!
                continuation_count += 1
                if AGENT_VERBOSE:
                    logger.info(
                        f"LLM Max Output Token Limit erreicht. Auto-Resume ({continuation_count}/{MAX_CONTINUATIONS})...")

                if active_cut_off_tool is not None:
                    # Ein Tool-Call ist abgerissen. Wir zwingen das LLM, nur noch die fehlenden JSON-Teile (als normalen Text) auszugeben.
                    resume_msg = (
                        f"Du hast das Output-Token-Limit erreicht, während du das Tool '{active_cut_off_tool['name']}' ausgeführt hast. "
                        f"Hier ist der JSON-Argument-String, den du bisher geschrieben hast:\n`{active_cut_off_tool['args']}`\n\n"
                        f"Bitte antworte AUSSCHLIESSLICH mit den fehlenden Zeichen, um das JSON zu vervollständigen. "
                        f"Gib keine Erklärungen und keinen Code-Block ab. Mache genau da weiter, wo der String abgerissen ist."
                    )
                    llm_kwargs["tools"] = None  # Verhindert, dass das LLM es erneut in native tools kapselt

                    # Alte Tool-Calls aus der History löschen (API würde meckern wegen "Incomplete Tool Call")
                    llm_kwargs["messages"] = original_messages + [
                        {"role": "assistant",
                         "content": final_text_content + f"\n[Starte Tool: {active_cut_off_tool['name']}]"},
                        {"role": "user", "content": resume_msg}
                    ]
                else:
                    # Reiner Text-Output ist abgerissen. Wir versuchen den letzten Satz/Zeilenanfang als Kontext zu geben.
                    last_words = final_text_content[-100:]
                    resume_msg = (
                        f"Du hast das maximale Output-Token-Limit deines Modells erreicht. "
                        f"Bitte fahre exakt an dem Punkt fort, an dem du aufgehört hast. "
                        f"Hier sind deine letzten Worte zur Orientierung:\n'...{last_words}'\n\n"
                        f"Setze den Text/Code lückenlos fort. Bitte benutze keine Floskeln wie 'Hier geht es weiter'."
                    )
                    llm_kwargs["tools"] = original_tools
                    llm_kwargs["messages"] = original_messages + [
                        {"role": "assistant", "content": final_text_content},
                        {"role": "user", "content": resume_msg}
                    ]

            # --- NACH DER SCHLEIFE: ZUSAMMENFÜHREN ---
            reconstructed_tool_calls = []
            for tc in final_tool_calls_raw:
                reconstructed_tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=tc["id"],
                        type="function",
                        function=Function(name=tc["name"], arguments=tc["args"])
                    )
                )

            final_message = Message(
                role="assistant",
                content=final_text_content or None,
                tool_calls=reconstructed_tool_calls if reconstructed_tool_calls else None
            )

            # Statistik Update
            input_tokens = accumulated_usage.prompt_tokens if accumulated_usage else 0
            output_tokens = accumulated_usage.completion_tokens if accumulated_usage else 0
            cost = self.calculate_llm_cost(model, input_tokens, output_tokens, None)

            self.total_tokens_in += input_tokens
            self.total_tokens_out += output_tokens
            self.total_cost_accumulated += cost
            self.total_llm_calls += (continuation_count + 1)

            result_to_return = final_message if get_response_message else (final_text_content or "")

            # Wenn Tool-Ausführung in FlowAgent verlangt war
            if do_tool_execution and original_tools and reconstructed_tool_calls:
                tool_response = await self.run_tool_response(final_message, session_id)
                llm_kwargs["messages"] = original_messages + [
                    {
                        "role": "assistant",
                        "content": final_text_content,
                        "tool_calls": final_message.tool_calls
                    }
                ] + tool_response
                llm_kwargs["tools"] = original_tools

                return await self.a_run_llm_completion(
                    llm_kwargs["messages"],
                    model_preference,
                    with_context,
                    stream,
                    get_response_message,
                    task_id,
                    session_id,
                    _media_retry=_media_retry,
                    _removed_types=_removed_types,
                    **kwargs,
                )

            # NEU: Audit-Log Success (mit echten Kosten aus der Response)
            try:
                execution_time = time.time() - start_time

                # Echte Kosten und Token aus der Response holen
                input_tokens = 0
                output_tokens = 0
                cost = 0

                if accumulated_usage:
                    if hasattr(accumulated_usage, "prompt_tokens"):
                        input_tokens = accumulated_usage.prompt_tokens or 0
                        output_tokens = accumulated_usage.completion_tokens or 0
                    elif hasattr(accumulated_usage, "_asdict"):
                        # Dict-Form probieren
                        usage_dict = accumulated_usage._asdict() if hasattr(accumulated_usage, "_asdict") else {}
                        input_tokens = usage_dict.get("prompt_tokens", 0)
                        output_tokens = usage_dict.get("completion_tokens", 0)

                # Prüfe auf LiteLLM _hidden_params für echte Kosten
                if "result_to_return" in locals() and result_to_return and hasattr(result_to_return, "_hidden_params"):
                    hidden_params = getattr(result_to_return, "_hidden_params", None) or {}
                    if hidden_params:
                        input_tokens = hidden_params.get("prompt_tokens", input_tokens)
                        output_tokens = hidden_params.get("completion_tokens", output_tokens)
                        cost = hidden_params.get("llm_cost", 0) or 0

                get_app("audit-isaa").audit_logger.log_action(
                    user_id=user_id,
                    action="llm.call.success",
                    resource=f"/llm/{model}",
                    status="SUCCESS",
                    details={
                        "model": model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost": cost,
                        "duration": round(execution_time, 2),
                        "prompt_hash": prompt_hash,
                        "continuation_count": continuation_count,
                        "session_id": session_id or "none",
                    }
                )
            except Exception:
                pass  # Audit-Errors dürfen nicht crashen

            return result_to_return

        except Exception as e:
            # =====================================================================
            # NEU: Intelligente Media-Fehlerbehandlung mit Auto-Retry
            # =====================================================================
            if _is_media_error(e) and _media_retry < 3:
                logger.warning(f"Media-Fehler erkannt (Versuch {_media_retry + 1}): {e}")

                # Bestimme welchen Typ entfernen
                failed_type = _extract_failed_media_type(e)

                if failed_type:
                    new_types = list(set(_removed_types + [failed_type]))
                else:
                    # Entferne schrittweise: erst pdf, dann audio, dann video
                    priority_order = ["pdf", "audio", "video", "image"]
                    for ptype in priority_order:
                        if ptype not in _removed_types:
                            new_types = _removed_types + [ptype]
                            break
                    else:
                        new_types = _removed_types + ["image"]  # Fallback

                logger.info(f"Retry ohne Media-Typen: {new_types}")

                return await self.a_run_llm_completion(
                    messages,  # Original messages!
                    model_preference,
                    with_context,
                    stream,
                    get_response_message,
                    task_id,
                    session_id,
                    do_tool_execution,
                    _media_retry=_media_retry + 1,
                    _removed_types=new_types,
                    **kwargs,
                )
            # =====================================================================

            logger.error(f"LLM call failed: {e}")

            # NEU: Audit-Log Error (vollständige Fehlermeldung)
            try:
                execution_time = time.time() - start_time
                error_msg = str(e)

                # Erste 200 + letzte 200 Zeichen für vollständiges Bild
                if len(error_msg) > 400:
                    error_msg_truncated = error_msg[:200] + " [...] " + error_msg[-200:]
                else:
                    error_msg_truncated = error_msg

                get_app("audit-isaa").audit_logger.log_action(
                    user_id=user_id,
                    action="llm.call.error",
                    resource=f"/llm/{model}",
                    status="FAILURE",
                    details={
                        "model": model,
                        "error_type": type(e).__name__,
                        "error_message": error_msg_truncated,
                        "error_length": len(error_msg),
                        "duration": round(execution_time, 2),
                        "prompt_hash": prompt_hash,
                        "session_id": session_id or "none",
                    }
                )
            except Exception:
                pass  # Audit-Errors dürfen nicht crashen

            raise

    @staticmethod
    def calculate_llm_cost(
        model: str,
        input_tokens: int,
        output_tokens: int,
        completion_response: Any = None,
    ) -> float:
        """Calculate approximate LLM cost"""
        cost = (input_tokens / 1000) * 0.002 + (output_tokens / 1000) * 0.01
        if hasattr(completion_response, "_hidden_params"):
            cost = completion_response._hidden_params.get("response_cost", 0)

        try:
            devnull = open(os.devnull, "w")
        except Exception as e:
            devnull = io.StringIO()
            if os.getenv("AGENT_VERBOSE", "false") == "true":
                print(e, "\nWhile trying to get devnull")

        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            try:
                import litellm

                cost = litellm.completion_cost(
                    model=model, completion_response=completion_response
                )
            except ImportError:
                pass
            except Exception as e:
                try:
                    import litellm

                    cost = litellm.completion_cost(
                        model=model.split("/")[-1], completion_response=completion_response
                    )
                except Exception:
                    pass
        return cost or (input_tokens / 1000) * 0.002 + (output_tokens / 1000) * 0.01


    async def _process_streaming_response(self, response, task_id, model, get_response_message):
        from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message

        result = ""
        tool_calls_acc = {}
        final_chunk = None
        finish_reason = None

        async for chunk in response:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            result += content
            if hasattr(self, 'stream_callback') and self.stream_callback:
                res = self.stream_callback(content)
                if res and asyncio.iscoroutine(res):
                    await res


            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = ChatCompletionMessageToolCall(
                            id=tc.id,
                            type="function",
                            function=Function(name="", arguments=""),
                        )
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx].function.name += tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx].function.arguments += tc.function.arguments

            if hasattr(chunk.choices[0], "finish_reason") and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            final_chunk = chunk

        usage = final_chunk.usage if hasattr(final_chunk, "usage") else None

        if get_response_message:
            result = Message(
                role="assistant",
                content=result or None,
                tool_calls=list(tool_calls_acc.values()) if tool_calls_acc else [],
            )

        return result, usage, finish_reason

    async def run_tool_response(self, response, session_id):
        tool_calls = response.tool_calls
        session = None
        if session_id:
            session = self.session_manager.get(session_id)
        all_results = []
        for tc in tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments or "{}")
            try:
                result = await self.arun_function(tool_name, **tool_args)
            except Exception as e:
                result = f"Error: {str(e)}"
            tool_response = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            }
            all_results.append(tool_response)
            if session:
                await session.add_message(tool_response)
        return all_results

    # =========================================================================
    # CORE: arun_function
    # =========================================================================

    async def arun_function(self, function_name: str, **kwargs) -> Any:
        if self.active_session:
            session = self.session_manager.get(self.active_session)
            if session and not session.is_tool_allowed(function_name):
                raise PermissionError(
                    f"Tool '{function_name}' restricted in session '{self.active_session}'"
                )

        start_time = time.perf_counter()
        result = await self.tool_manager.execute(function_name, **kwargs)

        return result

    # =========================================================================
    # CORE: a_format_class
    # =========================================================================

    async def a_format_class(
        self,
        pydantic_model: type[BaseModel],
        prompt: str,
        message_context: list[dict] | None = None,
        max_retries: int = 1,
        model_preference: str = "fast",
        auto_context: bool = False,
        max_tokens: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        schema = pydantic_model.model_json_schema()
        model_name = pydantic_model.__name__

        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        fields_desc = [
            f"  {name}{'*' if name in required else ''}: {info.get('type', 'string')}"
            for name, info in props.items()
        ]

        enhanced_prompt = f"{prompt}"

        try:
            from litellm import supports_response_schema

            for mp in [
                model_preference,
                "complex" if model_preference == "fast" else "fast",
            ]:
                data = await self.a_run_llm_completion(
                    messages=(message_context or [])
                    + [{"role": "user", "content": enhanced_prompt}],
                    model_preference=mp,
                    stream=False,
                    with_context=auto_context,
                    max_tokens=max_tokens,
                    task_id=f"format_{model_name.lower()}",
                    response_format=pydantic_model,
                )
                if isinstance(data, str):
                    data = json.loads(data)
                validated = pydantic_model.model_validate(data)
                return validated.model_dump()

        except ImportError as e:
            logger.error(f"LLM call failed: {e}")
            print("LLM call failed:", e, "falling back to YAML")

        messages = (message_context or []) + [
            {
                "role": "system",
                "content": "You are a YAML formatter. format the input to valid YAML.",
            },
            {"role": "user", "content": enhanced_prompt},
            {
                "role": "system",
                "content": "Return YAML with fields:\n" + "\n".join(fields_desc),
            },
        ]

        for attempt in range(max_retries + 1):
            try:
                response = await self.a_run_llm_completion(
                    messages=messages,
                    model_preference=model_preference,
                    stream=False,
                    with_context=auto_context,
                    temperature=0.1 + (attempt * 0.1),
                    max_tokens=max_tokens,
                    task_id=f"format_{model_name.lower()}_{attempt}",
                )

                if not response or not response.strip():
                    raise ValueError("Empty response")

                yaml_content = self._extract_yaml_content(response)
                if not yaml_content:
                    raise ValueError("No YAML found")

                parsed_data = yaml.safe_load(yaml_content)
                if not isinstance(parsed_data, dict):
                    raise ValueError(f"Expected dict, got {type(parsed_data)}")

                validated = pydantic_model.model_validate(parsed_data)
                return validated.model_dump()

            except Exception as e:
                if attempt < max_retries:
                    messages[-1]["content"] = enhanced_prompt + f"\n\nFix error: {str(e)}"
                else:
                    raise RuntimeError(f"Failed after {max_retries + 1} attempts: {e}")

    @staticmethod
    def _extract_yaml_content(response: str) -> str:
        if "```yaml" in response:
            try:
                return response.split("```yaml")[1].split("```")[0].strip()
            except IndexError:
                pass
        if "```" in response:
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    lines = part.strip().split("\n")
                    if len(lines) > 1:
                        return (
                            "\n".join(lines[1:]).strip()
                            if lines[0].strip().isalpha()
                            else part.strip()
                        )
        if ":" in response and not response.strip().startswith("<"):
            return response.strip()
        return ""

    # =========================================================================
    # CORE: a_run - ExecutionEngine based with Pause/Continue
    # =========================================================================
    def _get_execution_engine(
        self,
        human_online: bool = False,
    ) -> "ExecutionEngine":
        """
        Get or create ExecutionEngine instance.

        Caches the engine for reuse within the same agent instance.
        Creates a new engine only if parameters change significantly.

        Args:
            human_online: Allow human-in-the-loop

        Returns:
            ExecutionEngine instance
        """
        from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine

        # Check if we can reuse cached engine
        if self._execution_engine_cache is not None:
            # Update dynamic parameters
            self._execution_engine_cache.human_online = human_online
            return self._execution_engine_cache

        # Create new engine
        engine = ExecutionEngine(
            agent=self,
            human_online=human_online,
            is_sub_agent=False,
        )

        self._execution_engine_cache = engine
        return engine

    # =========================================================================
    # MAIN ENTRY POINT (your existing a_run should work with this)
    # =========================================================================

    async def a_run(
        self,
        query: str,
        session_id: str = "default",
        execution_id: str | None = None,
        human_online: bool = False,
        max_iterations: int = os.getenv("DEFAULT_MAX_ITERATIONS",30),
        get_ctx: bool = False,
        **kwargs,
    ) -> str | tuple[str, Any]:
        """
        Main entry point for agent execution.

        Args:
            query: User query
            session_id: Session identifier
            execution_id: For continuing paused execution
            human_online: Allow human-in-the-loop
            max_iterations: Max ReAct iterations (default 25)
            get_ctx: Return (result, ExecutionContext) tuple
            **kwargs: Additional options

        Returns:
            Response string, or (response, ctx) if get_ctx=True
        """
        # Handle defaults
        if not session_id:
            session_id = "default"
        if session_id == "default" and getattr(self, "active_session", None):
            session_id = self.active_session

        self.active_session = session_id
        self.is_running = True

        session = await self.session_manager.get_or_create(session_id)
        self.init_session_tools(session)

        # Check for resume
        ctx = None
        if execution_id:
            engine = self._get_execution_engine(
                human_online=human_online,
            )
            ctx = engine.get_execution(execution_id)

        try:
            # Get or create execution engine
            engine = self._get_execution_engine(
                human_online=human_online,
            )

            # Execute
            result = await engine.execute(
                query=query,
                session_id=session_id,
                max_iterations=max_iterations,
                ctx=ctx,
                get_ctx=get_ctx,
            )

            return result

        except Exception as e:
            import logging
            import traceback

            logging.error(f"a_run failed: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"
        finally:
            self.is_running = False

    # =========================================================================
    # PAUSE / CANCEL / LIST (these should work as-is)
    # =========================================================================

    async def pause_execution(self, execution_id: str) -> dict | None:
        """
        Pause a running execution.

        Args:
            execution_id: The run_id to pause

        Returns:
            Checkpoint dict or None
        """
        engine = self._get_execution_engine()
        ctx = await engine.pause(execution_id)
        return ctx.to_checkpoint() if ctx else None

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an execution.

        Args:
            execution_id: The run_id to cancel

        Returns:
            True if cancelled
        """
        engine = self._get_execution_engine()
        return await engine.cancel(execution_id)

    def list_executions(self) -> list[dict]:
        """
        List all active/paused executions.

        Returns:
            List of execution summaries
        """
        engine = self._get_execution_engine()
        return engine.list_executions()

    async def resume_execution(self, execution_id: str, max_iterations: int = os.getenv("DEFAULT_MAX_ITERATIONS", 30),
                               content="", stream=False) -> str:
        """
        Resume a paused execution.

        Args:
            execution_id: The run_id to resume
            max_iterations: Max additional iterations

        Returns:
            Final response
        """
        engine = self._get_execution_engine()
        return await engine.resume(execution_id, max_iterations, content=content, stream=stream)

    def get_execution_state(self, execution_id: str) -> dict | None:
        """
        Get current state of an execution.

        Args:
            execution_id: The run_id

        Returns:
            Checkpoint dict or None
        """
        engine = self._get_execution_engine()
        ctx = engine.get_execution(execution_id)
        return ctx.to_checkpoint() if ctx else None

    # =========================================================================
    # CORE: a_stream - Voice-First Intelligent Streaming
    # =========================================================================

    async def a_stream(
        self,
        query: str,
        session_id: str = "default",
        execution_id: str | None = None,
        human_online: bool = False,
        max_iterations: int = os.getenv("DEFAULT_MAX_ITERATIONS", 30),
        user_lightning_model: bool | None = None,
        **kwargs,
    ) -> AsyncGenerator[dict, None]:
        """
        Streaming execution - yields chunks during execution.

        Yields:
            dict with 'type' key:
            - {"type": "content", "chunk": "..."}
            - {"type": "tool_start", "name": "..."}
            - {"type": "tool_result", "name": "...", "result": "..."}
            - {"type": "final_answer", "answer": "..."}
            - {"type": "paused", "run_id": "..."}
            - {"type": "done", "success": bool, "final_answer": "..."}
        """
        if not session_id:
            session_id = "default"
        if session_id == "default" and self.active_session is not None:
            session_id = self.active_session

        self.active_session = session_id
        self.is_running = True

        session = await self.session_manager.get_or_create(session_id)
        self.init_session_tools(session)

        # Check for resume
        ctx = None
        if execution_id:
            engine = self._get_execution_engine(
                human_online=human_online,
            )
            ctx = engine.get_execution(execution_id)

        try:
            engine = self._get_execution_engine(
                human_online=human_online,
            )

            # Get stream generator
            stream_func, ctx = await engine.execute_stream(
                query=query,
                session_id=session_id,
                max_iterations=max_iterations,
                ctx=ctx,
                model=os.getenv("BLITZMODEL") if user_lightning_model else None,
            )

            # Yield all chunks
            async for chunk in stream_func(ctx):
                yield chunk

        except Exception as e:
            import traceback

            traceback.print_exc()
            yield {"type": "error", "error": str(e)}
        finally:
            self.is_running = False

    async def a_stream_verbose(
        self,
        query: str,
        session_id: str = "default",
        execution_id: str | None = None,
        human_online: bool = False,
        max_iterations: int = 15,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Vereinfachtes Streaming - gibt nur Strings zurück.

        Yields:
            str: Lesbarer Text für jeden Schritt
        """
        async for chunk in self.a_stream(
            query=query,
            session_id=session_id,
            execution_id=execution_id,
            human_online=human_online,
            max_iterations=max_iterations,
            **kwargs,
        ):
            chunk_type = chunk.get("type", "")

            if chunk_type == "content":
                yield chunk["chunk"]

            elif chunk_type == "reasoning":
                yield Style.GREEN(chunk["chunk"])

            elif chunk_type == "tool_start":
                yield f"\n🔧 Nutze Tool: {chunk['name']}\n"

            elif chunk_type == "tool_result":
                result = chunk.get("result", "")
                if len(result) > 200:
                    result = result[:200] + "..."
                yield f"   ✓ Ergebnis: {result}\n"

            elif chunk_type == "final_answer":
                yield f"\n\n📝 {chunk['answer']}"

            elif chunk_type == "paused":
                yield f"\n⏸️ Pausiert (ID: {chunk['run_id']})\n"

            elif chunk_type == "max_iterations":
                yield f"\n⚠️ Max Iterationen erreicht\n{chunk['answer']}"

            elif chunk_type == "error":
                yield f"\n❌ Fehler: {chunk['error']}\n"

            elif chunk_type == "done":
                status = "✅" if chunk.get("success") else "⚠️"
                yield f"\n{status} Fertig\n"




    # =========================================================================
    # Dreaming
    # =========================================================================

        # =========================================================================
        # Dreaming V3 — Agent-based Meta-Learning
        # =========================================================================

        async def a_dream(self, config=None) -> dict:
            """
            Run a meta-learning dream cycle (blocking).

            Collects the stream internally and returns the final report dict.

            Usage:
                report = await agent.a_dream(DreamConfig(max_budget=5000))
                report = await agent.a_dream()  # defaults
            """
            final_answer = ""
            async for chunk in self.a_dream_stream(config):
                if chunk.get("type") == "final_answer":
                    final_answer = chunk.get("answer", "")
            return {"report": final_answer}

        async def a_dream_stream(self, config=None):
            """
            Streaming dream cycle — identical interface to a_stream().

            Yields the same chunk format (type, agent, iter, tool_start, etc.)
            so icli, TaskView, and any other consumer works unchanged.

            Usage:
                async for chunk in agent.a_dream_stream(DreamConfig()):
                    handle(chunk)
            """
            from toolboxv2.mods.isaa.base.dreamer.types import DreamConfig as DreamConfigV3

            if config is None:
                config = DreamConfigV3()
            elif not isinstance(config, DreamConfigV3):
                # Support old DreamConfig by converting
                config = DreamConfigV3(**{k: v for k, v in config.__dict__.items()
                                          if k in DreamConfigV3.__dataclass_fields__})

            dreamer_agent = await self._get_or_create_dreamer_agent()

            # Pre-harvest: parse logs + snapshot current state (no LLM needed)
            from toolboxv2.mods.isaa.base.dreamer.harvest import (
                harvest_from_vfs, get_cutoff, filter_records,
            )
            from toolboxv2.mods.isaa.base.dreamer.agent import (
                build_dream_query, prepare_dreamer_vfs,
            )
            from toolboxv2.mods.isaa.base.dreamer.prompts import (
                build_dreamer_system_prompt,
            )

            # Get parent session for log access
            parent_session = await self.session_manager.get_or_create(
                self.active_session or "default"
            )
            vfs = parent_session.vfs

            # Harvest logs
            cutoff = get_cutoff(
                max_history_time=config.max_history_time,
                last_run_ts=None,  # TODO: read from VFS /global/.memory/dreamer/last_run
            )
            records = harvest_from_vfs(vfs, "/global/.memory/logs", cutoff)

            # Snapshots
            sm = self.session_manager.skills_manager if hasattr(self.session_manager, 'skills_manager') else None
            rule_set = getattr(parent_session, 'rule_set', None)

            harvest_data = {
                "records": records,
                "skill_snapshot": sm.to_checkpoint() if sm else {},
                "rule_snapshot": rule_set.to_checkpoint() if rule_set else {},
                "persona_snapshot": {},  # loaded from VFS by dreamer tools
            }

            # Setup dreamer session + VFS
            import time as _time
            dreamer_session_id = f"dreamer_{self.amd.name}_{int(_time.time())}"
            dreamer_session = await dreamer_agent.session_manager.get_or_create(dreamer_session_id)
            dreamer_agent.init_session_tools(dreamer_session)
            prepare_dreamer_vfs(dreamer_session.vfs, harvest_data)

            # Build system prompt with context
            skill_count = len(sm.skills) if sm else 0
            active_count = sum(1 for s in sm.skills.values() if s.is_active()) if sm else 0
            rule_count = len(rule_set.situation_rules) if rule_set else 0

            system_prompt = build_dreamer_system_prompt(
                parent_agent_name=self.amd.name,
                budget=config.max_budget,
                harvest_window=f"last {config.max_history_time or 72}h",
                record_count=len(records),
                skill_count=skill_count,
                active_count=active_count,
                rule_count=rule_count,
                persona_count=0,
            )

            # Override dreamer agent system message for this run
            dreamer_agent.amd.system_message = system_prompt

            # Build the dream query
            query = build_dream_query(
                config=config,
                record_count=len(records),
                skill_count=skill_count,
                rule_count=rule_count,
            )

            # Stream the dream run like any other agent run
            async for chunk in dreamer_agent.a_stream(
                query=query,
                session_id=dreamer_session_id,
                max_iterations=50,
            ):
                # Enrich chunks with dream metadata
                chunk["_dream"] = True
                chunk["_dream_id"] = dreamer_session_id
                chunk["_parent_agent"] = self.amd.name
                yield chunk

        async def _get_or_create_dreamer_agent(self):
            """Get or create the DreamerAgent (a standalone FlowAgent)."""
            if hasattr(self, '_dreamer_agent') and self._dreamer_agent is not None:
                return self._dreamer_agent

            from toolboxv2.mods.isaa.base.dreamer.agent import create_dreamer_agent_config
            from toolboxv2.mods.isaa.base.dreamer.tools import get_all_dream_tool_definitions
            from toolboxv2.mods.isaa.base.dreamer.tool_handler import DreamerToolHandler

            # Get ISAA module for builder access
            try:
                from toolboxv2 import get_app
                isaa = get_app().get_mod("isaa")
            except Exception:
                isaa = None

            agent_config = create_dreamer_agent_config(
                parent_name=self.amd.name,
                parent_fast_model=self.amd.fast_llm_model,
            )

            if isaa:
                # Use ISAA builder pattern (gets memory, web search, etc.)
                builder = isaa.get_agent_builder(
                    name=agent_config["name"],
                    add_base_tools=True,
                    with_dangerous_shell=False,
                )
                builder.with_models(
                    agent_config["fast_llm_model"],
                    agent_config["complex_llm_model"],
                )
                builder.with_stream(True)

                # Build the agent
                dreamer_agent = await builder.build()

                # Register in ISAA so it's visible
                await isaa.register_agent(builder)
            else:
                # Standalone fallback (no ISAA module)
                from toolboxv2.mods.isaa.base.Agent.builder import FlowAgentBuilder, AgentConfig
                config = AgentConfig(
                    name=agent_config["name"],
                    fast_llm_model=agent_config["fast_llm_model"],
                    complex_llm_model=agent_config["complex_llm_model"],
                )
                dreamer_agent = await FlowAgentBuilder(config=config).build()

            # Register dream_* tools as callable functions on the agent
            # The tool handler holds references to parent's skills/rules/personas
            # and will be re-initialized on each dream run with fresh harvest data
            self._dreamer_agent = dreamer_agent
            self._register_dream_tools(dreamer_agent)

            return dreamer_agent

        def _register_dream_tools(self, dreamer_agent):
            """Register all dream_* tools on the dreamer agent."""
            from toolboxv2.mods.isaa.base.dreamer.tool_handler import DreamerToolHandler

            sm = self.session_manager.skills_manager if hasattr(self.session_manager, 'skills_manager') else None

            # Create handler with current parent data
            handler = DreamerToolHandler(
                skills=dict(sm.skills) if sm else {},
                rules={},
                patterns=[],
                personas={},
                records=[],
            )

            # Map tool names to handler methods
            tool_map = {
                "dream_get_records": lambda query_filter="", success_only=False,
                                            failure_only=False, limit=50, **kw: handler.handle_get_records(
                    query_filter, success_only, failure_only, limit),
                "dream_get_skills": lambda **kw: handler.handle_get_skills(),
                "dream_get_rules": lambda **kw: handler.handle_get_rules(),
                "dream_get_personas": lambda **kw: handler.handle_get_personas(),
                "dream_cluster_records": lambda record_ids=None, threshold=0.65, **kw:
                handler.handle_cluster_records(record_ids, threshold),
                "dream_evolve_skill": lambda **kw: handler.handle_evolve_skill(**kw),
                "dream_create_skill": lambda **kw: handler.handle_create_skill(**kw),
                "dream_merge_skills": lambda primary_skill_id="", secondary_skill_id="",
                                             merged_instruction="", **kw: handler.handle_merge_skills(
                    primary_skill_id, secondary_skill_id, merged_instruction),
                "dream_split_skill": lambda skill_id="", sub_intents=None, **kw:
                handler.handle_split_skill(skill_id, sub_intents or []),
                "dream_compress_skill": lambda skill_id="", **kw:
                handler.handle_compress_skill(skill_id),
                "dream_extract_rules": lambda rules=None, **kw:
                handler.handle_extract_rules(rules or []),
                "dream_learn_pattern": lambda pattern="", source_situation="",
                                              category="general", tags=None, **kw: handler.handle_learn_pattern(
                    pattern, source_situation, category, tags),
                "dream_evolve_persona": lambda **kw: handler.handle_evolve_persona(**kw),
                "dream_prune_personas": lambda **kw: handler.handle_prune_personas(),
                "dream_cleanup_skills": lambda **kw: handler.handle_cleanup_skills(),
                "dream_cleanup_rules": lambda **kw: handler.handle_cleanup_rules(),
                "dream_delete_skill": lambda skill_id="", reason="", **kw:
                handler.handle_delete_skill(skill_id, reason),
                "dream_delete_rule": lambda rule_id="", reason="", **kw:
                handler.handle_delete_rule(rule_id, reason),
                "dream_extract_memories": lambda memories=None, **kw:
                handler.handle_extract_memories(memories or []),
                "dream_persist_checkpoint": lambda **kw:
                handler.handle_persist_checkpoint(None),  # VFS injected at call time
            }

            # Get tool definitions for descriptions
            from toolboxv2.mods.isaa.base.dreamer.tools import get_all_dream_tool_definitions
            tool_defs = {t["function"]["name"]: t for t in get_all_dream_tool_definitions()}

            for tool_name, tool_func in tool_map.items():
                desc = tool_defs.get(tool_name, {}).get("function", {}).get("description", "")
                dreamer_agent.add_tool(
                    tool_func=tool_func,
                    name=tool_name,
                    description=desc,
                    category=["dream"],
                )

    # =========================================================================
    # audio processing
    # =========================================================================
    def setup_audio(
        self,
        tts_config: TTSConfig | None = None,
        player: str = "null",  # "local" | "web" | "null"
        web_queue_size: int = 50,
        enable_enhancement: bool = False,
    ) -> AudioStreamPlayer:
        """
        Aktiviert Audio-Output für diesen Agent.

        Nach dem Call:
          - speak() Tool ist registriert
          - System-Prompt hat den Audio-Contract
          - self._audio_player ist gesetzt

        Args:
            tts_config:  TTSConfig, default = GROQ_TTS "autumn"
            player:      "local"  → sounddevice Hardware
                         "web"    → WebPlayer relay für WS-Streaming
                         "null"   → NullPlayer (testing / headless)
            web_queue_size: Max queue depth für WebPlayer

        Returns:
            AudioStreamPlayer — `await player.start()` muss danach gecallt werden.

        Example (lokal):
            player = agent.setup_audio(player="local")
            await player.start()

        Example (WebSocket relay):
            web = WebPlayer(max_queue=50)
            player = agent.setup_audio(player="web")
            # player.player ist der WebPlayer
            await player.start()
            async for chunk, meta in web.iter_chunks():
                await ws.send_bytes(chunk)
        """
        backend_map = {
            "local": LocalPlayer(),
            "web": WebPlayer(max_queue=web_queue_size),
            "null": NullPlayer(),
        }
        player_backend = backend_map.get(player, NullPlayer())

        self._audio_player = setup_isaa_audio(
            agent=self,
            tts_config=tts_config,
            player_backend=player_backend,
            enable_enhancement=enable_enhancement,
            session_id=self.active_session or "default",
        )
        return self._audio_player

    @property
    def audio_player(self) -> AudioStreamPlayer | None:
        return getattr(self, "_audio_player", None)

    def set_audio_player_device(self, device=-1):
        if device == -1:
            import sounddevice as sd
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_output_channels"] > 0:
                    print(f"[{i}] {dev['name']}  ({dev['default_samplerate']}Hz)")
        else:
            player = self.audio_player
            if player is None:
                if AGENT_VERBOSE:
                    print("first run setup_audio")
                    return
            if hasattr(player.player, "device"):
                player.player.device = device
            else:
                print(f"not local player player is {player.player.__class__.__name__}")



    async def a_stream_audio(
        self,
        audio_chunks: Generator[bytes, None, None],
        session_id: str = "default",
        language: str = "en",
        **kwargs,
    ) -> AsyncGenerator[bytes, None]:
        """
        Process a stream of audio chunks through the agent.

        Use this for real-time audio processing where you want
        to yield audio output as soon as possible.

        Args:
            audio_chunks: Generator yielding audio byte chunks
            session_id: Session identifier
            language: Response language ("en", "de")
            **kwargs: Additional options

        Yields:
            Audio bytes chunks for immediate playback
        """
        from toolboxv2.mods.isaa.base.audio_io.audioIo import process_audio_stream

        # futool wrp wit arg log user_lightning_model=True for self.a_stream_verbose
        self.active_session = session_id
        async for chunk in process_audio_stream(
            audio_chunks, self.a_stream_verbose, language=language, **kwargs
        ):
            yield chunk

    async def a_audio(
        self,
        audio: Union[bytes, Path, str],
        session_id: str = "default",
        language: str = "en",
        **kwargs,
    ) -> tuple[bytes | None, str, list, dict]:
        """
        Process a complete audio file/buffer through the agent.

        This function handles the full pipeline:
        1. Audio input (file, bytes, or path)
        2. Understanding (STT or native audio model)
        3. Processing (your agent logic via processor callback)
        4. Response generation (TTS or native audio model)

        Args:
            audio: Audio input (bytes, file path, or Path object)
            session_id: Session identifier
            language: Response language ("en", "de")
            **kwargs: Additional options

        Returns:
            Audio bytes for playback
        """
        from toolboxv2.mods.isaa.base.audio_io.audioIo import process_audio_raw

        self.active_session = session_id
        result = await process_audio_raw(
            audio, self.a_stream_verbose, language=language, **kwargs
        )
        # text_input = result.text_input
        text_output = result.text_output
        audio_output = result.audio_output
        tool_calls = result.tool_calls
        metadata = result.metadata

        return audio_output, text_output, tool_calls, metadata

    @staticmethod
    async def tts(text: str, language: str = "en", **kwargs) -> "TTSResult":
        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSResult, synthesize

        return synthesize(text, language=language, **kwargs)

    @staticmethod
    async def stt(
        audio: Union[bytes, Path, str], language: str = "en", **kwargs
    ) -> "STTResult":
        from toolboxv2.mods.isaa.base.audio_io.Stt import STTResult, transcribe

        return transcribe(audio, language=language, **kwargs)

    # =========================================================================
    # TOOL MANAGEMENT
    # =========================================================================

    def add_tool(
        self,
        tool_func: Callable,
        name: str | None = None,
        description: str | None = None,
        category: list[str] | str | None = None,
        flags: dict[str, bool] | None = None,
        live_test_inputs=None,
        cleanup_func=None,
        result_contract=None,
        **kwargs,
    ):
        """Register a tool."""
        # Audit-Log: Tool Added to Agent
        try:
            tool_name = name or (tool_func.__name__ if hasattr(tool_func, '__name__') else "unknown")
            get_app("audit-isaa").audit_logger.log_action(
                user_id="system",
                action="agent.tool_added",
                resource=f"/agent/agents/{self.amd.name}/tools",
                status="SUCCESS",
                details={
                    "agent_name": self.amd.name,
                    "tool_name": tool_name,
                    "category": category if isinstance(category, list) else [category] if category else [],
                    "flags": flags or {},
                }
            )
        except Exception:
            pass

        self.tool_manager.register(
            func=tool_func,
            name=name,
            description=description,
            category=category,
            flags=flags,
            live_test_inputs=live_test_inputs,
            cleanup_func=cleanup_func,
            result_contract=result_contract,
        )

    def remove_tool(self, name: str):
        """Remove a tool."""
        self.tool_manager.un_register(name)

    def add_tools(self, tools: list[dict]):
        """Register multiple tools."""
        for tool in tools:
            self.add_tool(**tool)

    def remove_tools(self, names: list[str]):
        """Remove multiple tools."""
        if names is None:
            return
        for name in names:
            if isinstance(name, dict):
                name = name.get("name")
            if name:
                self.remove_tool(name)

    def get_tool(self, name: str) -> Callable | None:
        """Get tool function by name."""
        return self.tool_manager.get_function(name)

    # =========================================================================
    # SESSION TOOLS INITIALIZATION
    # =========================================================================

    def clear_session_history(self, session_id: str = None):
        session_id = session_id or self.active_session
        _session = self.session_manager.get(session_id)
        if _session:
            _session.clear_history()

    def init_session_tools(self, session: "AgentSession"):
        """
        Register VFS tools for the agent session.

        Primary tools (cover ~18 former individual tools):
          vfs_shell  — all filesystem operations
          vfs_view   — context window / scroll control

        Specialist tools (kept separate — complex params or async/flagged):
          mount/unmount/refresh/sync  |  share  |  docker
          fs_copy (real filesystem)   |  vfs_diagnostics (async LSP)
          history  |  situation/rules  |  local code execution
        """
        if session.tools_initialized:
            return

        from toolboxv2.mods.isaa.base.patch.vfs_shell_tool import (
            make_vfs_shell,
            make_vfs_view,
        )

        # ── Build the two primary tools ────────────────────────────────────
        vfs_shell_fn = make_vfs_shell(session)
        vfs_view_fn = make_vfs_view(session)

        from functools import partial
        search_vfs_fn = partial(search_vfs, vfs=session.vfs)

        # ── Filesystem copy helpers (real FS ↔ VFS) ───────────────────────
        def fs_copy_to_vfs(
            local_path: str,
            vfs_path: str | None = None,
            allowed_dirs: list[str] | None = None,
            max_size_bytes: int = 1024 * 1024,
        ) -> dict:
            """Copy a file from the real filesystem into VFS.
            Security: only reads from allowed_dirs if specified."""
            return session.vfs.load_from_local(
                local_path=local_path,
                vfs_path=vfs_path,
                allowed_dirs=allowed_dirs,
                max_size_bytes=max_size_bytes,
            )

        def fs_copy_from_vfs(
            vfs_path: str,
            local_path: str,
            allowed_dirs: list[str] | None = None,
            overwrite: bool = False,
            create_dirs: bool = True,
        ) -> dict:
            """Copy a file from VFS to the real filesystem.
            Security: only writes to allowed_dirs if specified."""
            return session.vfs.save_to_local(
                vfs_path=vfs_path,
                local_path=local_path,
                allowed_dirs=allowed_dirs,
                overwrite=overwrite,
                create_dirs=create_dirs,
            )

        def fs_copy_dir_from_vfs(
            vfs_path: str,
            local_path: str,
            overwrite: bool = False,
            allowed_dirs: list[str] | None = None,
        ) -> str:
            """Recursively export a VFS directory to the real filesystem."""
            if not session.vfs._is_directory(vfs_path):
                return f"VFS path is not a directory: `{vfs_path}`"
            vfs_path = session.vfs._normalize_path(vfs_path)
            prefix = vfs_path.rstrip("/") + "/"
            files_to_copy = [
                (fp, fp[len(prefix):])
                for fp in session.vfs.files
                if fp.startswith(prefix)
            ]
            if not files_to_copy:
                return f"No files found in `{vfs_path}`"
            ok_count, errors = 0, []
            os.makedirs(local_path, exist_ok=True)
            for vfs_fp, rel_path in files_to_copy:
                target = os.path.join(local_path, rel_path.replace("/", os.sep))
                r = session.vfs.save_to_local(
                    vfs_path=vfs_fp, local_path=target,
                    allowed_dirs=allowed_dirs, overwrite=overwrite, create_dirs=True,
                )
                if r["success"]:
                    ok_count += 1
                else:
                    errors.append(f"{rel_path}: {r['error']}")
            if errors:
                return (
                    f"⚠️ Partial: {ok_count} files copied\\n"
                    + "\\n".join(f"  - {e}" for e in errors[:5])
                )
            return f"Exported {ok_count} files from `{vfs_path}` → `{local_path}`"

        # ── Mount / unmount / sync ─────────────────────────────────────────
        def vfs_mount(
            local_path: str,
            vfs_path: str = "/project",
            allowed_extensions: list[str] | None = None,
            exclude_patterns: list[str] | None = None,
            readonly: bool = False,
            auto_sync: bool = True,
        ) -> dict:
            """Mount a local folder as a shadow into VFS (lazy loading — no content until opened)."""
            return session.vfs.mount(
                local_path=local_path, vfs_path=vfs_path,
                allowed_extensions=allowed_extensions, exclude_patterns=exclude_patterns,
                readonly=readonly, auto_sync=auto_sync,
            )

        def vfs_unmount(vfs_path: str, save_changes: bool = True) -> dict:
            """Unmount a shadow mount, optionally saving all dirty files to disk."""
            return session.vfs.unmount(vfs_path, save_changes=save_changes)

        def vfs_refresh_mount(vfs_path: str) -> dict:
            """Re-scan a mount to detect new / deleted files on disk."""
            return session.vfs.refresh_mount(vfs_path)

        def vfs_sync_all() -> dict:
            """Sync all modified VFS files back to disk."""
            return session.vfs.sync_all()

        # ── Sharing ───────────────────────────────────────────────────────
        def vfs_share_create(
            vfs_path: str,
            readonly: bool = False,
            expires_hours: float | None = None,
        ) -> dict:
            """Create a shareable ID for a VFS directory."""
            from toolboxv2.mods.isaa.base.patch.power_vfs import get_sharing_manager
            return get_sharing_manager().create_share(session.vfs, vfs_path, readonly, expires_hours)

        def vfs_share_list() -> list:
            """List all VFS directories currently shared with this agent."""
            from toolboxv2.mods.isaa.base.patch.power_vfs import get_sharing_manager
            shares = get_sharing_manager().list_shares_for_agent(session.agent_name)
            return [{"id": s.share_id, "path": s.source_path, "owner": s.source_agent} for s in shares]

        def vfs_share_mount(share_id: str, mount_point: str | None = None) -> dict:
            """Mount a shared directory from another agent/session into your VFS."""
            from toolboxv2.mods.isaa.base.patch.power_vfs import get_sharing_manager
            return get_sharing_manager().mount_share(session.vfs, share_id, mount_point)

        # ── LSP Diagnostics (async) ────────────────────────────────────────
        async def vfs_diagnostics(path: str) -> dict:
            """Get LSP diagnostics (errors / warnings / hints) for a code file."""
            return await session.vfs_diagnostics(path)

        # ── Docker ────────────────────────────────────────────────────────
        async def docker_run(
            command: str,
            timeout: int = 300,
            sync_before: bool = True,
            sync_after: bool = True,
        ) -> dict:
            """Execute a shell command inside the Docker container.
            VFS files are synced to /workspace before (sync_before) and
            container changes are pulled back after (sync_after)."""
            return await session.docker_run_command(command, timeout, sync_before, sync_after)

        async def docker_start_app(
            entrypoint: str, port: int = 8080, env: dict[str, str] | None = None
        ) -> dict:
            """Start a web application in the Docker container."""
            return await session.docker_start_web_app(entrypoint, port, env)

        async def docker_stop_app() -> dict:
            """Stop the running web application in Docker."""
            return await session.docker_stop_web_app()

        async def docker_logs(lines: int = 100) -> dict:
            """Get the last N log lines from the running web application."""
            return await session.docker_get_logs(lines)

        def docker_status() -> dict:
            """Get Docker container status (running, ports, etc.)."""
            return session.docker_status()

        # ── History / Situation ────────────────────────────────────────────
        def history(last_n: int = 10) -> list[dict]:
            """Return the last N messages from the conversation history."""
            return session.get_history_for_llm(last_n)

        def set_agent_situation(situation: str, intent: str) -> dict:
            """Set the current situation and intent for rule-based behaviour."""
            session.set_situation(situation, intent)
            return {"success": True, "situation": situation, "intent": intent}

        def check_permissions(action: str, context: dict | None = None) -> dict:
            """Check if an action is permitted under the active rule set."""
            result = session.rule_on_action(action, context)
            return {"allowed": result.allowed, "reason": asdict(result), "rule": result.rule_name}


        # ── Utility tools test ────────────────────────────────────────────

        async def agent_tool_test(tool_name: str, custom_inputs: dict | None = None) -> dict:
            """
            Test any registered tool and report its health status.

            Use this when a tool is not working as expected. Returns a detailed
            diagnostic report including execution result, contract violations, and
            a concrete suggestion for what to do next.

            Args:
                tool_name:     Name of the tool to test (exactly as registered)
                custom_inputs: Optional dict of kwargs to use instead of stored live_test_inputs

            Returns:
                {
                    "tool_name":          str,
                    "status":             "HEALTHY|DEGRADED|FAILED|SKIPPED|GUARANTEED|NOT_FOUND",
                    "execution_time_ms":  float,
                    "result_preview":     str | None,   # max 300 chars
                    "contract_violations": list[str],
                    "error":              str | None,
                    "suggestion":         str
                }
            """
            tm = self.tool_manager  # session aus closure

            entry = tm.get(tool_name)
            if entry is None:
                available = tm.list_names()
                close = [n for n in available if tool_name.lower() in n.lower()][:5]
                return {
                    "tool_name": tool_name,
                    "status": "NOT_FOUND",
                    "execution_time_ms": 0.0,
                    "result_preview": None,
                    "contract_violations": [],
                    "error": f"Tool '{tool_name}' not registered",
                    "suggestion": (
                        f"Similar tools: {close}" if close
                        else f"Use list_tools() to see all {len(available)} registered tools"
                    ),
                }

            # Bestimme test inputs: custom_inputs > live_test_inputs > leer
            effective_inputs: dict | None = custom_inputs
            if effective_inputs is None and entry.live_test_inputs:
                effective_inputs = entry.live_test_inputs[0]

            # Delegiere an ToolManager.health_check_single (modifiziert ggf. temp. live_test_inputs)
            if effective_inputs is not None and not entry.live_test_inputs:
                # Temporär setzen damit health_check_single es aufgreift
                entry.live_test_inputs = [effective_inputs]
                result = await tm.health_check_single(tool_name)
                entry.live_test_inputs = []
            else:
                if effective_inputs is not None:
                    orig = entry.live_test_inputs[:]
                    entry.live_test_inputs = [effective_inputs]
                    result = await tm.health_check_single(tool_name)
                    entry.live_test_inputs = orig
                else:
                    result = await tm.health_check_single(tool_name)

            # Suggestion je Status
            suggestions = {
                "HEALTHY": "Tool is working correctly.",
                "GUARANTEED": "Tool is marked as manually verified — no live test run.",
                "SKIPPED": (
                    "Add live_test_inputs when registering this tool, or call again with "
                    "custom_inputs={'param': 'value'} to test directly."
                ),
                "DEGRADED": (
                    f"Contract violations detected: {result.contract_violations}. "
                    "Check the tool's return value against its result_contract definition."
                ),
                "FAILED": (
                    f"Tool raised an exception: {result.error}. "
                    "Check the tool implementation or its dependencies."
                ),
            }

            return {
                "tool_name": result.tool_name,
                "status": result.status,
                "execution_time_ms": round(result.execution_time_ms, 2),
                "result_preview": (result.result_preview or "")[:300],
                "contract_violations": result.contract_violations,
                "error": result.error,
                "suggestion": suggestions.get(result.status, "Unknown status."),
            }

        # ══════════════════════════════════════════════════════════════════
        # TOOL REGISTRY
        # ══════════════════════════════════════════════════════════════════
        _TOOL_HEALTH_EXTENSIONS = {

            # ── PRIMARY ───────────────────────────────────────────────────────────────

            "vfs_shell": {
                "live_test_inputs": [{"command": "ls /"}],
                "result_contract": {
                    "allow_none": False,
                    "allow_empty_string": False,
                    "expected_type": "dict",
                    "semantic_check_hint": (
                        "If success=False then stderr must not be empty. "
                        "stdout and stderr must both be strings. returncode must be int."
                    ),
                },
                # ls / hat keine Seiteneffekte → kein cleanup nötig
                "cleanup_func": None,
            },

            "vfs_view": {
                "live_test_inputs": [{"path": "/system_context.md", "line_start": 1, "line_end": 5}],
                "result_contract": {
                    "allow_none": False,
                    "expected_type": "dict",
                    "semantic_check_hint": (
                        "If success=True, 'content' and 'showing' keys must be present. "
                        "If success=False, 'error' must be present."
                    ),
                },
                # Datei nach Test wieder schließen
                "cleanup_func": lambda inputs, result: (
                    session.vfs.files.get(
                        session.vfs._normalize_path(inputs["path"])
                    ).__setattr__("state", "closed")
                    if result and result.get("success")
                    else None
                ),
            },

            "search_vfs": {
                "live_test_inputs": [{"query": "system", "mode": "filename"}],
                "result_contract": {
                    "allow_none": False,
                    "expected_type": "list",
                },
                "cleanup_func": None,
            },

            # ── FILESYSTEM COPY ───────────────────────────────────────────────────────

            "fs_copy_to_vfs": {
                # Testet mit allowed_dirs blockiert → erwarteter Fehler ist valide
                "live_test_inputs": [{"local_path": "/nonexistent_probe", "allowed_dirs": ["/tmp"]}],
                "result_contract": {
                    "allow_none": False,
                    "expected_type": "dict",
                },
                "cleanup_func": None,
            },

            "fs_copy_from_vfs": {
                "live_test_inputs": [
                    {"vfs_path": "/system_context.md", "local_path": "/tmp/_vfs_probe_out.md", "allowed_dirs": ["/tmp"],
                     "overwrite": True}],
                "result_contract": {
                    "allow_none": False,
                    "expected_type": "dict",
                },
                "cleanup_func": lambda inputs, result: (
                    __import__("os").unlink("/tmp/_vfs_probe_out.md")
                    if __import__("os").path.exists("/tmp/_vfs_probe_out.md")
                    else None
                ),
            },

            "fs_copy_dir_from_vfs": {
                # Testet mit nicht-existierendem Verzeichnis — erwartet Fehler-String
                "live_test_inputs": [{"vfs_path": "/_nonexistent_probe_dir", "local_path": "/tmp/_vfs_dir_probe"}],
                "result_contract": {
                    "allow_none": False,
                    "expected_type": "str",
                },
                "cleanup_func": None,
            },

            # ── MOUNT ─────────────────────────────────────────────────────────────────

            "vfs_mount": {
                "live_test_inputs": [{"local_path": "/tmp", "vfs_path": "/_probe_mount", "readonly": True}],
                "result_contract": {
                    "allow_none": False,
                    "expected_type": "dict",
                },
                "cleanup_func": lambda inputs, result: (
                    session.vfs.unmount("/_probe_mount", save_changes=False)
                    if result and result.get("success")
                    else None
                ),
            },

            "vfs_unmount": {
                # Testet mit nicht-gemounteten Pfad — soll graceful scheitern
                "live_test_inputs": [{"vfs_path": "/_not_mounted_probe"}],
                "result_contract": {
                    "allow_none": False,
                    "expected_type": "dict",
                },
                "cleanup_func": None,
            },

            "vfs_refresh_mount": {
                "live_test_inputs": [{"vfs_path": "/_not_mounted_probe"}],
                "result_contract": {
                    "allow_none": False,
                    "expected_type": "dict",
                },
                "cleanup_func": None,
            },

            "vfs_sync_all": {
                "live_test_inputs": [{}],
                "result_contract": {
                    "allow_none": False,
                    "expected_type": "dict",
                },
                "cleanup_func": None,
            },

            # ── SHARING ───────────────────────────────────────────────────────────────
            # Sharing hat inter-session dependencies → guaranteed_healthy

            "vfs_share_create": {"flags": {"guaranteed_healthy": True}},
            "vfs_share_list": {"flags": {"guaranteed_healthy": True}},
            "vfs_share_mount": {"flags": {"guaranteed_healthy": True}},

            # ── LSP / DOCKER ──────────────────────────────────────────────────────────
            # Externe Deps → guaranteed_healthy

            "vfs_diagnostics": {"flags": {"guaranteed_healthy": True}},
            "docker_run": {"flags": {"guaranteed_healthy": True}},
            "docker_start_app": {"flags": {"guaranteed_healthy": True}},
            "docker_stop_app": {"flags": {"guaranteed_healthy": True}},
            "docker_logs": {"flags": {"guaranteed_healthy": True}},
            "docker_status": {"flags": {"guaranteed_healthy": True}},

            # ── HISTORY / RULES ───────────────────────────────────────────────────────

            "history": {
                "live_test_inputs": [{"last_n": 1}],
                "result_contract": {
                    "allow_none": False,
                    "expected_type": "list",
                },
                "cleanup_func": None,
            },

            "set_agent_situation": {
                "live_test_inputs": [{"situation": "_probe", "intent": "_probe_intent"}],
                "result_contract": {
                    "allow_none": False,
                    "expected_type": "dict",
                },
                "cleanup_func": lambda inputs, result: (
                    session.set_situation("", "") if result and result.get("success") else None
                ),
            },

            "check_permissions": {
                "live_test_inputs": [{"action": "_probe_action", "context": {}}],
                "result_contract": {
                    "allow_none": False,
                    "expected_type": "dict",
                },
                "cleanup_func": None,
            },
        }

        tools = [
            # ── PRIMARY (replaces ~18 individual vfs_* tools) ─────────────
            {
                "tool_func": vfs_shell_fn,
                "name": "vfs_shell",
                "category": ["vfs", "shell"],
                "description": (
                    "Unix-like shell for VFS: ls cat head tail wc stat tree "
                    "find grep touch write edit echo mkdir rm mv cp close exec. "
                    "Returns {success, stdout, stderr, returncode}."
                ),
            },
            {
                "tool_func": vfs_view_fn,
                "name": "vfs_view",
                "category": ["vfs", "context"],
                "description": (
                    "Open / scroll a file in the context window. "
                    "Use scroll_to= to jump to a pattern, close_others=True to "
                    "reset context. Files opened here appear in EVERY following prompt."
                ),
            },
            {
                "tool_func": search_vfs_fn,
                "name": "search_vfs",
                "category": ["vfs", "discovery", "context"],
                "description": (
                    "Search the Virtual File System (VFS) for files or code snippets matching a query.\n"
                    "\n"
                    "Use this tool when you need to locate where something is defined or used "
                    "(function names, classes, variables, configuration keys, error messages).\n"
                    "\n"
                    "Typical workflow:\n"
                    "1. Use search_vfs to find relevant files or code locations.\n"
                    "2. Use vfs_view to open the most relevant results in the context window.\n"
                    "\n"
                    "Supports filename search, content search, or both. Can also use regex.\n"
                    "\n"
                    "Best used when:\n"
                    "- The location of code or files is unknown\n"
                    "- You need to find where a function/class is defined\n"
                    "- Searching for references, error messages, or configuration keys\n"
                    "\n"
                    "Important parameters:\n"
                    "- query: search string or regex\n"
                    "- mode: 'filename', 'content', or 'both'\n"
                    "- path: restrict search to a directory\n"
                    "- file_extensions: limit search to specific file types\n"
                    "- max_results: limit result count\n"
                    "\n"
                    "Returns a list of matching files and optional code snippets."
                ),
            },
            # ── FILESYSTEM COPY (real FS ↔ VFS) ──────────────────────────
            {
                "tool_func": fs_copy_to_vfs,
                "name": "fs_copy_to_vfs",
                "category": ["filesystem", "vfs"],
                "flags": {"filesystem_access": True},
            },
            {
                "tool_func": fs_copy_from_vfs,
                "name": "fs_copy_from_vfs",
                "category": ["filesystem", "vfs"],
                "flags": {"filesystem_access": True},
            },
            {
                "tool_func": fs_copy_dir_from_vfs,
                "name": "fs_copy_dir_from_vfs",
                "category": ["filesystem", "vfs"],
                "flags": {"filesystem_access": True},
            },
            # ── MOUNT ─────────────────────────────────────────────────────
            {
                "tool_func": vfs_mount,
                "name": "vfs_mount",
                "category": ["filesystem", "vfs"],
                "flags": {"filesystem_access": True},
                "description": "Mount a local folder as a lazy-loaded shadow into VFS.",
            },
            {
                "tool_func": vfs_unmount,
                "name": "vfs_unmount",
                "category": ["filesystem", "vfs"],
                "flags": {"filesystem_access": True},
            },
            {
                "tool_func": vfs_refresh_mount,
                "name": "vfs_refresh_mount",
                "category": ["filesystem", "vfs"],
                "flags": {"filesystem_access": True},
            },
            {
                "tool_func": vfs_sync_all,
                "name": "vfs_sync_all",
                "category": ["filesystem", "vfs"],
                "flags": {"filesystem_access": True},
            },
            # ── SHARING ───────────────────────────────────────────────────
            {
                "tool_func": vfs_share_create,
                "name": "vfs_share_create",
                "category": ["vfs", "sharing"],
            },
            {
                "tool_func": vfs_share_list,
                "name": "vfs_share_list",
                "category": ["vfs", "sharing"],
            },
            {
                "tool_func": vfs_share_mount,
                "name": "vfs_share_mount",
                "category": ["vfs", "sharing"],
            },
            # ── LSP DIAGNOSTICS ───────────────────────────────────────────
            {
                "tool_func": vfs_diagnostics,
                "name": "vfs_diagnostics",
                "category": ["vfs", "lsp"],
                "is_async": True,
            },
            # ── DOCKER ────────────────────────────────────────────────────
            {
                "tool_func": docker_run,
                "name": "docker_run",
                "category": ["docker", "execute"],
                "flags": {"requires_docker": True},
                "is_async": True,
            },
            {
                "tool_func": docker_start_app,
                "name": "docker_start_app",
                "category": ["docker", "web"],
                "flags": {"requires_docker": True},
                "is_async": True,
            },
            {
                "tool_func": docker_stop_app,
                "name": "docker_stop_app",
                "category": ["docker", "web"],
                "flags": {"requires_docker": True},
                "is_async": True,
            },
            {
                "tool_func": docker_logs,
                "name": "docker_logs",
                "category": ["docker", "read"],
                "flags": {"requires_docker": True},
                "is_async": True,
            },
            {
                "tool_func": docker_status,
                "name": "docker_status",
                "category": ["docker", "read"],
                "flags": {"requires_docker": True},
            },
            # ── HISTORY / RULES ───────────────────────────────────────────
            {
                "tool_func": history,
                "name": "history",
                "category": ["memory", "history"],
            },
            {
                "tool_func": set_agent_situation,
                "name": "set_agent_situation",
                "category": ["situation"],
            },
            {
                "tool_func": check_permissions,
                "name": "check_permissions",
                "category": ["situation", "rules"],
            },
            # ── Utility tools test ────────────
            {
                 "tool_func":  agent_tool_test,
                 "name":       "agent_tool_test",
                 "category":   ["meta", "diagnostics"],
                 "flags":      {"guaranteed_healthy": True},
                 "description": (
                     "Test any registered tool and get a diagnostic report. "
                     "Use when a tool fails or behaves unexpectedly. "
                     "Pass tool_name and optional custom_inputs={'param': 'value'}."
                 ),
             },
        ]

        for i,elm in enumerate(tools):
            name = elm.get("name")
            if name in _TOOL_HEALTH_EXTENSIONS:
                tools[i].update(_TOOL_HEALTH_EXTENSIONS[name])

        # ── Optional: Google Tools ────────────────────────────────────────
        if os.getenv("WITH_GOOGLE_TOOLS", "false") == "true":
            if gmail_toolkit:
                tools.extend(gmail_toolkit.get_tools(session.session_id))
            else:
                print("No gmail_toolkit")
            if calendar_toolkit:
                tools.extend(calendar_toolkit.get_tools(session.session_id))
            else:
                print("No calendar_toolkit")

        # ── Optional: Local Code Execution ────────────────────────────────
        if os.getenv("WITH_CODE_TOOLS", "true") == "true":
            from toolboxv2.mods.isaa.base.Agent.executors import create_local_code_exec_tool
            tools.append(create_local_code_exec_tool(self))

        self.add_tools(tools)
        session.tools_initialized = True
        logger.info(
            f"[{session.session_id}] {len(tools)} tools registered "
            f"(vfs_shell + vfs_view replace former ~18 individual VFS tools)"
        )
        return tools

    # =========================================================================
    # CONTEXT AWARENESS & ANALYTICS
    # =========================================================================

    async def context_overview(
        self,
        session_id: str | None = None,
        execution_id: str | None = None,
        print_visual: bool = True,
        f_print=None,
    ) -> dict:
        """
        Analysiert den *exakten* Token-Verbrauch durch Simulation eines echten Engine-Schritts.
        Schlüsselt System-Prompt, Tools und History präzise auf.
        """
        if not LITELLM_AVAILABLE:
            if f_print: f_print("LiteLLM not available.")
            return {}

        target_session = session_id or self.active_session or "default"
        session = await self.session_manager.get_or_create(target_session)

        # 1. Engine holen (Wichtig für exakten Prompt-Bau inkl. Rules & Sub-Agent Constraints)
        engine = self._get_execution_engine()

        ctx = None
        if execution_id:
            ctx = engine.get_execution(execution_id)
        # Versuch, den aktiven Kontext wiederherzustellen oder einen neuen zu simulieren

        if ctx is None:
            if engine._active_executions:
                for c in engine._active_executions.values():
                    if c.session_id == target_session:
                        ctx = c
                        break

        if not ctx:
            from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionContext
            ctx = ExecutionContext(session_id=target_session)
            # Simuliere Start-Zustand für korrekte Tool-Berechnung
            # (Relevanz-Berechnung triggern, damit Dynamic Tools korrekt simuliert werden)
            try:
                engine._calculate_tool_relevance(ctx, "status check")
                engine._preload_skill_tools(ctx, "status check")
            except Exception:
                pass  # Fallback falls SkillsManager noch nicht bereit

        # 2. Exakte Komponenten generieren (DRY RUN)

        # A. System Prompt (Der echte String, den die Engine bauen würde)
        sys_prompt_content = engine._build_system_prompt(ctx, session)

        # B. History-Komponenten
        perm_history = session.get_history_for_llm(last_n=6)
        work_history = ctx.working_history

        # C. Tools (Das kritische Delta: Exakte API-Definition holen)
        active_tools = engine._get_tool_definitions(ctx)

        # 3. Message Stack rekonstruieren (Engine-Logik)
        final_messages = [{"role": "system", "content": sys_prompt_content}] + perm_history

        # Wenn Working History existiert, ist der System Prompt dort meist Index 0.
        # Wir ersetzen ihn durch den FRISCHEN System Prompt (mit aktuellen VFS-Daten).
        if work_history and len(work_history) > 0 and work_history[0].get('role') == 'system':
            final_messages.extend(work_history[1:])

        # 4. Präzises Token Counting mit Overhead
        model = self.amd.fast_llm_model.split("/")[-1]
        try:
            import litellm
            try:
                devnull = open(os.devnull, "w")
            except Exception as e:
                devnull = io.StringIO()
                if os.getenv("AGENT_VERBOSE", "false") == "true":
                    print(e, "\nWhile trying to get devnull")

            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                model_info = litellm.get_model_info(model)
            context_limit = model_info.get("max_input_tokens") or model_info.get("max_tokens") or 128000
        except:
            context_limit = 128000

        def count(msgs, tools=None):
            # Nutzt die exakte Tokenizer-Logik des Modells inkl. Protokoll-Overhead

            try:
                devnull = open(os.devnull, "w")
            except Exception as e:
                devnull = io.StringIO()
                if os.getenv("AGENT_VERBOSE", "false") == "true":
                    print(e, "\nWhile trying to get devnull")

            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                import litellm
                try:
                    if tools:
                        return litellm.token_counter(model=model, messages=msgs, tools=tools)
                    return litellm.token_counter(model=model, messages=msgs)
                except Exception:
                    # Fallback: Grobe Schätzung
                    return len(str(msgs)) // 3 + len(str(tools or "")) // 3

        # -- Deep Dive Analyse der System-Komponenten --
        # Wir zerlegen den System-Prompt, um zu sehen, was Platz frisst
        vfs_content = session.build_vfs_context()
        base_sys = self.amd.get_system_message()
        skills_content = ""
        if ctx.matched_skills and hasattr(engine, 'skills_manager'):
            skills_content = engine.skills_manager.build_skill_prompt_section(ctx.matched_skills)

        # 5. Metriken berechnen
        # System-Prompt gesamt & Sub-Komponenten
        t_sys_total = count([{"role": "system", "content": sys_prompt_content}])
        t_vfs = count([{"role": "system", "content": vfs_content}]) if vfs_content else 0
        t_base = count([{"role": "system", "content": base_sys}]) if base_sys else 0
        t_skills = count([{"role": "system", "content": skills_content}]) if skills_content else 0

        # Vollständiges Skill-Volumen (alle, nicht nur gematchte)
        t_skills_all = 0
        try:
            if hasattr(engine, 'skills_manager') and engine.skills_manager:
                sm = engine.skills_manager
                all_skills = sm.skills.values()
                if all_skills:
                    all_skills_str = sm.build_skill_prompt_section(all_skills)
                    t_skills_all = count([{"role": "system", "content": all_skills_str}])
        except Exception:
            pass

        # Tools
        t_tools = count([], tools=active_tools)

        # History: Perm und Work getrennt zählen
        work_slice = (work_history[1:]
                      if work_history and work_history[0].get('role') == 'system'
                      else work_history or [])
        t_hist_perm = count(perm_history)
        t_hist_work = count(work_slice)

        # Letzte Nachricht (Next / Last Input)
        t_last = count(perm_history[-1:]) if perm_history else 0

        # Total (Tokenizer-Fusion kann von Summe abweichen)
        t_total = count(final_messages, tools=active_tools)

        metrics = {
            "session_id": target_session,
            "model": model,
            "t_total": t_total,
            "t_last": t_last,
            "limit": context_limit,
            "breakdown": {
                "System Prompt Total": t_sys_total,
                "Active Tools": t_tools,
                "History (Perm)": t_hist_perm,
                "History (Work)": t_hist_work,
                "Last Input": t_last,
            },
            "system_details": {
                "Base System Prompt": t_base,
                "VFS Content": t_vfs,
                "Active Skills": t_skills,
                "All Skills (Volume)": t_skills_all,
            },
            "meta": {
                "tool_count": len(active_tools),
                "msg_count": len(final_messages),
                "w_msg_count": len(work_slice),
                "dynamic_tools_loaded": len(ctx.dynamic_tools) if ctx else 0,
                "active_skill_count": len(ctx.matched_skills) if (ctx and ctx.matched_skills) else 0,
            },
        }

        if print_visual:
            self._print_context_visual(metrics, model, f_print=f_print)

        return metrics

    def _print_context_visual(self, m: dict, model_name: str, f_print=None):
        if f_print is None:
            f_print = print

        C_RESET = "\033[0m"
        C_DIM = "\033[90m"
        C_CYAN = "\033[36m"
        C_GREEN = "\033[32m"
        C_YELLOW = "\033[33m"
        C_RED = "\033[31m"
        C_BOLD = "\033[1m"
        C_WHITE = "\033[97m"
        C_BLUE = "\033[34m"

        total = m["t_total"]
        t_last = m.get("t_last", 0)
        limit = m["limit"]
        usage_pct = (total / limit * 100) if limit else 0
        free = limit - total

        bar_color = C_GREEN if usage_pct < 50 else (C_YELLOW if usage_pct < 80 else C_RED)

        # ── HEADER ──────────────────────────────────────────────────────────────
        f_print(f"\n{C_BOLD}🔍 CONTEXT X-RAY{C_RESET}  "
                f"{C_DIM}Session:{C_RESET} {C_CYAN}{m['session_id']}{C_RESET}")
        f_print(f"{C_DIM}Model: {model_name} | Limit: {limit:,} tokens{C_RESET}")
        f_print(f"{C_DIM}{'─' * 62}{C_RESET}")

        # ── HAUPT-BAR ───────────────────────────────────────────────────────────
        bar_width = 44
        filled = int((total / limit) * bar_width) if limit else 0
        bar = "█" * filled + C_DIM + "░" * (bar_width - filled) + C_RESET
        f_print(f"Load:  {bar_color}{bar}{C_RESET} {C_BOLD}{usage_pct:.1f}%{C_RESET}")
        f_print(f"       {C_WHITE}{total:,}{C_RESET}{C_DIM} / {limit:,} tokens used"
                f"   ·   {free:,} frei ({100 - usage_pct:.1f}%){C_RESET}\n")

        # ── TABELLE ─────────────────────────────────────────────────────────────
        COL_W = (28, 10, 10)  # COMPONENT | TOKENS | % LOAD
        header = (f"{C_BOLD}{'COMPONENT':<{COL_W[0]}} {'TOKENS':>{COL_W[1]}} "
                  f"{'% LOAD':>{COL_W[2]}}{C_RESET}")
        f_print(header)
        f_print(f"{C_DIM}{'─' * (sum(COL_W) + 2)}{C_RESET}")

        def row(label, tokens, is_sub=False, extra="", color=C_WHITE):
            pct = (tokens / total * 100) if total else 0
            prefix = f"{C_DIM}  └ " if is_sub else ""
            end = C_RESET
            f_print(
                f"{prefix}{color}{label:<{COL_W[0]}}{end}"
                f" {C_CYAN}{tokens:>{COL_W[1]},}{C_RESET}"
                f" {C_YELLOW}{pct:>{COL_W[2] - 1}.1f}%{C_RESET}"
                + (f"  {extra}" if extra else "")
            )

        bd = m["breakdown"]
        sys_det = m["system_details"]
        meta = m["meta"]

        # System Prompt + Sub-Details
        row("System Prompt Total", bd["System Prompt Total"], color=C_WHITE)
        row("Base System Prompt", sys_det["Base System Prompt"], is_sub=True, color=C_DIM)
        row("VFS Content", sys_det["VFS Content"], is_sub=True, color=C_DIM)
        if sys_det["Active Skills"] > 0:
            skills_extra = (f"{C_DIM}({meta['active_skill_count']} aktiv  |  "
                            f"Gesamt-Vol: {sys_det['All Skills (Volume)']:,}){C_RESET}")
            row("Active Skills", sys_det["Active Skills"], is_sub=True,
                color=C_DIM, extra=skills_extra)
        elif sys_det["All Skills (Volume)"] > 0:
            f_print(f"  {C_DIM}└ Skills (inaktiv)  Gesamt-Vol: "
                    f"{sys_det['All Skills (Volume)']:,} tokens{C_RESET}")

        # Active Tools
        tools_extra = (f"{C_DIM}({meta['tool_count']} defs, "
                       f"{meta['dynamic_tools_loaded']} dyn){C_RESET}")
        row("Active Tools", bd["Active Tools"], color=C_WHITE, extra=tools_extra)

        # History getrennt
        row("History (Perm)", bd["History (Perm)"], color=C_WHITE)
        row("History (Work)", bd["History (Work)"], color=C_WHITE,
            extra=f"{C_DIM}{meta['msg_count']} msgs im Stack{C_RESET}")

        # Last Input
        row("Last Input", bd["Last Input"], color=C_BLUE)

        f_print(f"{C_DIM}{'─' * (sum(COL_W) + 2)}{C_RESET}")
        row("TOTAL", total, color=C_BOLD + C_WHITE)

        # ── WARNINGS ────────────────────────────────────────────────────────────
        hist_total = bd["History (Perm)"] + bd["History (Work)"]
        if usage_pct > 85:
            f_print(f"\n{C_RED}⚠  KRITISCHE AUSLASTUNG  –  {usage_pct:.1f}%{C_RESET}")
            f_print("   Aktion: 'shift_focus' ausführen um History zu komprimieren.")
        elif sys_det["VFS Content"] > 4000:
            f_print(f"\n{C_YELLOW}⚠  Hohe VFS-Last  ({sys_det['VFS Content']:,} tokens){C_RESET}")
            f_print("   Aktion: Nicht benötigte Dateien mit 'vfs_close' schließen.")
        elif hist_total > 6000:
            f_print(f"\n{C_YELLOW}⚠  Langer Kontext  ({hist_total:,} History-Tokens){C_RESET}")
            f_print("   Hinweis: Working History wächst. Zusammenfassung empfohlen.")

        f_print("")

    # =========================================================================
    # CHECKPOINT
    # =========================================================================

    async def save(self) -> str:
        """Save checkpoint."""
        return await self.checkpoint_manager.save_current()

    async def restore(self, function_registry: dict[str, Callable] | None = None) -> dict:
        """Restore from checkpoint."""
        return await self.checkpoint_manager.auto_restore(function_registry)

    # =========================================================================
    # BINDING
    # =========================================================================

    async def bind(
        self, partner: "FlowAgent", mode: str = "public", session_id: str = "default"
    ):
        """Bind to another agent."""
        return await self.bind_manager.bind(partner, mode, session_id)

    def unbind(self, partner_name: str) -> bool:
        """Unbind from partner."""
        return self.bind_manager.unbind(partner_name)

    # =========================================================================
    # SERVERS
    # =========================================================================

    def setup_mcp_server(self, name: str | None = None):
        try:
            from mcp.server.fastmcp import FastMCP

            MCP_AVAILABLE = True
        except ImportError:
            MCP_AVAILABLE = False

        if not MCP_AVAILABLE:
            logger.warning("MCP not available")
            return

        server_name = name or f"{self.amd.name}_MCP"
        self.mcp_server = FastMCP(server_name)

        @self.mcp_server.tool()
        async def agent_run(query: str, session_id: str = "mcp_session") -> str:
            return await self.a_run(query, session_id=session_id)

    def setup_a2a_server(self, host: str = "0.0.0.0", port: int = 5000):
        try:
            from python_a2a import A2AServer, AgentCard
            A2A_AVAILABLE = True
        except ImportError:
            A2A_AVAILABLE = False

        if not A2A_AVAILABLE:
            logger.warning("A2A not available")
            return

        self.a2a_server = A2AServer(
            host=host,
            port=port,
            agent_card=AgentCard(
                name=self.amd.name, description="FlowAgent", version="2.0"
            ),
        )

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def close(self):
        """Clean shutdown."""
        self.is_running = False
        print("Saving checkpoint...")
        await self.save()
        if self.amd.enable_docker:
            await self.session_manager.cleanup_docker_containers()
        await self.session_manager.close_all()

        if self.a2a_server:
            await self.a2a_server.close()
        if self.mcp_server:
            await self.mcp_server.close()
        logger.info(f"FlowAgent '{self.amd.name}' closed")

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def total_cost(self) -> float:
        return self.total_cost_accumulated

    def get_stats(self) -> dict:
        return {
            "agent_name": self.amd.name,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_cost": self.total_cost_accumulated,
            "total_llm_calls": self.total_llm_calls,
            "sessions": self.session_manager.get_stats(),
            "tools": self.tool_manager.get_stats(),
            "bindings": self.bind_manager.get_stats(),
        }

    def __repr__(self) -> str:
        return f"<FlowAgent '{self.amd.name}' [{len(self.session_manager.sessions)} sessions] [{len(self.tool_manager._registry.keys())} tools] [{len(self.bind_manager.bindings)} bindings]>"

    def __rshift__(self, other):
        return Chain(self) >> other

    def __add__(self, other):
        return Chain(self) + other

    def __and__(self, other):
        return Chain(self) & other

    def __mod__(self, other):
        """Implements % operator for conditional branching"""
        return ConditionalChain(self, other)
