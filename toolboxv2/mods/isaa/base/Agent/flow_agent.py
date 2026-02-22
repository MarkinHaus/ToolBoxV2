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
import copy
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Coroutine, Generator, Union

import yaml
from pydantic import BaseModel, ValidationError

from toolboxv2 import Style, get_logger
from toolboxv2.mods.isaa.base.Agent.chain import Chain, ConditionalChain
from toolboxv2.mods.isaa.base.Agent.types import (
    AgentModelData,
    NodeStatus,
    PersonaConfig,
    ProgressEvent,
)

# Framework imports
try:
    import litellm

    # Unterdrückt die störenden LiteLLM Konsolen-Logs
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    from python_a2a import A2AServer, AgentCard

    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False

    class A2AServer:
        pass

    class AgentCard:
        pass


try:
    from mcp.server.fastmcp import FastMCP

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

    class FastMCP:
        pass


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
AGENT_VERBOSE = os.environ.get("AGENT_VERBOSE", "false").lower() == "true"
litellm.suppress_debug_info = not AGENT_VERBOSE

MAX_CONTINUATIONS = os.environ.get("AGENT_INTERN_MAX_CONTINUATIONS", 100)

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
        self.amd = amd
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

        self.executor = ThreadPoolExecutor(max_workers=max_parallel_tasks)

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
            default_max_history=100,
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
        processed_messages = self._process_media_in_messages(messages.copy())

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
            **kwargs,
        }

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

        try:
            from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message

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

    @staticmethod
    async def _process_streaming_response(response, task_id, model, get_response_message):
        from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message

        result = ""
        tool_calls_acc = {}
        final_chunk = None
        finish_reason = None

        async for chunk in response:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            result += content

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
        max_iterations: int = 25,
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

    async def resume_execution(self, execution_id: str, max_iterations: int = 15) -> str:
        """
        Resume a paused execution.

        Args:
            execution_id: The run_id to resume
            max_iterations: Max additional iterations

        Returns:
            Final response
        """
        engine = self._get_execution_engine()
        return await engine.resume(execution_id, max_iterations)

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
        max_iterations: int = 15,
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
    # audio processing
    # =========================================================================

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
        **kwargs,
    ):
        """Register a tool."""
        self.tool_manager.register(
            func=tool_func,
            name=name,
            description=description,
            category=category,
            flags=flags,
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
        Initialize session-specific tools for VFS V2, Docker, and filesystem operations.

        Tools are categorized:
        - vfs: Virtual File System operations
        - docker: Container execution (flag: requires_docker)
        - filesystem: Real filesystem copy operations (flag: filesystem_access)
        - memory: RAG and history
        - situation: Behavior control
        """

        if session.tools_initialized:
            return

        # =========================================================================
        # VFS TOOLS (V2)
        # =========================================================================

        # --- File Operations ---

        def vfs_list(path: str = "/", recursive: bool = False) -> dict:
            """
            List directory contents in VFS.

            Args:
                path: Directory path to list (default: root)
                recursive: If True, list recursively

            Returns:
                Dict with contents list including files and directories
            """
            try:
                result = session.vfs_ls(path, recursive)
                if isinstance(result, dict) and not result.get("success", True):
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(f"VFS List failed for {path}: {error_msg}")
                return result
            except Exception as e:
                logger.error(f"VFS List exception for {path}: {e}")
                return {"success": False, "error": str(e)}

        def vfs_read(path: str) -> dict:
            """
            Read file content from VFS.

            Args:
                path: Path to file (e.g., "/src/main.py")

            Returns:
                Dict with file content
            """
            try:
                result = session.vfs_read(path)
                if isinstance(result, dict) and not result.get("success", True):
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(f"VFS Read failed for {path}: {error_msg}")
                return result
            except Exception as e:
                logger.error(f"VFS Read exception for {path}: {e}")
                return {"success": False, "error": str(e)}

        def vfs_create(path: str, content: str = "") -> dict:
            """
            Create a new file in VFS.

            Args:
                path: Path for new file (e.g., "/src/utils.py")
                content: Initial file content

            Returns:
                Dict with success status and file type info
            """
            try:
                result = session.vfs_create(path, content)
                if isinstance(result, dict) and not result.get("success", True):
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(f"VFS Create failed for {path}: {error_msg}")
                return result
            except Exception as e:
                logger.error(f"VFS Create exception for {path}: {e}")
                return {"success": False, "error": str(e)}

        def vfs_write(path: str, content: str) -> dict:
            """
            Write/overwrite file content in VFS.

            Args:
                path: Path to file
                content: New content

            Returns:
                Dict with success status
            """
            try:
                result = session.vfs_write(path, content)
                if isinstance(result, dict) and not result.get("success", True):
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(f"VFS Write failed for {path}: {error_msg}")
                return result
            except Exception as e:
                logger.error(f"VFS Write exception for {path}: {e}")
                return {"success": False, "error": str(e)}

        def vfs_edit(path: str, line_start: int, line_end: int, new_content: str) -> dict:
            """
            Edit file by replacing lines (1-indexed).

            Args:
                path: Path to file
                line_start: First line to replace (1-indexed)
                line_end: Last line to replace (inclusive)
                new_content: New content for those lines

            Returns:
                Dict with success status
            """
            try:
                result = session.vfs.edit(path, line_start, line_end, new_content)
                if isinstance(result, dict) and not result.get("success", True):
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(
                        f"VFS Edit failed for {path} (lines {line_start}-{line_end}): {error_msg}"
                    )
                return result
            except Exception as e:
                logger.error(
                    f"VFS Edit exception for {path} (lines {line_start}-{line_end}): {e}"
                )
                return {"success": False, "error": str(e)}

        def vfs_append(path: str, content: str) -> dict:
            """
            Append content to a file.

            Args:
                path: Path to file
                content: Content to append

            Returns:
                Dict with success status
            """
            try:
                result = session.vfs.append(path, content)
                if isinstance(result, dict) and not result.get("success", True):
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(f"VFS Append failed for {path}: {error_msg}")
                return result
            except Exception as e:
                logger.error(f"VFS Append exception for {path}: {e}")
                return {"success": False, "error": str(e)}

        def vfs_delete(path: str) -> dict:
            """
            Delete a file from VFS.

            Args:
                path: Path to file

            Returns:
                Dict with success status
            """
            try:
                result = session.vfs.delete(path)
                if isinstance(result, dict) and not result.get("success", True):
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(f"VFS Delete failed for {path}: {error_msg}")
                return result
            except Exception as e:
                logger.error(f"VFS Delete exception for {path}: {e}")
                return {"success": False, "error": str(e)}

        # --- Directory Operations ---

        def vfs_mkdir(path: str, parents: bool = True) -> dict:
            """
            Create a directory in VFS.

            Args:
                path: Directory path (e.g., "/src/components")
                parents: If True, create parent directories as needed

            Returns:
                Dict with success status
            """
            try:
                result = session.vfs_mkdir(path, parents)
                if isinstance(result, dict) and not result.get("success", True):
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(f"VFS Mkdir failed for {path}: {error_msg}")
                return result
            except Exception as e:
                logger.error(f"VFS Mkdir exception for {path}: {e}")
                return {"success": False, "error": str(e)}

        def vfs_rmdir(path: str, force: bool = False) -> dict:
            """
            Remove a directory from VFS.

            Args:
                path: Directory path
                force: If True, remove non-empty directories recursively

            Returns:
                Dict with success status
            """
            try:
                result = session.vfs_rmdir(path, force)
                if isinstance(result, dict) and not result.get("success", True):
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(f"VFS Rmdir failed for {path}: {error_msg}")
                return result
            except Exception as e:
                logger.error(f"VFS Rmdir exception for {path}: {e}")
                return {"success": False, "error": str(e)}

        def vfs_mv(source: str, destination: str) -> dict:
            """
            Move/rename a file or directory.

            Args:
                source: Source path
                destination: Destination path

            Returns:
                Dict with success status
            """
            try:
                result = session.vfs_mv(source, destination)
                if isinstance(result, dict) and not result.get("success", True):
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(
                        f"VFS Move failed from {source} to {destination}: {error_msg}"
                    )
                return result
            except Exception as e:
                logger.error(f"VFS Move exception from {source} to {destination}: {e}")
                return {"success": False, "error": str(e)}

        # --- Open/Close Operations ---

        def vfs_open(path: str, line_start: int = 1, line_end: int = -1) -> dict:
            """Open a file (make permanent visible in context)."""
            try:
                result = session.vfs_open(path, line_start, line_end)
                if isinstance(result, dict) and not result.get("success", True):
                    logger.warning(f"VFS Open failed for {path}: {result.get('error', 'Unknown error')}")
                return result
            except Exception as e:
                logger.error(f"VFS Open exception for {path}: {e}")
                return {"success": False, "error": str(e)}

        async def vfs_close(path: str) -> dict:
            """Close a file (remove from context, generate summary)."""
            try:
                result = await session.vfs_close(path)
                if isinstance(result, dict) and not result.get("success", True):
                    logger.warning(f"VFS Close failed for {path}: {result.get('error', 'Unknown error')}")
                return result
            except Exception as e:
                logger.error(f"VFS Close exception for {path}: {e}")
                return {"success": False, "error": str(e)}

        def vfs_view(path: str, line_start: int = 1, line_end: int = -1) -> dict:
            """
            View/adjust visible window of an open file.

            Args:
                path: Path to file
                line_start: First line to show
                line_end: Last line to show

            Returns:
                Dict with visible content
            """
            return session.vfs.view(path, line_start, line_end)

        # --- Info & Diagnostics ---

        def vfs_info(path: str) -> dict:
            """
            Get detailed info about a file or directory.

            Args:
                path: Path to file or directory

            Returns:
                Dict with metadata (type, size, lines, file_type, lsp_enabled, etc.)
            """
            return session.vfs.get_file_info(path)

        async def vfs_diagnostics(path: str) -> dict:
            """
            Get LSP diagnostics (errors, warnings, hints) for a code file.

            Args:
                path: Path to code file

            Returns:
                Dict with diagnostics list, error/warning/hint counts
            """
            return await session.vfs_diagnostics(path)

        def vfs_executables() -> list[dict]:
            """
            Get list of all executable files in VFS.

            Returns:
                List of executable files with path, language, size
            """
            return session.vfs.get_executable_files()

        def fs_copy_dir_from_vfs(
            vfs_path: str,
            local_path: str,
            overwrite: bool = False,
            allowed_dirs: list[str] | None = None,
        ) -> str:
            """
            Recursively export a VFS directory to the real filesystem.

            Args:
                vfs_path: Source path in VFS (e.g., "/src")
                local_path: Destination path on disk
                overwrite: If True, overwrite existing files
                allowed_dirs: Security restriction for write operations
            """
            # 1. Validate Source
            if not session.vfs._is_directory(vfs_path):
                return f"VFS path is not a directory: `{vfs_path}`"

            # 2. Collect files to copy
            files_to_copy = []
            vfs_path = session.vfs._normalize_path(vfs_path)
            prefix = vfs_path if vfs_path.endswith("/") else vfs_path + "/"

            for f_path in session.vfs.files:
                if f_path.startswith(prefix):
                    rel_path = f_path[len(prefix) :]
                    files_to_copy.append((f_path, rel_path))

            if not files_to_copy:
                return f"No files found in VFS directory: `{vfs_path}`"

            # 3. Perform Copy
            success_count = 0
            errors = []

            try:
                # Ensure target root exists
                if not os.path.exists(local_path):
                    os.makedirs(local_path, exist_ok=True)

                for vfs_file, rel_path in files_to_copy:
                    # Construct OS-specific path
                    target_file = os.path.join(local_path, rel_path.replace("/", os.sep))

                    # Use internal save_to_local for safety & consistency
                    res = session.vfs.save_to_local(
                        vfs_path=vfs_file,
                        local_path=target_file,
                        allowed_dirs=allowed_dirs,
                        overwrite=overwrite,
                        create_dirs=True,
                    )

                    if res["success"]:
                        success_count += 1
                    else:
                        errors.append(f"{rel_path}: {res['error']}")

            except Exception as e:
                return f"System error during recursive copy: {str(e)}"

            # 4. Format Output
            if errors:
                return (
                    f"⚠️ **Partial Success**\n"
                    f"Copied {success_count} files to `{local_path}`\n"
                    f"**Errors**:\n" + "\n".join([f"- {e}" for e in errors[:5]])
                )

            return f"Recursively exported {success_count} files from `{vfs_path}` to `{local_path}`"

        # =========================================================================
        # FILESYSTEM COPY TOOLS (Flag: filesystem_access)
        # =========================================================================

        def fs_copy_to_vfs(
            local_path: str,
            vfs_path: str | None = None,
            allowed_dirs: list[str] | None = None,
            max_size_bytes: int = 1024 * 1024,
        ) -> dict:
            """
            Copy a file from real filesystem into VFS.

            Args:
                local_path: Path on real filesystem
                vfs_path: Destination path in VFS (default: /<filename>)
                allowed_dirs: List of allowed directories for security
                max_size_bytes: Maximum file size (default: 1MB)

            Returns:
                Dict with success status, vfs_path, size, lines, file_type

            Security:
                Requires filesystem_access flag.
                Only reads from allowed_dirs if specified.
            """
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
            """
            Copy a file from VFS to real filesystem.

            Args:
                vfs_path: Path in VFS
                local_path: Destination path on real filesystem
                allowed_dirs: List of allowed directories for security
                overwrite: Allow overwriting existing files
                create_dirs: Create parent directories if needed

            Returns:
                Dict with success status, saved_path, size, lines

            Security:
                Requires filesystem_access flag.
                Only writes to allowed_dirs if specified.
            """
            return session.vfs.save_to_local(
                vfs_path=vfs_path,
                local_path=local_path,
                allowed_dirs=allowed_dirs,
                overwrite=overwrite,
                create_dirs=create_dirs,
            )

        def vfs_mount(
            local_path: str,
            vfs_path: str = "/project",
            allowed_extensions: list[str] | None = None,
            exclude_patterns: list[str] | None = None,
            readonly: bool = False,
            auto_sync: bool = True,
        ) -> dict:
            """
            Mount a local folder as shadow into VFS.

            Only scans metadata, does NOT load file contents.
            Files are loaded on-demand when opened.

            Args:
                local_path: Local folder path to mount
                vfs_path: Mount point in VFS (default: /project)
                allowed_extensions: Only include these extensions (e.g., [".py", ".js"])
                exclude_patterns: Exclude patterns (default: __pycache__, .git, node_modules, etc.)
                readonly: If True, no write operations allowed
                auto_sync: If True, changes are written to disk immediately

            Returns:
                Dict with success, mount_point, files_indexed, scan_time_ms
            """
            return session.vfs.mount(
                local_path=local_path,
                vfs_path=vfs_path,
                allowed_extensions=allowed_extensions,
                exclude_patterns=exclude_patterns,
                readonly=readonly,
                auto_sync=auto_sync,
            )

        def vfs_unmount(vfs_path: str, save_changes: bool = True) -> dict:
            """
            Unmount a shadow mount and optionally save all changes.

            Args:
                vfs_path: Mount point to unmount
                save_changes: If True, sync all dirty files before unmounting

            Returns:
                Dict with success, unmounted path, files_saved list
            """
            return session.vfs.unmount(vfs_path, save_changes=save_changes)

        def vfs_refresh_mount(vfs_path: str) -> dict:
            """
            Refresh a mount to detect new/deleted files on disk.

            Preserves modified files that haven't been synced yet.

            Args:
                vfs_path: Mount point to refresh

            Returns:
                Dict with success, files_indexed, modified_preserved
            """
            return session.vfs.refresh_mount(vfs_path)

        def vfs_sync_all() -> dict:
            """
            Sync all dirty files to disk.

            Returns:
                Dict with success, synced files list, errors list
            """
            return session.vfs.sync_all()

        def vfs_execute(
            path: str, args: list[str] | None = None, timeout: float = 30.0
        ) -> dict:
            """
            Execute an executable file in VFS.

            Supports Python, JavaScript, TypeScript, Shell scripts.
            Shadow files are executed directly from disk.
            Memory files are written to temp and executed.

            Args:
                path: VFS path to executable file
                args: Command line arguments
                timeout: Execution timeout in seconds (default: 30)

            Returns:
                Dict with success, return_code, stdout, stderr, command
            """
            return session.vfs.execute(path, args=args, timeout=timeout)

        def vfs_grep(
            pattern: str,
            path: str = "/",
            recursive: bool = True,
            case_sensitive: bool = False,
        ) -> str:
            """
            Search for a regex pattern in files.
            Efficiently searches in-memory VFS files and local shadow files.

            Args:
                pattern: Regex pattern to search for
                path: Root path to start search
                recursive: Search subdirectories
                case_sensitive: Respect case (default: False)
            """
            import re

            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return f"Invalid regex: {e}"

            matches = []
            files_scanned = 0

            # Normalize path for filtering
            search_root = path if path.startswith("/") else "/" + path
            search_root = search_root.rstrip("/") + "/"
            if search_root == "//":
                search_root = "/"

            # Get candidate files from VFS registry
            candidates = []
            for file_path, file_obj in session.vfs.files.items():
                if not file_path.startswith(search_root):
                    continue
                if not recursive and "/" in file_path[len(search_root) :].strip("/"):
                    continue
                candidates.append((file_path, file_obj))

            for f_path, f_obj in candidates:
                files_scanned += 1
                content = None

                # Strategy: Get content without fully loading into VFS state if possible (peek)
                # But for VFSFile, we usually just access what's available.

                # 1. In-Memory or Modified Files
                if hasattr(f_obj, "_content") and f_obj._content is not None:
                    content = f_obj._content

                # 2. Shadow Files (on disk)
                elif hasattr(f_obj, "local_path") and f_obj.local_path:
                    try:
                        if os.path.exists(f_obj.local_path):
                            # Read directly from disk to avoid polluting VFS memory with full loads
                            # for a simple grep
                            with open(
                                f_obj.local_path, "r", encoding="utf-8", errors="ignore"
                            ) as f:
                                content = f.read()
                    except:
                        continue  # Skip unreadable

                if content:
                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        if regex.search(line):
                            matches.append(f"{f_path}:{i + 1}: {line.strip()}")
                            if len(matches) > 50:  # Cap results
                                matches.append("... (limit reached)")
                                break
                if len(matches) > 50:
                    break

            if not matches:
                return f"🔍 No matches found for `{pattern}` in {files_scanned} files."

            return (
                f"🔍 **Found matches** in {files_scanned} files:\n```\n"
                + "\n".join(matches)
                + "\n```"
            )

        # --- Sharing Tools ---
        def vfs_share_create(vfs_path: str, readonly: bool = False, expires_hours: float = None) -> dict:
            """Creates a shareable link for a directory"""
            from toolboxv2.mods.isaa.base.patch.power_vfs import get_sharing_manager
            try:
                result = get_sharing_manager().create_share(session.vfs, vfs_path, readonly, expires_hours)
                if not result.get("success"):
                    logger.warning(f"Share create failed for {vfs_path}: {result.get('error')}")
                return result
            except Exception as e:
                logger.error(f"Share create exception: {e}")
                return {"success": False, "error": str(e)}

        def vfs_share_list() -> list:
            """Lists all directories currently shared with this agent"""
            from toolboxv2.mods.isaa.base.patch.power_vfs import get_sharing_manager
            shares = get_sharing_manager().list_shares_for_agent(session.agent_name)
            return [{"id": s.share_id, "path": s.source_path, "owner": s.source_agent} for s in shares]

        def vfs_share_mount(share_id: str, mount_point: str = None) -> dict:
            """Mounts a shared directory into your VFS"""
            from toolboxv2.mods.isaa.base.patch.power_vfs import get_sharing_manager
            try:
                result = get_sharing_manager().mount_share(session.vfs, share_id, mount_point)
                if not result.get("success"):
                    logger.warning(f"Share mount failed for {share_id}: {result.get('error')}")
                return result
            except Exception as e:
                logger.error(f"Share mount exception: {e}")
                return {"success": False, "error": str(e)}
        # =========================================================================
        # DOCKER TOOLS (Flag: requires_docker)
        # =========================================================================

        async def docker_run(
            command: str,
            timeout: int = 300,
            sync_before: bool = True,
            sync_after: bool = True,
        ) -> dict:
            """
            Execute a command in the Docker container.

            The container has VFS files synced to /workspace.
            Changes made in the container are synced back to VFS.

            Args:
                command: Shell command to execute
                timeout: Timeout in seconds (default: 300)
                sync_before: Sync VFS to container before execution
                sync_after: Sync container to VFS after execution

            Returns:
                Dict with stdout, stderr, exit_code, duration, success
            """
            return await session.docker_run_command(
                command, timeout, sync_before, sync_after
            )

        async def docker_start_app(
            entrypoint: str, port: int = 8080, env: dict[str, str] | None = None
        ) -> dict:
            """
            Start a web application in the Docker container.

            Args:
                entrypoint: Command to start the app (e.g., "python app.py")
                port: Port the app listens on (default: 8080)
                env: Environment variables

            Returns:
                Dict with url, host_port, status
            """
            return await session.docker_start_web_app(entrypoint, port, env)

        async def docker_stop_app() -> dict:
            """
            Stop the running web application.

            Returns:
                Dict with success status
            """
            return await session.docker_stop_web_app()

        async def docker_logs(lines: int = 100) -> dict:
            """
            Get logs from the web application.

            Args:
                lines: Number of log lines to retrieve

            Returns:
                Dict with logs content
            """
            return await session.docker_get_logs(lines)

        def docker_status() -> dict:
            """
            Get Docker container status.

            Returns:
                Dict with is_running, container_id, exposed_ports, etc.
            """
            return session.docker_status()

        # =========================================================================
        # history TOOLS
        # =========================================================================

        def history(last_n: int = 10) -> list[dict]:
            """
            Get recent conversation history.

            Args:
                last_n: Number of recent messages

            Returns:
                List of message dicts with role and content
            """
            return session.get_history_for_llm(last_n)

        # =========================================================================
        # SITUATION/BEHAVIOR TOOLS
        # =========================================================================

        def set_agent_situation(situation: str, intent: str) -> dict:
            """
            Set the current situation and intent for rule-based behavior.

            Args:
                situation: Current situation description
                intent: Current intent/goal

            Returns:
                Confirmation dict
            """
            session.set_situation(situation, intent)
            return {"success": True, "situation": situation, "intent": intent}

        def check_permissions(action: str, context: dict | None = None) -> dict:
            """
            Check if an action is allowed under current rules.

            Args:
                action: Action to check
                context: Optional context for rule evaluation

            Returns:
                Dict with allowed status and reason
            """
            result = session.rule_on_action(action, context)
            return {
                "allowed": result.allowed,
                "reason": result.reason,
                "rule": result.rule_name,
            }

        # =========================================================================
        # REGISTER ALL TOOLS
        # =========================================================================
        vfs_tools = [
            {"tool_func": vfs_list, "name": "vfs_list", "category": ["vfs", "read"]},
            {"tool_func": vfs_read, "name": "vfs_read", "category": ["vfs", "read"]},
            {"tool_func": vfs_create, "name": "vfs_create", "category": ["vfs", "write"]},
            {"tool_func": vfs_write, "name": "vfs_write", "category": ["vfs", "write"]},
            {"tool_func": vfs_edit, "name": "vfs_edit", "category": ["vfs", "write"]},
            {"tool_func": vfs_append, "name": "vfs_append", "category": ["vfs", "write"]},
            {"tool_func": vfs_delete, "name": "vfs_delete", "category": ["vfs", "write"]},
            # VFS Directory Operations
            {"tool_func": vfs_mkdir, "name": "vfs_mkdir", "category": ["vfs", "write"]},
            {"tool_func": vfs_rmdir, "name": "vfs_rmdir", "category": ["vfs", "write"]},
            {"tool_func": vfs_mv, "name": "vfs_mv", "category": ["vfs", "write"]},
        ]
        tools = [] if hasattr(self, "_vfs_tools_registered") else vfs_tools
        tools.extend(
            [
                # VFS File Operations
                # VFS Open/Close
                {
                    "tool_func": vfs_open,
                    "name": "vfs_open",
                    "category": ["vfs", "context"],
                },
                {
                    "tool_func": vfs_close,
                    "name": "vfs_close",
                    "category": ["vfs", "context"],
                    "is_async": True,
                },
                {
                    "tool_func": vfs_view,
                    "name": "vfs_view",
                    "category": ["vfs", "context"],
                },
                {
                    "tool_func": vfs_grep,
                    "name": "vfs_grep",
                    "category": ["vfs", "search"],
                    "description": "Full-text search (grep) in files using regex. Supports recursion and case sensitivity.",
                },
                {
                    "tool_func": fs_copy_dir_from_vfs,
                    "name": "fs_copy_dir_from_vfs",
                    "category": ["filesystem", "vfs"],
                    "flags": {"filesystem_access": True},
                    "description": "Recursively export a VFS directory to the real filesystem",
                },
                # VFS Info & Diagnostics
                {"tool_func": vfs_info, "name": "vfs_info", "category": ["vfs", "read"]},
                {
                    "tool_func": vfs_diagnostics,
                    "name": "vfs_diagnostics",
                    "category": ["vfs", "lsp"],
                    "is_async": True,
                },
                {
                    "tool_func": vfs_executables,
                    "name": "vfs_executables",
                    "category": ["vfs", "read"],
                },
                # Filesystem Copy (Flag-based)
                {
                    "tool_func": fs_copy_to_vfs,
                    "name": "fs_copy_to_vfs",
                    "category": ["filesystem", "vfs"],
                    "flags": {"filesystem_access": True},
                    "description": "Copy file from real filesystem to VFS",
                },
                {
                    "tool_func": fs_copy_from_vfs,
                    "name": "fs_copy_from_vfs",
                    "category": ["filesystem", "vfs"],
                    "flags": {"filesystem_access": True},
                    "description": "Copy file from VFS to real filesystem",
                },
                {
                    "tool_func": vfs_mount,
                    "name": "vfs_mount",
                    "category": ["filesystem", "vfs"],
                    "flags": {"filesystem_access": True},
                    "description": "Mount local folder as shadow into VFS (lazy loading, no content copied until file opened)",
                },
                {
                    "tool_func": vfs_unmount,
                    "name": "vfs_unmount",
                    "category": ["filesystem", "vfs"],
                    "flags": {"filesystem_access": True},
                    "description": "Unmount shadow mount and optionally save all changes to disk",
                },
                {
                    "tool_func": vfs_refresh_mount,
                    "name": "vfs_refresh_mount",
                    "category": ["filesystem", "vfs"],
                    "flags": {"filesystem_access": True},
                    "description": "Refresh mount to detect new/deleted files on disk",
                },
                {
                    "tool_func": vfs_sync_all,
                    "name": "vfs_sync_all",
                    "category": ["filesystem", "vfs"],
                    "flags": {"filesystem_access": True},
                    "description": "Sync all modified files to disk",
                },
                {
                    "tool_func": vfs_execute,
                    "name": "vfs_execute",
                    "category": ["filesystem", "vfs", "execution"],
                    "flags": {"filesystem_access": True, "code_execution": True},
                    "description": "Execute an executable file (Python, JS, Shell) in VFS",
                },
                # ============================================================
                # VFS Sharing Tools
                # ============================================================
                {
                    "tool_func": vfs_share_create,
                    "name": "vfs_share_create",
                    "category": ["vfs", "sharing"],
                    "description": "Create a shareable link/ID for a VFS directory to share it with other sessions/agents.",
                },
                {
                    "tool_func": vfs_share_list,
                    "name": "vfs_share_list",
                    "category": ["vfs", "sharing"],
                    "description": "List all directories currently shared with this agent.",
                },
                {
                    "tool_func": vfs_share_mount,
                    "name": "vfs_share_mount",
                    "category": ["vfs", "sharing"],
                    "description": "Mount a shared directory from another agent into your VFS using the share ID.",
                },
                # Docker (Flag-based)
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
                {
                    "tool_func": history,
                    "name": "history",
                    "category": ["memory", "history"],
                },
                # Situation/Behavior
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
            ]
        )

        if os.getenv("WITH_GOOGLE_TOOLS", "false") == "true":
            tools.extend(gmail_toolkit.get_tools(session.session_id))
            tools.extend(calendar_toolkit.get_tools(session.session_id))

        # =========================================================================
        # CODE EXECUTION TOOLS (Local + Docker)
        # =========================================================================
        from toolboxv2.mods.isaa.base.Agent.code_executor import (
            create_local_code_exec_tool,
            create_docker_code_exec_tool,
        )

        local_exec_tool = create_local_code_exec_tool(self)
        docker_exec_tool = create_docker_code_exec_tool(self)

        tools.append(local_exec_tool)
        tools.append(docker_exec_tool)

        # Register all tools
        self.add_tools(tools)

        session.tools_initialized = True
        logger.info(f"{len(tools)} Tools initialized for session {session.session_id}")

        return tools

    # =========================================================================
    # CONTEXT AWARENESS & ANALYTICS
    # =========================================================================

    async def context_overview(
        self,
        session_id: str | None = None,
        print_visual: bool = True,
        f_print=None,
        query_preview: str = ""
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

        # Versuch, den aktiven Kontext wiederherzustellen oder einen neuen zu simulieren
        ctx = None
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
                engine._calculate_tool_relevance(ctx, query_preview or "status check")
                engine._preload_skill_tools(ctx, query_preview or "status check")
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

        # D. Simulierter nächster User Input
        next_msg = [{"role": "user", "content": query_preview or "(Next User Input Placeholder)"}]

        # 3. Message Stack rekonstruieren (Engine-Logik)
        final_messages = []

        # Wenn Working History existiert, ist der System Prompt dort meist Index 0.
        # Wir ersetzen ihn durch den FRISCHEN System Prompt (mit aktuellen VFS-Daten).
        if work_history and len(work_history) > 0 and work_history[0].get('role') == 'system':
            final_messages = [{"role": "system", "content": sys_prompt_content}] + work_history[1:] + next_msg
        else:
            final_messages = [{"role": "system", "content": sys_prompt_content}] + perm_history + next_msg

        # 4. Präzises Token Counting mit Overhead
        model = self.amd.fast_llm_model.split("/")[-1]
        try:
            model_info = litellm.get_model_info(model)
            context_limit = model_info.get("max_input_tokens") or model_info.get("max_tokens") or 128000
        except:
            context_limit = 128000

        def count(msgs, tools=None):
            # Nutzt die exakte Tokenizer-Logik des Modells inkl. Protokoll-Overhead
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
        t_sys_total = count([{"role": "system", "content": sys_prompt_content}])
        t_tools = count([], tools=active_tools)  # Nur die Tool-Definitionen

        # History ist alles zwischen System (Index 0) und User (Letzter Index)
        msgs_between = final_messages[1:-1] if len(final_messages) > 2 else []
        t_hist = count(msgs_between)

        t_user = count(next_msg)

        # Total (Besser als Summe, da der Tokenizer Optimierungen bei Kontext-Fusion hat)
        t_total = count(final_messages, tools=active_tools)

        # Sub-Komponenten Zählung (String-basiert, da reine Text-Teile)
        t_vfs = count(vfs_content)
        t_base = count(base_sys)
        t_skills = count(skills_content)

        metrics = {
            "session_id": target_session,
            "model": model,
            "t_total": t_total,
            "limit": context_limit,
            "breakdown": {
                "System & VFS": t_sys_total,
                "Active Tools": t_tools,
                "History (Perm+Work)": t_hist,
                "Next Input": t_user
            },
            "system_details": {
                "VFS Content": t_vfs,
                "Base Rules": t_base,
                "Skills": t_skills
            },
            "meta": {
                "tool_count": len(active_tools),
                "msg_count": len(final_messages),
                "dynamic_tools_loaded": len(ctx.dynamic_tools) if ctx else 0
            }
        }

        if print_visual:
            self._print_context_visual(metrics, model, f_print=f_print)

        return metrics

    def _print_context_visual(self, m: dict, model_name: str, f_print=None):
        """
        Visualisiert den Context-Load als übersichtliches Budget-Dashboard.
        """
        if f_print is None:
            f_print = print

        # Colors & Styles
        C_RESET = "\033[0m"
        C_DIM = "\033[90m"
        C_CYAN = "\033[36m"
        C_GREEN = "\033[32m"
        C_YELLOW = "\033[33m"
        C_RED = "\033[31m"
        C_BOLD = "\033[1m"
        C_WHITE = "\033[97m"

        total = m["t_total"]
        limit = m["limit"]
        usage_pct = (total / limit) * 100

        # Ampel-Farben für die Load-Bar
        if usage_pct < 50:
            bar_color = C_GREEN
        elif usage_pct < 80:
            bar_color = C_YELLOW
        else:
            bar_color = C_RED

        # Header
        f_print(f"\n{C_BOLD}🔍 CONTEXT X-RAY{C_RESET}  {C_DIM}Session:{C_RESET} {C_CYAN}{m['session_id']}{C_RESET}")
        f_print(f"{C_DIM}Model: {model_name} | Limit: {limit:,} tokens{C_RESET}")
        f_print(f"{C_DIM}────────────────────────────────────────────────────────────{C_RESET}")

        # 1. The Budget Bar
        bar_width = 40
        filled = int((total / limit) * bar_width)
        bar = "█" * filled + C_DIM + "░" * (bar_width - filled) + C_RESET

        f_print(f"Load:  {bar_color}{bar}{C_RESET} {C_BOLD}{usage_pct:.1f}%{C_RESET}")
        f_print(f"       {total:,} / {limit:,} tokens used\n")

        # 2. Detailed Breakdown Table
        f_print(f"{C_BOLD}{'COMPONENT':<25} {'TOKENS':>10} {'% LOAD':>10}{C_RESET}")
        f_print(f"{C_DIM}{'-' * 48}{C_RESET}")

        def print_row(label, value, is_sub=False, extra_info=""):
            pct = (value / total) * 100 if total > 0 else 0

            if is_sub:
                prefix = f"{C_DIM}  └─ "
                color = C_DIM
            else:
                prefix = ""
                color = C_RESET

            fmt_label = f"{prefix}{label}"
            f_print(f"{color}{fmt_label:<25} {value:>10,} {pct:>9.1f}%{C_RESET} {extra_info}")

        # System Breakdown
        bd = m["breakdown"]
        sys_det = m["system_details"]

        print_row("System Prompt", bd["System & VFS"])
        print_row("VFS Files", sys_det["VFS Content"], True)
        print_row("Base Rules", sys_det["Base Rules"], True)
        if sys_det["Skills"] > 0:
            print_row("Skills", sys_det["Skills"], True)

        # Tools Breakdown
        meta = m["meta"]
        print_row("Active Tools", bd["Active Tools"])
        tool_info = f"{C_DIM}({meta['tool_count']} defs, {meta['dynamic_tools_loaded']} dyn){C_RESET}"
        f_print(f"     {C_DIM}└─ API Schema Cost{C_RESET}         {tool_info}")

        # History Breakdown
        print_row("History & Memory", bd["History (Perm+Work)"])
        f_print(f"     {C_DIM}└─ {meta['msg_count']} messages in stack{C_RESET}")

        # User Input
        print_row("Next Input (Sim)", bd["Next Input"])

        f_print(f"{C_DIM}{'-' * 48}{C_RESET}")

        # 3. Smart Warnings
        if usage_pct > 85:
            f_print(f"\n{C_RED}⚠️  CRITICAL TOKEN LOAD{C_RESET}")
            f_print("   Action: Execute 'shift_focus' tool to compress history.")
        elif sys_det["VFS Content"] > 4000:
            f_print(f"\n{C_YELLOW}⚠️  Heavy VFS Load{C_RESET}")
            f_print("   Action: Close unused files using 'vfs_close'.")
        elif bd["History (Perm+Work)"] > 6000:
            f_print(f"\n{C_YELLOW}⚠️  Long Context{C_RESET}")
            f_print("   Note: Working history is getting long. Consider summarizing.")

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
        if not MCP_AVAILABLE:
            logger.warning("MCP not available")
            return

        server_name = name or f"{self.amd.name}_MCP"
        self.mcp_server = FastMCP(server_name)

        @self.mcp_server.tool()
        async def agent_run(query: str, session_id: str = "mcp_session") -> str:
            return await self.a_run(query, session_id=session_id)

    def setup_a2a_server(self, host: str = "0.0.0.0", port: int = 5000):
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
        self.executor.shutdown(wait=True)

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
