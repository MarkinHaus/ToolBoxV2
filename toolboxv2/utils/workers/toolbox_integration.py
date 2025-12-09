#!/usr/bin/env python3
"""
toolbox_integration.py - ToolBoxV2 Integration Layer

Integration between the worker system and ToolBoxV2:
- server_helper() integration
- Module function routing
- Session verification via CloudM.AuthClerk
- Event manager bridge
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def get_toolbox_app(instance_id: str = "worker", **kwargs):
    """Get ToolBoxV2 App instance using server_helper."""
    try:
        from toolboxv2.__main__ import server_helper

        return server_helper(instance_id=instance_id, **kwargs)
    except ImportError as e:
        logger.error(f"Failed to import ToolBoxV2: {e}")
        raise


def verify_session_via_clerk(
    app,
    session_token: str,
    auth_module: str = "CloudM.AuthClerk",
    verify_func: str = "verify_session",
) -> Tuple[bool, Optional[Dict]]:
    """Verify session using CloudM.AuthClerk."""
    try:
        result = app.run_any(
            (auth_module, verify_func),
            session_token=session_token,
            get_results=True,
        )
        if result.is_error():
            return False, None
        data = result.get()
        if not data or not data.get("valid", False):
            return False, None
        return True, data
    except Exception as e:
        logger.error(f"Session verification error: {e}")
        return False, None


async def verify_session_via_clerk_async(
    app,
    session_token: str,
    auth_module: str = "CloudM.AuthClerk",
    verify_func: str = "verify_session",
) -> Tuple[bool, Optional[Dict]]:
    """Async version of verify_session_via_clerk."""
    try:
        result = await app.a_run_any(
            (auth_module, verify_func),
            session_token=session_token,
            get_results=True,
        )
        if result.is_error():
            return False, None
        data = result.get()
        if not data or not data.get("valid", False):
            return False, None
        return True, data
    except Exception as e:
        logger.error(f"Session verification error: {e}")
        return False, None


class ModuleRouter:
    """Routes API requests to ToolBoxV2 module functions."""

    def __init__(self, app, api_prefix: str = "/api"):
        self.app = app
        self.api_prefix = api_prefix

    def parse_path(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse /api/Module/function into (module, function)."""
        if not path.startswith(self.api_prefix):
            return None, None
        stripped = path[len(self.api_prefix) :].strip("/")
        if not stripped:
            return None, None
        parts = stripped.split("/", 1)
        if len(parts) == 1:
            return parts[0], None
        return parts[0], parts[1]

    async def call_function(
        self, module_name: str, function_name: str, request_data: Dict, **kwargs
    ) -> Dict[str, Any]:
        """Call a ToolBoxV2 module function."""
        try:
            kwargs["request"] = request_data
            result = await self.app.a_run_any(
                (module_name, function_name), get_results=True, **kwargs
            )
            return self._convert_result(result, module_name, function_name)
        except Exception as e:
            logger.error(f"Module call error: {module_name}.{function_name}: {e}")
            return {
                "error": "InternalError",
                "origin": [module_name, function_name],
                "result": {"data": None, "data_type": "NoneType"},
                "info": {"exec_code": 500, "help_text": str(e)},
            }

    def call_function_sync(
        self, module_name: str, function_name: str, request_data: Dict, **kwargs
    ) -> Dict[str, Any]:
        """Sync version of call_function."""
        try:
            kwargs["request"] = request_data
            result = self.app.run_any(
                (module_name, function_name), get_results=True, **kwargs
            )
            return self._convert_result(result, module_name, function_name)
        except Exception as e:
            logger.error(f"Module call error: {module_name}.{function_name}: {e}")
            return {
                "error": "InternalError",
                "origin": [module_name, function_name],
                "result": {"data": None, "data_type": "NoneType"},
                "info": {"exec_code": 500, "help_text": str(e)},
            }

    def _convert_result(self, result, module_name: str, function_name: str) -> Dict:
        """Convert ToolBoxV2 Result to API response format."""
        if hasattr(result, "to_api_result"):
            api_result = result.to_api_result()
            if hasattr(api_result, "model_dump"):
                return api_result.model_dump()
            elif hasattr(api_result, "__dict__"):
                return api_result.__dict__

        if hasattr(result, "is_error"):
            error_val = None
            if hasattr(result, "error") and result.error:
                error_val = (
                    result.error.name
                    if hasattr(result.error, "name")
                    else str(result.error)
                )

            data = result.get() if hasattr(result, "get") else result
            data_type = "unknown"
            data_info = ""

            if hasattr(result, "result"):
                data_type = getattr(result.result, "data_type", type(data).__name__)
                data_info = getattr(result.result, "data_info", "")

            exec_code = 0
            help_text = "OK"
            if hasattr(result, "info"):
                exec_code = getattr(result.info, "exec_code", 0)
                help_text = getattr(result.info, "help_text", "OK")

            return {
                "error": error_val if result.is_error() else None,
                "origin": [module_name, function_name],
                "result": {
                    "data": data,
                    "data_type": data_type,
                    "data_info": data_info,
                },
                "info": {
                    "exec_code": exec_code,
                    "help_text": help_text,
                },
            }

        return {
            "error": None,
            "origin": [module_name, function_name],
            "result": {"data": result, "data_type": type(result).__name__},
            "info": {"exec_code": 0, "help_text": "OK"},
        }


class ZMQEventBridge:
    """Bridge between ToolBoxV2 EventManager and ZeroMQ."""

    def __init__(self, app, zmq_event_manager):
        self.app = app
        self.zmq_em = zmq_event_manager
        self._tb_em = None

    def connect(self):
        """Connect to ToolBoxV2 EventManager if available."""
        try:
            if hasattr(self.app, "get_mod"):
                em_mod = self.app.get_mod("EventManager")
                if em_mod and hasattr(em_mod, "get_manager"):
                    self._tb_em = em_mod.get_manager()
                    self._register_bridges()
                    logger.info("Connected to ToolBoxV2 EventManager")
        except Exception as e:
            logger.debug(f"EventManager not available: {e}")

    def _register_bridges(self):
        """Register event bridges between ZMQ and TB."""
        from event_manager import EventType, Event

        @self.zmq_em.on(EventType.CUSTOM)
        async def forward_to_tb(event: Event):
            if self._tb_em and event.payload.get("forward_to_tb"):
                try:
                    self._tb_em.emit(
                        event.payload.get("tb_event_name", "zmq_event"),
                        event.payload.get("data", {}),
                    )
                except Exception as e:
                    logger.debug(f"Failed to forward to TB: {e}")


def create_worker_app(instance_id: str, config) -> Tuple[Any, ModuleRouter]:
    """
    Create ToolBoxV2 app and router for a worker.

    Returns:
        Tuple of (app, router)
    """
    preload = []
    api_prefix = "/api"

    if hasattr(config, "toolbox"):
        preload = getattr(config.toolbox, "modules_preload", [])
        api_prefix = getattr(config.toolbox, "api_prefix", "/api")

    app = get_toolbox_app(instance_id=instance_id, load_mods=preload)
    router = ModuleRouter(app, api_prefix)
    return app, router
