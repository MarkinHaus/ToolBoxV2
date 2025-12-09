#!/usr/bin/env python3
"""
toolbox_integration.py - ToolBoxV2 Integration Layer

Integration between the worker system and ToolBoxV2:
- server_helper() integration
- Module function routing with access control
- Session verification via CloudM.AuthClerk
- Event manager bridge
- Level-based authorization
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Access Level Constants
# ============================================================================


class AccessLevel:
    """User access levels for authorization."""
    ADMIN = -1           # Full access to everything
    NOT_LOGGED_IN = 0    # Anonymous user, only public endpoints
    LOGGED_IN = 1        # Authenticated user
    TRUSTED = 2          # Trusted/verified user


# ============================================================================
# ToolBoxV2 App Integration
# ============================================================================


def get_toolbox_app(instance_id: str = "worker", **kwargs):
    """Get ToolBoxV2 App instance using server_helper."""
    try:
        from toolboxv2.__main__ import server_helper
        return server_helper(instance_id=instance_id, **kwargs)
    except ImportError as e:
        logger.error(f"Failed to import ToolBoxV2: {e}")
        raise


# ============================================================================
# Session Verification
# ============================================================================


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


# ============================================================================
# Access Controller
# ============================================================================


class AccessController:
    """
    Controls access to API endpoints based on:
    - open_modules: Modules that are publicly accessible
    - admin_modules: Modules requiring admin level (-1)
    - Function names: Functions starting with 'open' are public
    - User level: -1=Admin, 0=not logged in, 1=logged in, 2=trusted
    - level_requirements: Per-module/function level overrides
    """

    def __init__(self, config=None):
        self.config = config
        self._open_modules: Set[str] = set()
        self._admin_modules: Set[str] = set()
        self._level_requirements: Dict[str, int] = {}
        self._default_level: int = AccessLevel.LOGGED_IN

        if config:
            self._load_config()

    def _load_config(self):
        """Load access control settings from config."""
        if not hasattr(self.config, 'toolbox'):
            return

        tb = self.config.toolbox

        # Open modules (public)
        self._open_modules = set(getattr(tb, 'open_modules', []))

        # Admin modules
        self._admin_modules = set(getattr(tb, 'admin_modules', [
            "CloudM.AuthClerk",
            "ToolBox",
        ]))

        # Default required level
        self._default_level = getattr(tb, 'default_required_level', AccessLevel.LOGGED_IN)

        # Per-module/function level requirements
        self._level_requirements = getattr(tb, 'level_requirements', {})

        logger.info(
            f"AccessController loaded: "
            f"open_modules={self._open_modules}, "
            f"admin_modules={self._admin_modules}, "
            f"default_level={self._default_level}"
        )

    def reload_config(self, config=None):
        """Reload configuration."""
        if config:
            self.config = config
        self._load_config()

    def is_public_endpoint(self, module_name: str, function_name: str) -> bool:
        """Check if endpoint is publicly accessible (no auth required)."""
        # Module in open_modules list
        if module_name in self._open_modules:
            return True

        # Function starts with 'open' (case insensitive)
        if function_name and function_name.lower().startswith("open"):
            return True

        return False

    def is_admin_only(self, module_name: str, function_name: str = None) -> bool:
        """Check if endpoint requires admin level."""
        # Module in admin_modules list
        if module_name in self._admin_modules:
            return True

        # Check specific function override
        if function_name:
            key = f"{module_name}.{function_name}"
            if key in self._level_requirements:
                return self._level_requirements[key] == AccessLevel.ADMIN

        # Check module-level override
        if module_name in self._level_requirements:
            return self._level_requirements[module_name] == AccessLevel.ADMIN

        return False

    def get_required_level(self, module_name: str, function_name: str) -> int:
        """Get the required access level for an endpoint."""
        # Public endpoints
        if self.is_public_endpoint(module_name, function_name):
            return AccessLevel.NOT_LOGGED_IN

        # Admin-only endpoints
        if self.is_admin_only(module_name, function_name):
            return AccessLevel.ADMIN

        # Check specific function override
        if function_name:
            key = f"{module_name}.{function_name}"
            if key in self._level_requirements:
                return self._level_requirements[key]

        # Check module-level override
        if module_name in self._level_requirements:
            return self._level_requirements[module_name]

        # Default level
        return self._default_level

    def check_access(
        self,
        module_name: str,
        function_name: str,
        user_level: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if user has access to endpoint.

        Args:
            module_name: The module being accessed
            function_name: The function being called
            user_level: The user's access level

        Returns:
            Tuple of (allowed: bool, error_message: Optional[str])
        """
        # Get required level for this endpoint
        required_level = self.get_required_level(module_name, function_name)

        # Admin has access to everything
        if user_level == AccessLevel.ADMIN:
            return True, None

        # Public endpoints (level 0 required)
        if required_level == AccessLevel.NOT_LOGGED_IN:
            return True, None

        # Not logged in but endpoint requires auth
        if user_level == AccessLevel.NOT_LOGGED_IN:
            return False, "Authentication required"

        # Admin-only endpoint
        if required_level == AccessLevel.ADMIN:
            return False, "Admin access required"

        # Check if user meets level requirement
        # Note: For positive levels, higher is better (1 < 2)
        # For admin (-1), we already handled it above
        if user_level >= required_level:
            return True, None

        return False, f"Insufficient permissions (level {user_level}, required {required_level})"

    @staticmethod
    def get_user_level(session) -> int:
        """Extract user level from session object."""
        if not session:
            return AccessLevel.NOT_LOGGED_IN

        level = None

        # Try different ways to get level
        if hasattr(session, 'level'):
            level = session.level
        elif hasattr(session, 'live_data') and isinstance(session.live_data, dict):
            level = session.live_data.get('level')
        elif hasattr(session, 'to_dict'):
            data = session.to_dict()
            level = data.get('level')
        elif isinstance(session, dict):
            level = session.get('level')

        if level is None:
            return AccessLevel.NOT_LOGGED_IN

        try:
            return int(level)
        except (ValueError, TypeError):
            return AccessLevel.NOT_LOGGED_IN


# ============================================================================
# Module Router
# ============================================================================


class ModuleRouter:
    """Routes API requests to ToolBoxV2 module functions with access control."""

    def __init__(
        self,
        app,
        api_prefix: str = "/api",
        access_controller: AccessController = None,
    ):
        self.app = app
        self.api_prefix = api_prefix
        self.access_controller = access_controller or AccessController()

    def parse_path(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse /api/Module/function into (module, function)."""
        if not path.startswith(self.api_prefix):
            return None, None
        stripped = path[len(self.api_prefix):].strip("/")
        if not stripped:
            return None, None
        parts = stripped.split("/", 1)
        if len(parts) == 1:
            return parts[0], None
        return parts[0], parts[1]

    def check_access(
        self,
        module_name: str,
        function_name: str,
        session,
    ) -> Tuple[bool, Optional[str], int]:
        """
        Check access for a request.

        Returns:
            Tuple of (allowed, error_message, user_level)
        """
        user_level = self.access_controller.get_user_level(session)
        allowed, error = self.access_controller.check_access(
            module_name, function_name, user_level
        )
        return allowed, error, user_level

    async def call_function(
        self,
        module_name: str,
        function_name: str,
        request_data: Dict,
        session=None,
        check_access: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Call a ToolBoxV2 module function with optional access check."""
        # Access check
        if check_access:
            allowed, error, user_level = self.check_access(
                module_name, function_name, session
            )
            if not allowed:
                logger.warning(
                    f"Access denied: {module_name}.{function_name} "
                    f"(level={user_level}): {error}"
                )
                return {
                    "error": "Forbidden" if user_level > 0 else "Unauthorized",
                    "origin": [module_name, function_name],
                    "result": {"data": None, "data_type": "NoneType"},
                    "info": {
                        "exec_code": 403 if user_level > 0 else 401,
                        "help_text": error,
                    },
                }

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
        self,
        module_name: str,
        function_name: str,
        request_data: Dict,
        session=None,
        check_access: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Sync version of call_function."""
        # Access check
        if check_access:
            allowed, error, user_level = self.check_access(
                module_name, function_name, session
            )
            if not allowed:
                logger.warning(
                    f"Access denied: {module_name}.{function_name} "
                    f"(level={user_level}): {error}"
                )
                return {
                    "error": "Forbidden" if user_level > 0 else "Unauthorized",
                    "origin": [module_name, function_name],
                    "result": {"data": None, "data_type": "NoneType"},
                    "info": {
                        "exec_code": 403 if user_level > 0 else 401,
                        "help_text": error,
                    },
                }

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


# ============================================================================
# ZMQ Event Bridge
# ============================================================================


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
        from toolboxv2.utils.workers.event_manager import EventType, Event

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


# ============================================================================
# Factory Functions
# ============================================================================


def create_worker_app(
    instance_id: str,
    config,
) -> Tuple[Any, ModuleRouter, AccessController]:
    """
    Create ToolBoxV2 app, router, and access controller for a worker.

    Returns:
        Tuple of (app, router, access_controller)
    """
    preload = []
    api_prefix = "/api"

    if hasattr(config, "toolbox"):
        preload = getattr(config.toolbox, "modules_preload", [])
        api_prefix = getattr(config.toolbox, "api_prefix", "/api")

    app = get_toolbox_app(instance_id=instance_id, load_mods=preload)

    access_controller = AccessController(config)
    router = ModuleRouter(app, api_prefix, access_controller)

    return app, router, access_controller


def create_access_controller(config) -> AccessController:
    """Create an AccessController from config."""
    return AccessController(config)
