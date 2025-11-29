# toolboxv2/mods/CloudM/UserInstances.py
"""
User Instance Management with Clerk Integration
Handles web and CLI sessions, user instances, and session lifecycle
"""

import json
import time
from typing import Optional, Dict, List, Any

from toolboxv2 import Style, get_app
from toolboxv2.utils import Singleton
from toolboxv2.utils.security.cryp import Code
from toolboxv2.utils.system.types import Result

app = get_app("UserInstances")
logger = app.logger
Name = "CloudM.UserInstances"
version = "0.1.0"
export = app.tb
e = export(mod_name=Name, api=False)
in_mem_cache_150 = export(mod_name=Name, memory_cache=True, memory_cache_max_size=150, version=version)


class UserInstances(metaclass=Singleton):
    """
    Singleton class managing all user instances and sessions.
    Supports both web (WebSocket) and CLI sessions.
    """
    live_user_instances: Dict[str, dict] = {}
    user_instances: Dict[str, str] = {}
    cli_sessions: Dict[str, dict] = {}  # CLI session tracking
    clerk_sessions: Dict[str, dict] = {}  # Clerk session mapping

    @property
    def app(self):
        return get_app("UserInstances")

    @app.setter
    def app(self, v):
        pass

    @staticmethod
    @in_mem_cache_150
    def get_si_id(uid: str) -> Result:
        """Generate Session Instance ID"""
        return Result.ok(data=Code.one_way_hash(uid, app.id, 'SiID'))

    @staticmethod
    @in_mem_cache_150
    def get_vt_id(uid: str) -> Result:
        """Generate Virtual Instance ID"""
        return Result.ok(data=Code.one_way_hash(uid, app.id, 'VirtualInstanceID'))

    @staticmethod
    @in_mem_cache_150
    def get_web_socket_id(uid: str) -> Result:
        """Generate WebSocket ID"""
        return Result.ok(data=Code.one_way_hash(uid, app.id, 'CloudM-Signed'))

    @staticmethod
    @in_mem_cache_150
    def get_cli_session_id(uid: str) -> Result:
        """Generate CLI Session ID"""
        return Result.ok(data=Code.one_way_hash(uid, app.id, 'CLI-Session'))

    @staticmethod
    @in_mem_cache_150
    def get_clerk_session_key(clerk_user_id: str) -> Result:
        """Generate Clerk Session Key for mapping"""
        return Result.ok(data=Code.one_way_hash(clerk_user_id, app.id, 'Clerk-Session'))


# =================== Web Instance Management ===================

@e
def close_user_instance(uid: str):
    """Close a user's web instance and save state"""
    if uid is None:
        return

    si_id = UserInstances.get_si_id(uid).get()

    if si_id not in UserInstances().live_user_instances:
        logger.warning(f"User instance not found for uid: {uid}")
        return "User instance not found"

    instance = UserInstances().live_user_instances[si_id]
    UserInstances().user_instances[instance['SiID']] = instance['webSocketID']

    # Save instance state to database
    app.run_any(
        'DB', 'set',
        query=f"User::Instance::{uid}",
        data=json.dumps({"saves": instance['save']})
    )

    if not instance.get('live'):
        save_user_instances(instance)
        logger.info("No modules to close")
        return "No modules to close"

    # Close all live modules
    for mod_name, spec in instance['live'].items():
        logger.info(f"Closing module: {mod_name}")
        app.remove_mod(mod_name=mod_name, spec=spec, delete=False)

    instance['live'] = {}
    logger.info(f"User instance closed for uid: {uid}")
    save_user_instances(instance)

    return "Instance closed successfully"


@e
def validate_ws_id(ws_id: str) -> tuple:
    """Validate WebSocket ID and return (is_valid, session_key)"""
    logger.debug(f"Validating WebSocket ID: {ws_id}")

    if len(UserInstances().user_instances) == 0:
        # Load from database
        data = app.run_any('DB', 'get', query=f"user_instances::{app.id}")
        if isinstance(data, str):
            try:
                UserInstances().user_instances = json.loads(data)
                logger.info(Style.GREEN("Loaded user instances from DB"))
            except Exception as e:
                logger.error(Style.RED(f"Error loading instances: {e}"))

    if not UserInstances().user_instances:
        return False, ""

    # Find matching session
    for key, value in UserInstances().user_instances.items():
        if value == ws_id:
            return True, key

    return False, ""


@e
def delete_user_instance(uid: str):
    """Delete a user instance completely"""
    if uid is None:
        return "UID required"

    si_id = UserInstances.get_si_id(uid).get()

    if si_id not in UserInstances().user_instances:
        return "User instance not found"

    if si_id in UserInstances().live_user_instances:
        del UserInstances().live_user_instances[si_id]

    del UserInstances().user_instances[si_id]
    app.run_any('DB', 'delete', query=f"User::Instance::{uid}")

    return "Instance deleted successfully"


@e
def save_user_instances(instance: dict):
    """Save user instance to memory and database"""
    if instance is None:
        return

    logger.debug("Saving user instance")
    UserInstances().user_instances[instance['SiID']] = instance['webSocketID']
    UserInstances().live_user_instances[instance['SiID']] = instance

    app.run_any(
        'DB', 'set',
        query=f"user_instances::{app.id}",
        data=json.dumps(UserInstances().user_instances)
    )


@e
def get_instance_si_id(si_id: str) -> Optional[dict]:
    """Get live instance by Session Instance ID"""
    return UserInstances().live_user_instances.get(si_id, None)


@e
def get_user_instance(uid: str, hydrate: bool = True) -> Optional[dict]:
    """
    Get or create a user instance.

    Args:
        uid: User identifier (can be Clerk user ID or legacy UID)
        hydrate: Whether to load modules into the instance

    Returns:
        Instance dictionary with session info and loaded modules
    """
    if uid is None:
        return None

    instance = {
        'save': {
            'uid': uid,
            'mods': [],
        },
        'live': {},
        'webSocketID': UserInstances.get_web_socket_id(uid).get(),
        'SiID': UserInstances.get_si_id(uid).get(),
        'VtID': UserInstances.get_vt_id(uid).get()
    }

    # Check if instance already exists in memory
    if instance['SiID'] in UserInstances().live_user_instances:
        instance_live = UserInstances().live_user_instances.get(instance['SiID'], {})
        if instance_live.get('live') and instance_live.get('save', {}).get('mods'):
            logger.info(Style.BLUEBG2("Instance returned from live cache"))
            return instance_live

    # Check known instances
    cache = {}
    if instance['SiID'] in UserInstances().user_instances:
        instance['webSocketID'] = UserInstances().user_instances[instance['SiID']]
    else:
        # Load from database
        cache_data = app.run_any('DB', 'get', query=f"User::Instance::{uid}", get_results=True)
        if not cache_data.is_data():
            cache = {"saves": instance['save']}
        else:
            cache = cache_data.get()

    # Process cached data
    if cache:
        if isinstance(cache, list):
            cache = cache[0]
        if isinstance(cache, dict):
            instance['save'] = cache.get("saves", instance['save'])
        else:
            try:
                instance['save'] = json.loads(cache).get("saves", instance['save'])
            except Exception as e:
                logger.error(Style.YELLOW(f"Error loading instance cache: {e}"))

    logger.info(Style.BLUEBG(f"Init mods: {instance['save']['mods']}"))

    if hydrate:
        instance = hydrate_instance(instance)

    save_user_instances(instance)
    return instance


@e
def hydrate_instance(instance: dict) -> dict:
    """Load modules into an instance"""
    if instance is None:
        return instance

    existing_mods = set(instance.get('live', {}).keys())

    for mod_name in instance['save']['mods']:
        if mod_name in existing_mods:
            continue

        mod = app.get_mod(mod_name, instance['VtID'])
        app.print(f"{mod_name}.instance_{mod.spec} online")
        instance['live'][mod_name] = mod.spec

    return instance


# =================== CLI Session Management ===================

@e
def register_cli_session(
    uid: str,
    session_token: str,
    session_info: Optional[dict] = None,
    clerk_user_id: Optional[str] = None
) -> Result:
    """
    Register a new CLI session.

    Args:
        uid: User identifier
        session_token: JWT or session token
        session_info: Additional session metadata
        clerk_user_id: Clerk user ID if using Clerk auth

    Returns:
        Result with session data
    """
    if uid is None:
        return Result.default_user_error("UID required")

    cli_session_id = UserInstances.get_cli_session_id(uid).get()

    # Close any existing CLI session for this user (nur eine Session pro User)
    existing_sessions = [
        sid for sid, data in UserInstances().cli_sessions.items()
        if data.get('uid') == uid and data.get('status') == 'active'
    ]
    for existing_sid in existing_sessions:
        logger.info(f"Closing existing CLI session for user {uid}: {existing_sid}")
        close_cli_session(existing_sid)

    session_data = {
        'uid': uid,
        'cli_session_id': cli_session_id,
        'session_token': session_token,
        'clerk_user_id': clerk_user_id,
        'created_at': time.time(),
        'last_activity': time.time(),
        'status': 'active',
        'session_info': session_info or {}
    }

    UserInstances().cli_sessions[cli_session_id] = session_data

    # Map Clerk session if provided
    if clerk_user_id:
        clerk_key = UserInstances.get_clerk_session_key(clerk_user_id).get()
        UserInstances().clerk_sessions[clerk_key] = {
            'cli_session_id': cli_session_id,
            'uid': uid
        }

    # Save to persistent storage
    app.run_any(
        'DB', 'set',
        query=f"CLI::Session::{uid}::{cli_session_id}",
        data=json.dumps(session_data)
    )

    logger.info(f"CLI session registered for user {uid}")
    return Result.ok(info="CLI session registered", data=session_data)


@e
def update_cli_session_activity(cli_session_id: str) -> bool:
    """Update last activity timestamp for CLI session"""
    if cli_session_id not in UserInstances().cli_sessions:
        return False

    UserInstances().cli_sessions[cli_session_id]['last_activity'] = time.time()
    session_data = UserInstances().cli_sessions[cli_session_id]

    # Update persistent storage
    app.run_any(
        'DB', 'set',
        query=f"CLI::Session::{session_data['uid']}::{cli_session_id}",
        data=json.dumps(session_data)
    )

    return True


@e
def close_cli_session(cli_session_id: str) -> str:
    """Close a CLI session"""
    if cli_session_id not in UserInstances().cli_sessions:
        return "CLI session not found"

    session_data = UserInstances().cli_sessions[cli_session_id]
    session_data['status'] = 'closed'
    session_data['closed_at'] = time.time()

    # Remove Clerk mapping if exists
    clerk_user_id = session_data.get('clerk_user_id')
    if clerk_user_id:
        clerk_key = UserInstances.get_clerk_session_key(clerk_user_id).get()
        if clerk_key in UserInstances().clerk_sessions:
            del UserInstances().clerk_sessions[clerk_key]

    # Remove from active sessions
    del UserInstances().cli_sessions[cli_session_id]

    # Update persistent storage to mark as closed
    app.run_any(
        'DB', 'set',
        query=f"CLI::Session::{session_data['uid']}::{cli_session_id}",
        data=json.dumps(session_data)
    )

    logger.info(f"CLI session {cli_session_id} closed")
    return "CLI session closed successfully"


@e
def get_user_cli_sessions(uid: str) -> List[dict]:
    """Get all CLI sessions for a user"""
    if uid is None:
        return []

    active_sessions = [
        session_data
        for session_id, session_data in UserInstances().cli_sessions.items()
        if session_data.get('uid') == uid
    ]

    return active_sessions


@e
def get_cli_session_by_clerk_id(clerk_user_id: str) -> Optional[dict]:
    """Get CLI session by Clerk user ID"""
    clerk_key = UserInstances.get_clerk_session_key(clerk_user_id).get()

    if clerk_key in UserInstances().clerk_sessions:
        cli_session_id = UserInstances().clerk_sessions[clerk_key].get('cli_session_id')
        if cli_session_id in UserInstances().cli_sessions:
            return UserInstances().cli_sessions[cli_session_id]

    return None


@e
def get_all_active_cli_sessions() -> List[dict]:
    """Get all active CLI sessions"""
    return [
        session_data
        for session_data in UserInstances().cli_sessions.values()
        if session_data.get('status') == 'active'
    ]


@e
def cleanup_expired_cli_sessions(max_age_hours: int = 24) -> str:
    """Clean up expired CLI sessions"""
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    expired_sessions = [
        session_id
        for session_id, session_data in list(UserInstances().cli_sessions.items())
        if current_time - session_data.get('last_activity', 0) > max_age_seconds
    ]

    for session_id in expired_sessions:
        close_cli_session(session_id)

    logger.info(f"Cleaned up {len(expired_sessions)} expired CLI sessions")
    return f"Cleaned up {len(expired_sessions)} expired CLI sessions"


# =================== Enhanced Instance Functions ===================

@e
def get_user_instance_with_cli_sessions(uid: str, hydrate: bool = True) -> Optional[dict]:
    """Get user instance with CLI sessions included"""
    instance = get_user_instance(uid, hydrate)

    if instance:
        cli_sessions = get_user_cli_sessions(uid)
        instance['cli_sessions'] = cli_sessions
        instance['active_cli_sessions'] = len([
            s for s in cli_sessions if s.get('status') == 'active'
        ])

    return instance


@e
def get_instance_overview(si_id: str = None) -> dict:
    """Get comprehensive overview of all instances and sessions"""
    overview = {
        'web_instances': {},
        'cli_sessions': {},
        'clerk_sessions': {},
        'total_active_web': 0,
        'total_active_cli': 0
    }

    # Web instances
    if si_id:
        if si_id in UserInstances().live_user_instances:
            overview['web_instances'][si_id] = UserInstances().live_user_instances[si_id]
            overview['total_active_web'] = 1
    else:
        overview['web_instances'] = dict(UserInstances().live_user_instances)
        overview['total_active_web'] = len(UserInstances().live_user_instances)

    # CLI sessions
    overview['cli_sessions'] = dict(UserInstances().cli_sessions)
    overview['total_active_cli'] = len([
        s for s in UserInstances().cli_sessions.values()
        if s.get('status') == 'active'
    ])

    # Clerk sessions
    overview['clerk_sessions'] = dict(UserInstances().clerk_sessions)

    return overview


# =================== Session Validation ===================

@export(mod_name=Name, state=False, test=False)
def save_close_user_instance(ws_id: str) -> Result:
    """Validate WebSocket ID and close associated instance"""
    valid, key = validate_ws_id(ws_id)

    if valid:
        user_instance = UserInstances().live_user_instances.get(key)
        if user_instance:
            logger.info(f"Logging out user with WebSocket ID: {ws_id}")
            close_user_instance(user_instance['save']['uid'])
            return Result.ok()

    return Result.default_user_error(info="Invalid WebSocket ID")


@e
def validate_cli_session_token(cli_session_id: str, token: str) -> bool:
    """Validate CLI session token"""
    if cli_session_id not in UserInstances().cli_sessions:
        return False

    session_data = UserInstances().cli_sessions[cli_session_id]

    if session_data.get('status') != 'active':
        return False

    if session_data.get('session_token') != token:
        return False

    # Update activity
    update_cli_session_activity(cli_session_id)
    return True
