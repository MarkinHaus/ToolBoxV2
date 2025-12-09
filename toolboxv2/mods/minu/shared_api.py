# toolboxv2/mods/Minu/shared_api.py
"""
API Endpunkte für Shared Sections.
Diese Endpunkte in __init__.py integrieren.

Ermöglicht:
- REST API für Shared Section Management
- WebSocket Events für Live-Updates
"""

import json
from typing import Any, Dict, List, Optional

from toolboxv2 import App, Result, RequestData, get_app

from .shared import (
    SharedManager,
    SharedSection,
    SharedParticipant,
    SharedPermission,
    SharedChange,
)

Name = "Minu"
export = get_app(f"{Name}.Export").tb
version = "0.1.0"


# ============================================================================
# REST API ENDPOINTS
# ============================================================================


@export(
    mod_name=Name,
    name="shared/create",
    api=True,
    api_methods=["POST"],
    version=version,
    request_as_kwarg=True,
)
async def create_shared_section(
    app: App,
    request: RequestData,
    name: str,
    initial_data: Optional[Dict[str, Any]] = None,
    max_participants: int = 100,
    allow_anonymous: bool = True,
    default_permission: str = "write",
    public: bool = False,
) -> Result:
    """
    Neue Shared Section erstellen.

    POST /api/Minu/shared/create
    {
        "name": "my_game_lobby",
        "initial_data": {"state": "waiting", "players": []},
        "max_participants": 4,
        "allow_anonymous": true,
        "public": true
    }

    Returns:
        Section ID und Details
    """
    manager = SharedManager.get_(app)

    try:
        section = await manager.create(
            request=request,
            name=name,
            initial_data=initial_data,
            max_participants=max_participants,
            allow_anonymous=allow_anonymous,
            default_permission=SharedPermission(default_permission),
            public=public,
        )

        return Result.ok(
            data={
                "section_id": section.id,
                "name": section.name,
                "owner_id": section.owner_id,
                "participant_count": len(section.participants),
            }
        )
    except Exception as e:
        app.logger.error(f"[Shared] Create error: {e}")
        return Result.default_internal_error(str(e))


@export(
    mod_name=Name,
    name="shared/join",
    api=True,
    api_methods=["POST"],
    version=version,
    request_as_kwarg=True,
)
async def join_shared_section(
    app: App,
    request: RequestData,
    section_id: str,
) -> Result:
    """
    Shared Section beitreten.

    POST /api/Minu/shared/join
    {"section_id": "shared-abc123"}
    """
    manager = SharedManager.get_(app)

    section = await manager.join(section_id, request)

    if not section:
        return Result.default_user_error(
            info="Section nicht gefunden oder Zugriff verweigert", exec_code=404
        )

    return Result.ok(data=section.to_dict())


@export(
    mod_name=Name,
    name="shared/leave",
    api=True,
    api_methods=["POST"],
    version=version,
    request_as_kwarg=True,
)
async def leave_shared_section(
    app: App,
    request: RequestData,
    section_id: str,
) -> Result:
    """
    Shared Section verlassen.

    POST /api/Minu/shared/leave
    {"section_id": "shared-abc123"}
    """
    manager = SharedManager.get_(app)

    result = await manager.leave(section_id, request)

    if not result:
        return Result.default_user_error(info="Section nicht gefunden")

    return Result.ok(data_info="Section verlassen")


@export(
    mod_name=Name,
    name="shared/get",
    api=True,
    api_methods=["GET"],
    version=version,
    request_as_kwarg=True,
)
async def get_shared_section(
    app: App,
    request: RequestData,
    section_id: str,
) -> Result:
    """
    Shared Section Details abrufen.

    GET /api/Minu/shared/get?section_id=shared-abc123
    """
    manager = SharedManager.get_(app)

    section = await manager.get(section_id)

    if not section:
        return Result.default_user_error(info="Section nicht gefunden", exec_code=404)

    return Result.ok(data=section.to_dict())


@export(
    mod_name=Name,
    name="shared/update",
    api=True,
    api_methods=["POST"],
    version=version,
    request_as_kwarg=True,
)
async def update_shared_data(
    app: App,
    request: RequestData,
    section_id: str,
    path: str,
    value: Any,
    operation: str = "set",  # set, merge, append, remove, delete
) -> Result:
    """
    Daten in Shared Section ändern.

    POST /api/Minu/shared/update
    {
        "section_id": "shared-abc123",
        "path": "state",
        "value": "playing",
        "operation": "set"
    }
    """
    from .user import MinuUser

    manager = SharedManager.get_(app)
    section = await manager.get(section_id)

    if not section:
        return Result.default_user_error(info="Section nicht gefunden", exec_code=404)

    user = await MinuUser.from_request(app, request)

    # Berechtigung prüfen
    if not section.has_permission(user.uid, SharedPermission.WRITE):
        return Result.default_user_error(info="Keine Schreibberechtigung", exec_code=403)

    # Operation ausführen
    try:
        if operation == "set":
            await section.set(path, value, author_id=user.uid)
        elif operation == "merge":
            await section.merge(path, value, author_id=user.uid)
        elif operation == "append":
            await section.append(path, value, author_id=user.uid)
        elif operation == "remove":
            await section.remove(path, value=value, author_id=user.uid)
        elif operation == "delete":
            await section.delete(path, author_id=user.uid)
        else:
            return Result.default_user_error(info=f"Unbekannte Operation: {operation}")

        return Result.ok(data={"path": path, "operation": operation})
    except Exception as e:
        return Result.default_internal_error(str(e))


@export(
    mod_name=Name,
    name="shared/list",
    api=True,
    api_methods=["GET"],
    version=version,
    request_as_kwarg=True,
)
async def list_shared_sections(
    app: App,
    request: RequestData,
    public_only: bool = False,
) -> Result:
    """
    Shared Sections des Users oder öffentliche auflisten.

    GET /api/Minu/shared/list
    GET /api/Minu/shared/list?public_only=true
    """
    manager = SharedManager.get_(app)

    if public_only:
        sections = await manager.list_public()
    else:
        sections = await manager.list_user_sections(request)

    return Result.ok(data=sections)


@export(
    mod_name=Name,
    name="shared/delete",
    api=True,
    api_methods=["DELETE", "POST"],
    version=version,
    request_as_kwarg=True,
)
async def delete_shared_section(
    app: App,
    request: RequestData,
    section_id: str,
) -> Result:
    """
    Shared Section löschen (nur Owner).

    DELETE /api/Minu/shared/delete?section_id=shared-abc123
    """
    manager = SharedManager.get_(app)

    result = await manager.delete(section_id, request)

    if not result:
        return Result.default_user_error(
            info="Section nicht gefunden oder keine Berechtigung"
        )

    return Result.ok(data_info="Section gelöscht")


# ============================================================================
# WEBSOCKET INTEGRATION
# ============================================================================


def get_shared_websocket_handlers(app: App):
    """
    WebSocket Handler für Shared Section Events.
    In den bestehenden Minu WebSocket Handler integrieren.

    Neue Message Types:
    - shared_subscribe: Section abonnieren
    - shared_unsubscribe: Abo beenden
    - shared_update: Daten ändern
    """

    # Session -> Subscribed Section IDs
    _subscriptions: Dict[str, set] = {}

    async def handle_shared_message(
        session_id: str,
        msg_type: str,
        payload: Dict[str, Any],
        request: RequestData,
    ) -> Optional[Dict]:
        """
        Shared-spezifische WebSocket Messages verarbeiten.
        """
        from .user import MinuUser

        manager = SharedManager.get_(app)

        if msg_type == "shared_subscribe":
            section_id = payload.get("sectionId")

            section = await manager.join(section_id, request)
            if not section:
                return {"type": "error", "message": "Section nicht gefunden"}

            # Subscription tracken
            if session_id not in _subscriptions:
                _subscriptions[session_id] = set()
            _subscriptions[session_id].add(section_id)

            return {
                "type": "shared_subscribed",
                "sectionId": section_id,
                "data": section.to_dict(),
            }

        elif msg_type == "shared_unsubscribe":
            section_id = payload.get("sectionId")

            if session_id in _subscriptions:
                _subscriptions[session_id].discard(section_id)

            await manager.leave(section_id, request)

            return {"type": "shared_unsubscribed", "sectionId": section_id}

        elif msg_type == "shared_update":
            section_id = payload.get("sectionId")
            path = payload.get("path")
            value = payload.get("value")
            operation = payload.get("operation", "set")

            section = await manager.get(section_id)
            if not section:
                return {"type": "error", "message": "Section nicht gefunden"}

            user = await MinuUser.from_request(app, request)

            if not section.has_permission(user.uid, SharedPermission.WRITE):
                return {"type": "error", "message": "Keine Schreibberechtigung"}

            # Operation ausführen
            if operation == "set":
                await section.set(path, value, author_id=user.uid)
            elif operation == "merge":
                await section.merge(path, value, author_id=user.uid)
            elif operation == "append":
                await section.append(path, value, author_id=user.uid)
            elif operation == "remove":
                await section.remove(path, value=value, author_id=user.uid)
            elif operation == "delete":
                await section.delete(path, author_id=user.uid)

            return {"type": "shared_updated", "sectionId": section_id, "path": path}

        return None

    def cleanup_subscriptions(session_id: str):
        """Aufräumen wenn Session endet"""
        if session_id in _subscriptions:
            del _subscriptions[session_id]

    return {
        "handle_message": handle_shared_message,
        "cleanup": cleanup_subscriptions,
        "subscriptions": _subscriptions,
    }


# ============================================================================
# INTEGRATION IN BESTEHENDEN WEBSOCKET HANDLER
# ============================================================================




# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "create_shared_section",
    "join_shared_section",
    "leave_shared_section",
    "get_shared_section",
    "update_shared_data",
    "list_shared_sections",
    "delete_shared_section",
    "get_shared_websocket_handlers",
]
