# toolboxv2/mods/Minu/shared.py
"""
Minu Shared Data System
=======================
Kontrollierter Echtzeit-Datenaustausch zwischen Nutzern.

Features:
- Shared Sections: Geteilte Bereiche die für mehrere Nutzer live synchronisiert werden
- Cross-User Support: Angemeldete, anonyme und gemischte Gruppen
- BlobDB Integration: Persistente Speicherung
- WebSocket-basierte Live-Updates
- Zugriffskontrolle: Owner, Participants, Permissions

Use Cases:
- Multiplayer Games
- Chat/Messaging
- Collaborative Editing
- Real-time Dashboards
- Shared Whiteboards
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
import weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from toolboxv2 import App, RequestData

    from .core import MinuSession, MinuView
    from .user import AnonymousUser, AuthenticatedUserWrapper


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================


class SharedPermission(str, Enum):
    """Berechtigungen für Shared Sections"""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"  # Kann andere einladen/entfernen


class ParticipantType(str, Enum):
    """Typ des Teilnehmers"""
    AUTHENTICATED = "authenticated"
    ANONYMOUS = "anonymous"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class SharedParticipant:
    """Ein Teilnehmer in einer Shared Section"""
    id: str  # uid für authenticated, session_id für anonymous
    type: ParticipantType
    name: str
    permission: SharedPermission = SharedPermission.READ
    joined_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Runtime - nicht persistiert
    session: MinuSession | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'name': self.name,
            'permission': self.permission.value,
            'joined_at': self.joined_at,
            'last_seen': self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> SharedParticipant:
        return cls(
            id=data['id'],
            type=ParticipantType(data['type']),
            name=data['name'],
            permission=SharedPermission(data.get('permission', 'read')),
            joined_at=data.get('joined_at', time.time()),
            last_seen=data.get('last_seen', time.time()),
        )


@dataclass
class SharedChange:
    """Eine Änderung in einer Shared Section"""
    path: str  # z.B. "messages", "state.score", "canvas.objects[0]"
    value: Any
    operation: str = "set"  # set, merge, delete, append, remove
    timestamp: float = field(default_factory=time.time)
    author_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'value': self.value,
            'operation': self.operation,
            'timestamp': self.timestamp,
            'author_id': self.author_id,
        }


@dataclass
class SharedSection:
    """
    Eine geteilte Daten-Sektion für mehrere Nutzer.

    Usage:
        # Section erstellen
        section = await SharedManager.create(
            app, request,
            name="game_lobby_123",
            initial_data={'players': [], 'state': 'waiting'}
        )

        # Daten ändern (wird automatisch an alle Teilnehmer gesendet)
        await section.set('state', 'playing')
        await section.append('players', {'name': 'Player1', 'score': 0})

        # Auf Änderungen reagieren
        section.on_change('state', lambda change: print(f"State: {change.value}"))
    """
    id: str
    name: str
    owner_id: str
    owner_type: ParticipantType
    created_at: float = field(default_factory=time.time)

    # Daten
    data: Dict[str, Any] = field(default_factory=dict)

    # Teilnehmer
    participants: Dict[str, SharedParticipant] = field(default_factory=dict)

    # Einstellungen
    max_participants: int = 100
    allow_anonymous: bool = True
    default_permission: SharedPermission = SharedPermission.WRITE
    public: bool = False  # Öffentlich auffindbar

    # Runtime
    _app: App | None = field(default=None, repr=False, compare=False)
    _change_handlers: Dict[str, List[Callable]] = field(default_factory=dict, repr=False, compare=False)
    _pending_changes: List[SharedChange] = field(default_factory=list, repr=False, compare=False)
    _broadcast_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    def to_dict(self, include_participants: bool = True) -> Dict[str, Any]:
        result = {
            'id': self.id,
            'name': self.name,
            'owner_id': self.owner_id,
            'owner_type': self.owner_type.value,
            'created_at': self.created_at,
            'data': self.data,
            'max_participants': self.max_participants,
            'allow_anonymous': self.allow_anonymous,
            'default_permission': self.default_permission.value,
            'public': self.public,
        }
        if include_participants:
            result['participants'] = {
                pid: p.to_dict() for pid, p in self.participants.items()
            }
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> SharedSection:
        participants = {}
        for pid, pdata in data.get('participants', {}).items():
            participants[pid] = SharedParticipant.from_dict(pdata)

        return cls(
            id=data['id'],
            name=data['name'],
            owner_id=data['owner_id'],
            owner_type=ParticipantType(data['owner_type']),
            created_at=data.get('created_at', time.time()),
            data=data.get('data', {}),
            participants=participants,
            max_participants=data.get('max_participants', 100),
            allow_anonymous=data.get('allow_anonymous', True),
            default_permission=SharedPermission(data.get('default_permission', 'write')),
            public=data.get('public', False),
        )

    # =================== Data Access ===================

    def get(self, path: str = None, default: Any = None) -> Any:
        """
        Daten lesen.

        Args:
            path: Pfad zu den Daten (z.B. "state", "players.0.score")
            default: Fallback-Wert
        """
        if path is None:
            return self.data

        parts = path.replace('[', '.').replace(']', '').split('.')
        current = self.data

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, default)
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (IndexError, ValueError):
                    return default
            else:
                return default

            if current is None:
                return default

        return current

    async def set(self, path: str, value: Any, author_id: str = "") -> bool:
        """
        Daten setzen und an alle Teilnehmer broadcasten.
        """
        change = SharedChange(
            path=path,
            value=value,
            operation="set",
            author_id=author_id
        )
        return await self._apply_and_broadcast(change)

    async def merge(self, path: str, value: Dict, author_id: str = "") -> bool:
        """
        Dict-Daten mergen (shallow merge).
        """
        change = SharedChange(
            path=path,
            value=value,
            operation="merge",
            author_id=author_id
        )
        return await self._apply_and_broadcast(change)

    async def append(self, path: str, value: Any, author_id: str = "") -> bool:
        """
        Wert zu Liste hinzufügen.
        """
        change = SharedChange(
            path=path,
            value=value,
            operation="append",
            author_id=author_id
        )
        return await self._apply_and_broadcast(change)

    async def remove(self, path: str, value: Any = None, index: int = None,
                     author_id: str = "") -> bool:
        """
        Wert aus Liste entfernen (by value oder index).
        """
        change = SharedChange(
            path=path,
            value={'value': value, 'index': index},
            operation="remove",
            author_id=author_id
        )
        return await self._apply_and_broadcast(change)

    async def delete(self, path: str, author_id: str = "") -> bool:
        """
        Daten löschen.
        """
        change = SharedChange(
            path=path,
            value=None,
            operation="delete",
            author_id=author_id
        )
        return await self._apply_and_broadcast(change)

    async def _apply_and_broadcast(self, change: SharedChange) -> bool:
        """Änderung anwenden und an alle Teilnehmer senden"""
        async with self._broadcast_lock:
            # 1. Lokal anwenden
            self._apply_change(change)

            # 2. Persistieren
            await self._persist()

            # 3. Change-Handler aufrufen
            self._trigger_handlers(change)

            # 4. An alle Teilnehmer broadcasten
            await self._broadcast_change(change)

            return True

    def _apply_change(self, change: SharedChange):
        """Änderung auf lokale Daten anwenden"""
        parts = change.path.replace('[', '.').replace(']', '').split('.')

        # Navigate to parent
        parent = self.data
        for part in parts[:-1]:
            if isinstance(parent, dict):
                if part not in parent:
                    parent[part] = {}
                parent = parent[part]
            elif isinstance(parent, list):
                parent = parent[int(part)]

        key = parts[-1]

        if change.operation == "set":
            if isinstance(parent, dict):
                parent[key] = change.value
            elif isinstance(parent, list):
                parent[int(key)] = change.value

        elif change.operation == "merge":
            if isinstance(parent, dict) and key in parent:
                if isinstance(parent[key], dict):
                    parent[key] = {**parent[key], **change.value}
                else:
                    parent[key] = change.value
            else:
                parent[key] = change.value

        elif change.operation == "append":
            if isinstance(parent, dict):
                if key not in parent:
                    parent[key] = []
                if isinstance(parent[key], list):
                    parent[key].append(change.value)
            elif isinstance(parent, list):
                parent[int(key)].append(change.value)

        elif change.operation == "remove":
            if isinstance(parent, dict) and key in parent:
                target = parent[key]
                if isinstance(target, list):
                    if change.value.get('index') is not None:
                        del target[change.value['index']]
                    elif change.value.get('value') is not None:
                        target.remove(change.value['value'])

        elif change.operation == "delete":
            if isinstance(parent, dict) and key in parent:
                del parent[key]
            elif isinstance(parent, list):
                del parent[int(key)]

    async def _persist(self):
        """Section in DB speichern"""
        if not self._app:
            return

        try:
            self._app.run_any(
                'DB', 'set',
                query=f"SharedSection::{self.id}",
                data=json.dumps(self.to_dict())
            )
        except Exception as e:
            if self._app:
                self._app.logger.error(f"[Shared] Error persisting section: {e}")

    async def _broadcast_change(self, change: SharedChange):
        """Änderung an alle Teilnehmer senden"""
        message = {
            'type': 'shared_change',
            'sectionId': self.id,
            'change': change.to_dict(),
        }

        for participant in self.participants.values():
            if participant.session and participant.session._send_callback:
                try:
                    await participant.session._send(json.dumps(message))
                except Exception as e:
                    if self._app:
                        self._app.logger.warning(
                            f"[Shared] Error broadcasting to {participant.id}: {e}"
                        )

    def _trigger_handlers(self, change: SharedChange):
        """Change-Handler aufrufen"""
        # Exakter Pfad
        if change.path in self._change_handlers:
            for handler in self._change_handlers[change.path]:
                try:
                    result = handler(change)
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                except Exception as e:
                    if self._app:
                        self._app.logger.error(f"[Shared] Handler error: {e}")

        # Wildcard-Handler "*"
        if "*" in self._change_handlers:
            for handler in self._change_handlers["*"]:
                try:
                    result = handler(change)
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                except Exception:
                    pass

    # =================== Change Handlers ===================

    def on_change(self, path: str, handler: Callable[[SharedChange], Any]):
        """
        Handler für Änderungen an einem Pfad registrieren.

        Args:
            path: Pfad oder "*" für alle Änderungen
            handler: Callback(change: SharedChange)
        """
        if path not in self._change_handlers:
            self._change_handlers[path] = []
        self._change_handlers[path].append(handler)

    def off_change(self, path: str, handler: Callable = None):
        """Handler entfernen"""
        if path in self._change_handlers:
            if handler:
                self._change_handlers[path].remove(handler)
            else:
                del self._change_handlers[path]

    # =================== Participant Management ===================

    def has_permission(self, participant_id: str, required: SharedPermission) -> bool:
        """Berechtigung prüfen"""
        if participant_id == self.owner_id:
            return True

        participant = self.participants.get(participant_id)
        if not participant:
            return False

        permission_levels = {
            SharedPermission.NONE: 0,
            SharedPermission.READ: 1,
            SharedPermission.WRITE: 2,
            SharedPermission.ADMIN: 3,
        }

        return permission_levels[participant.permission] >= permission_levels[required]

    async def add_participant(self, participant: SharedParticipant) -> bool:
        """Teilnehmer hinzufügen"""
        if len(self.participants) >= self.max_participants:
            return False

        if participant.type == ParticipantType.ANONYMOUS and not self.allow_anonymous:
            return False

        self.participants[participant.id] = participant
        await self._persist()

        # Benachrichtigen
        await self._broadcast_change(SharedChange(
            path="_participants",
            value={'action': 'join', 'participant': participant.to_dict()},
            operation="set"
        ))

        return True

    async def remove_participant(self, participant_id: str) -> bool:
        """Teilnehmer entfernen"""
        if participant_id not in self.participants:
            return False

        participant = self.participants[participant_id]
        del self.participants[participant_id]
        await self._persist()

        # Benachrichtigen
        await self._broadcast_change(SharedChange(
            path="_participants",
            value={'action': 'leave', 'participant': participant.to_dict()},
            operation="set"
        ))

        return True

    async def update_participant(self, participant_id: str,
                                 permission: SharedPermission = None) -> bool:
        """Teilnehmer aktualisieren"""
        if participant_id not in self.participants:
            return False

        participant = self.participants[participant_id]
        if permission:
            participant.permission = permission
        participant.last_seen = time.time()

        await self._persist()
        return True


# ============================================================================
# SHARED MANAGER
# ============================================================================


class SharedManager:
    """
    Manager für Shared Sections.
    Singleton pro App.
    """

    _instances: Dict[int, SharedManager] = {}

    def __init__(self, app: App):
        self.app = app
        self._sections: Dict[str, SharedSection] = {}
        self._user_sections: Dict[str, Set[str]] = {}  # user_id -> section_ids

    @classmethod
    def get_(cls, app: App) -> SharedManager:
        """Singleton-Instanz für App"""
        app_id = id(app)
        if app_id not in cls._instances:
            cls._instances[app_id] = cls(app)
        return cls._instances[app_id]

    async def create(
        self,
        request: RequestData,
        name: str,
        initial_data: Dict[str, Any] = None,
        max_participants: int = 100,
        allow_anonymous: bool = True,
        default_permission: SharedPermission = SharedPermission.WRITE,
        public: bool = False,
    ) -> SharedSection:
        """
        Neue Shared Section erstellen.

        Args:
            request: Request mit User-Info
            name: Name der Section
            initial_data: Initiale Daten
            max_participants: Max. Teilnehmer
            allow_anonymous: Anonyme erlauben
            default_permission: Standard-Berechtigung
            public: Öffentlich auffindbar

        Returns:
            SharedSection Instanz
        """
        from .user import MinuUser

        user = await MinuUser.from_request(self.app, request)

        section_id = f"shared-{uuid.uuid4().hex[:12]}"

        section = SharedSection(
            id=section_id,
            name=name,
            owner_id=user.uid,
            owner_type=ParticipantType.AUTHENTICATED if user.is_authenticated else ParticipantType.ANONYMOUS,
            data=initial_data or {},
            max_participants=max_participants,
            allow_anonymous=allow_anonymous,
            default_permission=default_permission,
            public=public,
            _app=self.app,
        )

        # Owner als ersten Teilnehmer
        owner_participant = SharedParticipant(
            id=user.uid,
            type=section.owner_type,
            name=user.name,
            permission=SharedPermission.ADMIN,
        )
        section.participants[user.uid] = owner_participant

        # Speichern
        self._sections[section_id] = section

        if user.uid not in self._user_sections:
            self._user_sections[user.uid] = set()
        self._user_sections[user.uid].add(section_id)

        await section._persist()

        self.app.logger.info(f"[Shared] Created section '{name}' ({section_id}) by {user.name}")

        return section

    async def get(self, section_id: str) -> SharedSection | None:
        """Section laden (aus Cache oder DB)"""
        # Cache
        if section_id in self._sections:
            return self._sections[section_id]

        # DB
        try:
            result = self.app.run_any(
                'DB', 'get',
                query=f"SharedSection::{section_id}",
                get_results=True
            )

            if result and not result.is_error() and result.get():
                data = result.get()
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
                if isinstance(data, bytes):
                    data = data.decode()
                if isinstance(data, str):
                    data = json.loads(data)

                section = SharedSection.from_dict(data)
                section._app = self.app
                self._sections[section_id] = section
                return section
        except Exception as e:
            self.app.logger.error(f"[Shared] Error loading section: {e}")

        return None

    async def join(
        self,
        section_id: str,
        request: RequestData,
        session: MinuSession = None,
    ) -> SharedSection | None:
        """
        Section beitreten.

        Args:
            section_id: ID der Section
            request: Request mit User-Info
            session: MinuSession für Live-Updates

        Returns:
            SharedSection oder None wenn nicht erlaubt
        """
        from .user import MinuUser

        section = await self.get(section_id)
        if not section:
            return None

        user = await MinuUser.from_request(self.app, request)

        # Prüfen ob bereits Teilnehmer
        if user.uid in section.participants:
            participant = section.participants[user.uid]
            participant.last_seen = time.time()
            participant.session = session
            return section

        # Prüfen ob beitreten erlaubt
        if not section.allow_anonymous and user.is_anonymous:
            return None

        if len(section.participants) >= section.max_participants:
            return None

        # Teilnehmer erstellen
        participant = SharedParticipant(
            id=user.uid,
            type=ParticipantType.AUTHENTICATED if user.is_authenticated else ParticipantType.ANONYMOUS,
            name=user.name,
            permission=section.default_permission,
            session=session,
        )

        await section.add_participant(participant)

        if user.uid not in self._user_sections:
            self._user_sections[user.uid] = set()
        self._user_sections[user.uid].add(section_id)

        self.app.logger.info(f"[Shared] {user.name} joined section '{section.name}'")

        return section

    async def leave(self, section_id: str, request: RequestData) -> bool:
        """Section verlassen"""
        from .user import MinuUser

        section = await self.get(section_id)
        if not section:
            return False

        user = await MinuUser.from_request(self.app, request)

        result = await section.remove_participant(user.uid)

        if user.uid in self._user_sections:
            self._user_sections[user.uid].discard(section_id)

        return result

    async def delete(self, section_id: str, request: RequestData) -> bool:
        """Section löschen (nur Owner)"""
        from .user import MinuUser

        section = await self.get(section_id)
        if not section:
            return False

        user = await MinuUser.from_request(self.app, request)

        if user.uid != section.owner_id:
            return False

        # Alle Teilnehmer benachrichtigen
        await section._broadcast_change(SharedChange(
            path="_section",
            value={'action': 'deleted'},
            operation="set"
        ))

        # Aus Cache entfernen
        if section_id in self._sections:
            del self._sections[section_id]

        # Aus DB löschen
        try:
            self.app.run_any('DB', 'delete', query=f"SharedSection::{section_id}")
        except Exception as e:
            self.app.logger.error(f"[Shared] Error deleting section: {e}")

        return True

    async def list_public(self, limit: int = 50) -> List[Dict]:
        """Öffentliche Sections auflisten"""
        public_sections = []

        for section in self._sections.values():
            if section.public:
                public_sections.append({
                    'id': section.id,
                    'name': section.name,
                    'participant_count': len(section.participants),
                    'max_participants': section.max_participants,
                    'created_at': section.created_at,
                })

        return public_sections[:limit]

    async def list_user_sections(self, request: RequestData) -> List[Dict]:
        """Sections eines Users auflisten"""
        from .user import MinuUser

        user = await MinuUser.from_request(self.app, request)

        user_section_ids = self._user_sections.get(user.uid, set())
        sections = []

        for section_id in user_section_ids:
            section = await self.get(section_id)
            if section:
                sections.append({
                    'id': section.id,
                    'name': section.name,
                    'is_owner': section.owner_id == user.uid,
                    'permission': section.participants.get(user.uid, SharedParticipant(
                        id="", type=ParticipantType.ANONYMOUS, name=""
                    )).permission.value,
                    'participant_count': len(section.participants),
                })

        return sections


# ============================================================================
# SHARED MIXIN FOR MINUVIEW
# ============================================================================


class SharedMixin:
    """
    Mixin für MinuView mit Shared-Funktionalität.

    Usage:
        class GameView(MinuView, SharedMixin):
            async def on_mount(self):
                self.shared = await self.join_shared('game_lobby')

                # Auf Änderungen reagieren
                self.shared.on_change('state', self.on_state_change)

            async def on_player_move(self, event):
                await self.shared.set('players.0.position', event['position'])
    """

    _shared_sections: Dict[str, SharedSection] = None
    _app: App | None = None
    request_data: RequestData | None = None
    _session: MinuSession | None = None

    @property
    def shared_manager(self) -> SharedManager:
        """SharedManager Instanz"""
        return SharedManager.get_(self._app)

    async def create_shared(
        self,
        name: str,
        initial_data: Dict[str, Any] = None,
        **kwargs
    ) -> SharedSection:
        """Neue Shared Section erstellen"""
        if self._shared_sections is None:
            self._shared_sections = {}

        section = await self.shared_manager.create(
            self.request_data,
            name,
            initial_data,
            **kwargs
        )

        self._shared_sections[section.id] = section
        return section

    async def join_shared(self, section_id: str) -> SharedSection | None:
        """Shared Section beitreten"""
        if self._shared_sections is None:
            self._shared_sections = {}

        section = await self.shared_manager.join(
            section_id,
            self.request_data,
            self._session
        )

        if section:
            self._shared_sections[section.id] = section

        return section

    async def leave_shared(self, section_id: str) -> bool:
        """Shared Section verlassen"""
        result = await self.shared_manager.leave(section_id, self.request_data)

        if result and self._shared_sections and section_id in self._shared_sections:
            del self._shared_sections[section_id]

        return result

    def get_shared(self, section_id: str) -> SharedSection | None:
        """Lokale Shared Section abrufen"""
        if self._shared_sections:
            return self._shared_sections.get(section_id)
        return None


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    'SharedSection',
    'SharedParticipant',
    'SharedChange',
    'SharedPermission',
    'ParticipantType',
    'SharedManager',
    'SharedMixin',
]
