"""
Google Calendar & Tasks Toolkit for Agent Sessions.

Session-isolated: Each agent session can have a different Google user.
Credentials persist on disk and survive restarts.
Agent-facing interface uses human-readable indices (#1, #2, ...)
instead of opaque Google IDs.
"""

import json
import logging
import os
import webbrowser
from typing import Any

try:
    from datetime import UTC, datetime, timedelta
except ImportError:
    from datetime import datetime, timedelta, timezone
    UTC = timezone.utc

from dateutil import parser as dateutil_parser
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Internal index map: translates #1, #2, ... → Google IDs
# ──────────────────────────────────────────────────────────────

class _IndexMap:
    """Bidirectional map: human index ↔ Google ID, per namespace."""

    def __init__(self):
        self._namespaces: dict[str, list[str]] = {}  # ns → [google_id, ...]

    def clear(self, namespace: str):
        self._namespaces[namespace] = []

    def register(self, namespace: str, google_id: str) -> int:
        """Register a Google ID, return 1-based index."""
        ns = self._namespaces.setdefault(namespace, [])
        if google_id not in ns:
            ns.append(google_id)
        return ns.index(google_id) + 1

    def resolve(self, namespace: str, ref: str | int) -> str:
        """Resolve '#3' or 3 or 'abc123' → Google ID. Raises ValueError."""
        ns = self._namespaces.get(namespace, [])
        # Accept "#3", "3", or int 3
        if isinstance(ref, int):
            idx = ref
        elif isinstance(ref, str) and ref.startswith("#"):
            idx = int(ref[1:])
        elif isinstance(ref, str) and ref.isdigit():
            idx = int(ref)
        else:
            # Assume raw Google ID passthrough (for programmatic callers)
            return ref
        if idx < 1 or idx > len(ns):
            raise ValueError(f"Index #{idx} not found. Valid range: #1–#{len(ns)}")
        return ns[idx - 1]


# ──────────────────────────────────────────────────────────────
# Markdown formatting helpers
# ──────────────────────────────────────────────────────────────

def _fmt_event(idx: int, ev: dict) -> str:
    """Format single event as compact MD line block."""
    start = ev["start"].get("dateTime", ev["start"].get("date", ""))
    end = ev["end"].get("dateTime", ev["end"].get("date", ""))
    try:
        s = dateutil_parser.parse(start)
        e = dateutil_parser.parse(end)
        if "T" in start:
            time_str = f"{s.strftime('%a %d %b %H:%M')}–{e.strftime('%H:%M')}"
        else:
            time_str = f"{s.strftime('%d %b %Y')} (ganztägig)"
    except Exception:
        time_str = "–"

    title = ev.get("summary", "Kein Titel")
    loc = ev.get("location", "")
    desc = (ev.get("description") or "")[:120]

    lines = [f"**#{idx}** {title}", f"  📅 {time_str}"]
    if loc:
        lines.append(f"  📍 {loc}")
    if desc:
        lines.append(f"  ℹ️ {desc}")
    return "\n".join(lines)


def _fmt_task(idx: int, t: dict) -> str:
    """Format single task as compact MD."""
    status = "✅" if t.get("status") == "completed" else "☐"
    title = t.get("title", "Kein Titel")
    due = t.get("due", "")
    notes = (t.get("notes") or "")[:100]

    line = f"{status} **#{idx}** {title}"
    parts = []
    if due:
        try:
            d = dateutil_parser.parse(due)
            parts.append(f"Fällig: {d.strftime('%d %b %Y')}")
        except Exception:
            pass
    if notes:
        parts.append(notes)
    if parts:
        line += f"\n  {' | '.join(parts)}"
    return line


def _fmt_slot(idx: int, slot_start: str, slot_end: str, duration: int) -> str:
    try:
        s = dateutil_parser.parse(slot_start)
        e = dateutil_parser.parse(slot_end)
        return f"**#{idx}** {s.strftime('%a %d %b %H:%M')}–{e.strftime('%H:%M')} ({duration} min)"
    except Exception:
        return f"**#{idx}** {slot_start}–{slot_end} ({duration} min)"


def _fmt_tasklist(idx: int, tl: dict) -> str:
    return f"**#{idx}** {tl['title']}"


# ──────────────────────────────────────────────────────────────
# Time parsing
# ──────────────────────────────────────────────────────────────

def _parse_time(time_str: str, reference: datetime | None = None, tzinfo: object = None) -> str:
    """Parse natural language or ISO time → UTC ISO string ending in Z."""
    if not time_str:
        return ""
    if reference is None:
        reference = datetime.now(UTC)
    try:
        import dateparser
        ref_naive = reference.replace(tzinfo=tzinfo) if reference.tzinfo else reference
        parsed = dateparser.parse(time_str, settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": ref_naive,
            "RETURN_AS_TIMEZONE_AWARE": True if tzinfo is None else False,
        })
        if parsed is None:
            parsed = dateutil_parser.parse(time_str, fuzzy=True, default=ref_naive)
        # Normalize to UTC
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(UTC)
        else:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        if isinstance(reference, datetime):
            ref = reference.astimezone(UTC) if reference.tzinfo else reference.replace(tzinfo=UTC)
            return ref.strftime("%Y-%m-%dT%H:%M:%SZ")
        return str(reference)


# ──────────────────────────────────────────────────────────────
# Main Toolkit
# ──────────────────────────────────────────────────────────────

class CalendarToolkit:
    """Session-aware Google Calendar + Tasks toolkit.

    Agent-facing tools return Markdown strings (not dicts).
    Items are referenced by #index (not Google IDs).
    Auth persists on disk.
    """

    SCOPES = [
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/tasks",
    ]

    def __init__(
        self,
        credentials_path: str = "/root/Toolboxv2/credentials.json",
        token_dir: str = "token",
    ):
        self.credentials_path = credentials_path
        self.token_dir = token_dir
        os.makedirs(self.token_dir, exist_ok=True)
        self._sessions: dict[str, dict[str, Any]] = {}

    # ── Session management ────────────────────────────────────

    def _get_session(self, sid: str) -> dict[str, Any]:
        if sid not in self._sessions:
            self._sessions[sid] = {
                "credentials": None,
                "calendar_service": None,
                "tasks_service": None,
                "index_map": _IndexMap(),
                "flow": None,
            }
        return self._sessions[sid]

    def _token_path(self, sid: str) -> str:
        return os.path.join(self.token_dir, f"cal_{sid}.json")

    def _save_credentials(self, sid: str):
        s = self._get_session(sid)
        creds = s["credentials"]
        if creds is None:
            return
        data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes or []),
        }
        with open(self._token_path(sid), "w") as f:
            json.dump(data, f)

    def _load_credentials(self, sid: str) -> bool:
        path = self._token_path(sid)
        if not os.path.exists(path):
            return False
        try:
            creds = Credentials.from_authorized_user_file(path, self.SCOPES)
            if creds and creds.expired and creds.refresh_token:
                from google.auth.transport.requests import Request
                creds.refresh(Request())
                # Save refreshed token
                s = self._get_session(sid)
                s["credentials"] = creds
                self._save_credentials(sid)
            elif creds and creds.valid:
                self._get_session(sid)["credentials"] = creds
            else:
                return False
            self._init_services(sid)
            return True
        except Exception as e:
            logger.warning("Failed to load credentials for %s: %s", sid, e)
            return False

    def _init_services(self, sid: str):
        from googleapiclient.discovery import build
        s = self._get_session(sid)
        creds = s["credentials"]
        s["calendar_service"] = build("calendar", "v3", credentials=creds)
        s["tasks_service"] = build("tasks", "v1", credentials=creds)

    def _ensure_auth(self, sid: str):
        s = self._get_session(sid)
        if s["calendar_service"] is None:
            if not self._load_credentials(sid):
                raise RuntimeError("Nicht angemeldet. Bitte `calendar_login` aufrufen.")

    def _imap(self, sid: str) -> _IndexMap:
        return self._get_session(sid)["index_map"]

    # ── Auth: CLI (fully internal) ────────────────────────────

    def _login_cli(self, sid: str) -> str:
        """Full CLI login: opens browser, reads code from stdin, returns status MD."""
        from google_auth_oauthlib.flow import InstalledAppFlow
        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_path, scopes=self.SCOPES,
            )
            creds = flow.run_local_server(port=0, open_browser=True)
            s = self._get_session(sid)
            s["credentials"] = creds
            self._save_credentials(sid)
            self._init_services(sid)
            return "✅ Google Calendar & Tasks angemeldet. Session ist persistent."
        except Exception as e:
            return f"❌ Login fehlgeschlagen: {e}"

    # ── Auth: 2-Part (for external/programmatic use) ──────────

    def _auth_url(self, sid: str) -> str:
        """Part 1: Generate OAuth URL. Returns MD with the link."""
        from google_auth_oauthlib.flow import Flow
        flow = Flow.from_client_secrets_file(
            self.credentials_path,
            scopes=self.SCOPES,
            redirect_uri="urn:ietf:wg:oauth:2.0:oob",
        )
        url, _ = flow.authorization_url(access_type="offline", prompt="consent")
        self._get_session(sid)["flow"] = flow
        return url

    def _auth_callback(self, sid: str, code: str) -> str:
        """Part 2: Exchange code for tokens. Returns status MD."""
        s = self._get_session(sid)
        flow = s.get("flow")
        if flow is None:
            return "❌ Kein Auth-Flow aktiv. Zuerst `calendar_auth_url` aufrufen."
        try:
            flow.fetch_token(code=code)
            s["credentials"] = flow.credentials
            s["flow"] = None
            self._save_credentials(sid)
            self._init_services(sid)
            return "✅ Google Calendar & Tasks angemeldet. Session ist persistent."
        except Exception as e:
            return f"❌ Auth fehlgeschlagen: {e}"

    # ── Calendar operations ───────────────────────────────────

    def _list_events(self, sid: str, time_min: str = "", time_max: str = "",
                     max_results: int = 10) -> str:
        self._ensure_auth(sid)
        service = self._get_session(sid)["calendar_service"]
        imap = self._imap(sid)
        imap.clear("events")

        now = datetime.now(UTC)
        t_min = _parse_time(time_min, now) if time_min else now.strftime("%Y-%m-%dT%H:%M:%SZ")
        t_max = (_parse_time(time_max, now) if time_max
                 else (now + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"))

        try:
            result = service.events().list(
                calendarId="primary", timeMin=t_min, timeMax=t_max,
                singleEvents=True, orderBy="startTime", maxResults=max_results,
            ).execute()
            items = result.get("items", [])
            if not items:
                return "Keine Events im Zeitraum."
            lines = [f"**Events** ({len(items)}):\n"]
            for ev in items:
                idx = imap.register("events", ev["id"])
                lines.append(_fmt_event(idx, ev))
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Fehler: {e}"

    def _get_event(self, sid: str, ref: str | int) -> str:
        self._ensure_auth(sid)
        service = self._get_session(sid)["calendar_service"]
        imap = self._imap(sid)
        try:
            eid = imap.resolve("events", ref)
            ev = service.events().get(calendarId="primary", eventId=eid).execute()
            idx = imap.register("events", ev["id"])

            start = ev["start"].get("dateTime", ev["start"].get("date", ""))
            end = ev["end"].get("dateTime", ev["end"].get("date", ""))
            try:
                s = dateutil_parser.parse(start)
                e = dateutil_parser.parse(end)
                time_str = f"{s.strftime('%a %d %b %Y %H:%M')}–{e.strftime('%H:%M')}"
            except Exception:
                time_str = f"{start} – {end}"

            parts = [
                f"**#{idx} {ev.get('summary', 'Kein Titel')}**",
                f"📅 {time_str}",
            ]
            if ev.get("location"):
                parts.append(f"📍 {ev['location']}")
            if ev.get("description"):
                parts.append(f"ℹ️ {ev['description'][:500]}")
            attendees = [a.get("email") for a in ev.get("attendees", []) if a.get("email")]
            if attendees:
                parts.append(f"👥 {', '.join(attendees)}")
            if ev.get("htmlLink"):
                parts.append(f"🔗 {ev['htmlLink']}")
            return "\n".join(parts)
        except ValueError as e:
            return f"❌ {e}"
        except Exception as e:
            return f"❌ Fehler: {e}"

    def _create_event(self, sid: str, summary: str, start: str, end: str,
                      description: str = "", location: str = "") -> str:
        self._ensure_auth(sid)
        service = self._get_session(sid)["calendar_service"]
        imap = self._imap(sid)
        try:
            # Detect if this is an all-day event (no time component)
            is_all_day = not any(char in start for char in [':', 'T', 't'])

            body: dict[str, Any] = {
                "summary": summary,
            }

            if is_all_day:
                # All-day event uses 'date' field (YYYY-MM-DD format)
                body["start"] = {"date": start[:10] if len(start) >= 10 else start}

            if description:
                body["description"] = description
            if location:
                body["location"] = location
            created = service.events().insert(calendarId="primary", body=body).execute()
            idx = imap.register("events", created["id"])
            return f"✅ Event erstellt: **#{idx}** {summary}"
        except Exception as e:
            return f"❌ Fehler: {e}"

    def _update_event(self, sid: str, ref: str | int,
                      summary: str | None = None, start: str | None = None,
                      end: str | None = None, description: str | None = None,
                      location: str | None = None) -> str:
        self._ensure_auth(sid)
        service = self._get_session(sid)["calendar_service"]
        imap = self._imap(sid)
        try:
            eid = imap.resolve("events", ref)
            ev = service.events().get(calendarId="primary", eventId=eid).execute()

            # Determine current event type to preserve it
            current_is_all_day = "date" in ev.get("start", {})

            if summary is not None:
                ev["summary"] = summary
            if start is not None:
                ev["start"] = {"dateTime": _parse_time(start)}
            if end is not None:
                ev["end"] = {"dateTime": _parse_time(end)}
            if description is not None:
                ev["description"] = description
            if location is not None:
                ev["location"] = location
            service.events().update(calendarId="primary", eventId=eid, body=ev).execute()
            idx = imap.register("events", eid)
            return f"✅ Event #{idx} aktualisiert."
        except ValueError as e:
            return f"❌ {e}"
        except Exception as e:
            return f"❌ Fehler: {e}"

    def _delete_event(self, sid: str, ref: str | int) -> str:
        self._ensure_auth(sid)
        service = self._get_session(sid)["calendar_service"]
        imap = self._imap(sid)
        try:
            eid = imap.resolve("events", ref)
            service.events().delete(calendarId="primary", eventId=eid).execute()
            return f"✅ Event gelöscht."
        except ValueError as e:
            return f"❌ {e}"
        except Exception as e:
            return f"❌ Fehler: {e}"

    def _find_free_slots(self, sid: str, duration_minutes: int = 30,
                         hours_ahead: int = 24, max_slots: int = 5) -> str:
        self._ensure_auth(sid)
        service = self._get_session(sid)["calendar_service"]
        imap = self._imap(sid)
        imap.clear("slots")
        try:
            now = datetime.now(UTC)
            end_time = now + timedelta(hours=hours_ahead)
            freebusy = service.freebusy().query(body={
                "timeMin": now.isoformat(),
                "timeMax": end_time.isoformat(),
                "items": [{"id": "primary"}],
            }).execute()
            busy = freebusy["calendars"]["primary"]["busy"]

            slots = []
            current = now
            while current < end_time and len(slots) < max_slots:
                slot_end = current + timedelta(minutes=duration_minutes)
                if slot_end > end_time:
                    break
                is_free = all(
                    slot_end <= dateutil_parser.parse(b["start"])
                    or current >= dateutil_parser.parse(b["end"])
                    for b in busy
                )
                if is_free:
                    # Register a synthetic ID for slot booking
                    slot_id = f"slot_{current.isoformat()}"
                    idx = imap.register("slots", slot_id)
                    slots.append(_fmt_slot(idx, current.isoformat(), slot_end.isoformat(), duration_minutes))
                    current = slot_end
                else:
                    current += timedelta(minutes=15)

            if not slots:
                return "Keine freien Slots gefunden."
            return f"**Freie Slots** ({len(slots)}):\n" + "\n".join(slots)
        except Exception as e:
            return f"❌ Fehler: {e}"

    # ── Tasks operations ──────────────────────────────────────

    def _list_task_lists(self, sid: str) -> str:
        self._ensure_auth(sid)
        service = self._get_session(sid)["tasks_service"]
        imap = self._imap(sid)
        imap.clear("tasklists")
        try:
            result = service.tasklists().list(maxResults=100).execute()
            items = result.get("items", [])
            if not items:
                return "Keine Task-Listen vorhanden."
            lines = [f"**Task-Listen** ({len(items)}):\n"]
            for tl in items:
                idx = imap.register("tasklists", tl["id"])
                lines.append(_fmt_tasklist(idx, tl))
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Fehler: {e}"

    def _resolve_tasklist(self, sid: str, ref: str | int | None) -> str:
        """Resolve tasklist ref to Google ID. None → '@default'."""
        if ref is None:
            return "@default"
        try:
            return self._imap(sid).resolve("tasklists", ref)
        except ValueError:
            # If it doesn't resolve, assume it's a raw ID or "@default"
            return str(ref) if ref else "@default"

    def _list_tasks(self, sid: str, tasklist_ref: str | int | None = None,
                    show_completed: bool = False, max_results: int = 50) -> str:
        self._ensure_auth(sid)
        service = self._get_session(sid)["tasks_service"]
        imap = self._imap(sid)
        imap.clear("tasks")
        tl_id = self._resolve_tasklist(sid, tasklist_ref)
        try:
            result = service.tasks().list(
                tasklist=tl_id, maxResults=max_results,
                showCompleted=show_completed,
            ).execute()
            items = result.get("items", [])
            if not items:
                return "Keine Tasks vorhanden."
            lines = [f"**Tasks** ({len(items)}):\n"]
            for t in items:
                idx = imap.register("tasks", t["id"])
                lines.append(_fmt_task(idx, t))
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Fehler: {e}"

    def _create_task(self, sid: str, title: str, notes: str = "",
                     due: str = "", tasklist_ref: str | int | None = None) -> str:
        self._ensure_auth(sid)
        service = self._get_session(sid)["tasks_service"]
        imap = self._imap(sid)
        tl_id = self._resolve_tasklist(sid, tasklist_ref)
        try:
            body: dict[str, Any] = {"title": title}
            if notes:
                body["notes"] = notes
            if due:
                body["due"] = _parse_time(due)
            created = service.tasks().insert(tasklist=tl_id, body=body).execute()
            idx = imap.register("tasks", created["id"])
            return f"✅ Task erstellt: **#{idx}** {title}"
        except Exception as e:
            return f"❌ Fehler: {e}"

    def _complete_task(self, sid: str, ref: str | int,
                       tasklist_ref: str | int | None = None) -> str:
        self._ensure_auth(sid)
        service = self._get_session(sid)["tasks_service"]
        imap = self._imap(sid)
        tl_id = self._resolve_tasklist(sid, tasklist_ref)
        try:
            tid = imap.resolve("tasks", ref)
            task = service.tasks().get(tasklist=tl_id, task=tid).execute()
            task["status"] = "completed"
            service.tasks().update(tasklist=tl_id, task=tid, body=task).execute()
            return f"✅ Task #{ref} abgeschlossen."
        except ValueError as e:
            return f"❌ {e}"
        except Exception as e:
            return f"❌ Fehler: {e}"

    def _update_task(self, sid: str, ref: str | int,
                     title: str | None = None, notes: str | None = None,
                     due: str | None = None,
                     tasklist_ref: str | int | None = None) -> str:
        self._ensure_auth(sid)
        service = self._get_session(sid)["tasks_service"]
        imap = self._imap(sid)
        tl_id = self._resolve_tasklist(sid, tasklist_ref)
        try:
            tid = imap.resolve("tasks", ref)
            task = service.tasks().get(tasklist=tl_id, task=tid).execute()
            if title is not None:
                task["title"] = title
            if notes is not None:
                task["notes"] = notes
            if due is not None:
                task["due"] = _parse_time(due)
            service.tasks().update(tasklist=tl_id, task=tid, body=task).execute()
            return f"✅ Task #{ref} aktualisiert."
        except ValueError as e:
            return f"❌ {e}"
        except Exception as e:
            return f"❌ Fehler: {e}"

    def _delete_task(self, sid: str, ref: str | int,
                     tasklist_ref: str | int | None = None) -> str:
        self._ensure_auth(sid)
        service = self._get_session(sid)["tasks_service"]
        imap = self._imap(sid)
        tl_id = self._resolve_tasklist(sid, tasklist_ref)
        try:
            tid = imap.resolve("tasks", ref)
            service.tasks().delete(tasklist=tl_id, task=tid).execute()
            return f"✅ Task gelöscht."
        except ValueError as e:
            return f"❌ {e}"
        except Exception as e:
            return f"❌ Fehler: {e}"

    # ──────────────────────────────────────────────────────────
    # Tool registration (agent interface)
    # ──────────────────────────────────────────────────────────

    def get_tools(self, session_id: str) -> list[dict]:
        """Return tool dicts for agent registration, bound to session_id."""
        # Try to auto-load persistent credentials
        self._load_credentials(session_id)

        # ── Auth tools ──

        def calendar_login() -> str:
            """Login to Google Calendar & Tasks via local browser (CLI).
            Opens browser automatically, no input needed from agent."""
            return self._login_cli(session_id)

        def calendar_auth_url() -> str:
            """Get OAuth2 authorization URL for external login (Discord, Web, etc.).
            Returns the URL the user must visit. After authorization,
            call calendar_auth_callback with the code."""
            return self._auth_url(session_id)

        def calendar_auth_callback(code: str) -> str:
            """Complete external OAuth2 login with the authorization code.
            Args:
                code: Authorization code from the OAuth redirect
            """
            return self._auth_callback(session_id, code)

        # ── Calendar tools ──

        def calendar_list_events(time_min: str = "", time_max: str = "",
                                 max_results: int = 10) -> str:
            """List calendar events. Returns numbered list (#1, #2, ...).
            Args:
                time_min: Start (ISO or natural: 'today', 'tomorrow', 'next monday')
                time_max: End (ISO or natural, default: +7 days)
                max_results: Max events (default 10)
            """
            return self._list_events(session_id, time_min, time_max, max_results)

        def calendar_get_event(ref: str) -> str:
            """Get full event details by index (#1, #2, ...) from last listing.
            Args:
                ref: Event reference, e.g. '#1' or '3'
            """
            return self._get_event(session_id, ref)

        def calendar_create_event(summary: str, start: str, end: str,
                                  description: str = "", location: str = "") -> str:
            """Create a new calendar event.
            Args:
                summary: Event title
                start: Start time (natural: 'tomorrow 2pm' or ISO)
                end: End time (natural: 'tomorrow 3pm' or ISO)
                description: Optional description
                location: Optional location
            """
            return self._create_event(session_id, summary, start, end,
                                      description, location)

        def calendar_update_event(ref: str, summary: str | None = None,
                                  start: str | None = None, end: str | None = None,
                                  description: str | None = None,
                                  location: str | None = None) -> str:
            """Update event by index. Only provided fields change.
            Args:
                ref: Event reference (#1, #2, ...) from last listing
                summary: New title (or None)
                start: New start (or None)
                end: New end (or None)
                description: New description (or None)
                location: New location (or None)
            """
            return self._update_event(session_id, ref, summary, start,
                                      end, description, location)

        def calendar_delete_event(ref: str) -> str:
            """Delete event by index.
            Args:
                ref: Event reference (#1, #2, ...) from last listing
            """
            return self._delete_event(session_id, ref)

        def calendar_find_free_slots(duration_minutes: int = 30,
                                     hours_ahead: int = 24,
                                     max_slots: int = 5) -> str:
            """Find free time slots in calendar.
            Args:
                duration_minutes: Slot length (default 30)
                hours_ahead: Search window (default 24h)
                max_slots: Max results (default 5)
            """
            return self._find_free_slots(session_id, duration_minutes,
                                         hours_ahead, max_slots)

        # ── Tasks tools ──

        def tasks_list_tasklists() -> str:
            """List all task lists. Returns numbered list (#1, #2, ...)."""
            return self._list_task_lists(session_id)

        def tasks_list(tasklist_ref: str = "", show_completed: bool = False,
                       max_results: int = 50) -> str:
            """List tasks. Returns numbered list.
            Args:
                tasklist_ref: Task list (#1 from listing, or empty for default)
                show_completed: Include completed tasks
                max_results: Max results
            """
            ref = tasklist_ref if tasklist_ref else None
            return self._list_tasks(session_id, ref, show_completed, max_results)

        def tasks_create(title: str, notes: str = "", due: str = "",
                         tasklist_ref: str = "") -> str:
            """Create a new task.
            Args:
                title: Task title
                notes: Optional notes
                due: Due date (natural or ISO)
                tasklist_ref: Task list ref (or empty for default)
            """
            ref = tasklist_ref if tasklist_ref else None
            return self._create_task(session_id, title, notes, due, ref)

        def tasks_complete(ref: str, tasklist_ref: str = "") -> str:
            """Mark task as done by index.
            Args:
                ref: Task reference (#1, #2, ...) from last listing
                tasklist_ref: Task list ref (or empty for default)
            """
            tl = tasklist_ref if tasklist_ref else None
            return self._complete_task(session_id, ref, tl)

        def tasks_update(ref: str, title: str | None = None,
                         notes: str | None = None, due: str | None = None,
                         tasklist_ref: str = "") -> str:
            """Update task by index. Only provided fields change.
            Args:
                ref: Task reference (#1, #2, ...)
                title: New title (or None)
                notes: New notes (or None)
                due: New due date (or None)
                tasklist_ref: Task list ref (or empty for default)
            """
            tl = tasklist_ref if tasklist_ref else None
            return self._update_task(session_id, ref, title, notes, due, tl)

        def tasks_delete(ref: str, tasklist_ref: str = "") -> str:
            """Delete task by index.
            Args:
                ref: Task reference (#1, #2, ...)
                tasklist_ref: Task list ref (or empty for default)
            """
            tl = tasklist_ref if tasklist_ref else None
            return self._delete_task(session_id, ref, tl)

        return [
            # Auth
            {"tool_func": calendar_login, "name": "calendar_login",
             "category": ["google", "calendar", "auth"]},
            {"tool_func": calendar_auth_url, "name": "calendar_auth_url",
             "category": ["google", "calendar", "auth"]},
            {"tool_func": calendar_auth_callback, "name": "calendar_auth_callback",
             "category": ["google", "calendar", "auth"]},
            # Calendar
            {"tool_func": calendar_list_events, "name": "calendar_list_events",
             "category": ["google", "calendar", "read"]},
            {"tool_func": calendar_get_event, "name": "calendar_get_event",
             "category": ["google", "calendar", "read"]},
            {"tool_func": calendar_create_event, "name": "calendar_create_event",
             "category": ["google", "calendar", "write"]},
            {"tool_func": calendar_update_event, "name": "calendar_update_event",
             "category": ["google", "calendar", "write"]},
            {"tool_func": calendar_delete_event, "name": "calendar_delete_event",
             "category": ["google", "calendar", "write"]},
            {"tool_func": calendar_find_free_slots, "name": "calendar_find_free_slots",
             "category": ["google", "calendar", "read"]},
            # Tasks
            {"tool_func": tasks_list_tasklists, "name": "tasks_list_tasklists",
             "category": ["google", "tasks", "read"]},
            {"tool_func": tasks_list, "name": "tasks_list",
             "category": ["google", "tasks", "read"]},
            {"tool_func": tasks_create, "name": "tasks_create",
             "category": ["google", "tasks", "write"]},
            {"tool_func": tasks_complete, "name": "tasks_complete",
             "category": ["google", "tasks", "write"]},
            {"tool_func": tasks_update, "name": "tasks_update",
             "category": ["google", "tasks", "write"]},
            {"tool_func": tasks_delete, "name": "tasks_delete",
             "category": ["google", "tasks", "write"]},
        ]

    def cleanup_session(self, session_id: str):
        """Remove in-memory session state. Credentials remain on disk."""
        self._sessions.pop(session_id, None)
