"""
Google Calendar & Tasks Toolkit for Agent Sessions.

Session-isolated: Each agent session can have a different Google user.
Credentials are stored per session_id.
"""

import logging
import os
from typing import Any

try:
    from datetime import UTC, datetime, timedelta
except ImportError:
    from datetime import datetime, timedelta, timezone
    UTC = timezone.utc

from dateutil import parser as dateutil_parser
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)


class CalendarToolkit:
    """Session-aware Google Calendar + Tasks toolkit."""

    SCOPES = [
        'https://www.googleapis.com/auth/calendar',
        'https://www.googleapis.com/auth/tasks',
    ]

    def __init__(self, credentials_path: str = "/root/Toolboxv2/credentials.json",
                 token_dir: str = "token"):
        self.credentials_path = credentials_path
        self.token_dir = token_dir
        self._sessions: dict[str, dict[str, Any]] = {}

    def _get_session(self, session_id: str) -> dict[str, Any]:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "credentials": None,
                "calendar_service": None,
                "tasks_service": None,
            }
        return self._sessions[session_id]

    def _token_path(self, session_id: str) -> str:
        return os.path.join(self.token_dir, f"calendar_token_{session_id}.json")

    def _ensure_calendar(self, session_id: str):
        s = self._get_session(session_id)
        if s["calendar_service"] is None:
            raise RuntimeError("Calendar not authenticated. Call calendar_auth_start first.")
        return s["calendar_service"]

    def _ensure_tasks(self, session_id: str):
        s = self._get_session(session_id)
        if s["tasks_service"] is None:
            raise RuntimeError("Tasks not authenticated. Call calendar_auth_start first.")
        return s["tasks_service"]

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _auth_start(self, session_id: str) -> dict:
        from google_auth_oauthlib.flow import Flow
        flow = Flow.from_client_secrets_file(
            self.credentials_path, scopes=self.SCOPES,
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'
        )
        url, _ = flow.authorization_url(access_type='offline', prompt='consent')
        return {"success": True, "authorization_url": url}

    def _auth_complete(self, session_id: str, authorization_code: str) -> dict:
        from google_auth_oauthlib.flow import Flow
        flow = Flow.from_client_secrets_file(
            self.credentials_path, scopes=self.SCOPES,
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'
        )
        flow.fetch_token(code=authorization_code)
        s = self._get_session(session_id)
        s["credentials"] = flow.credentials
        self._save_credentials(session_id)
        self._init_services(session_id)
        return {"success": True, "message": "Calendar & Tasks authenticated."}

    def _save_credentials(self, session_id: str):
        s = self._get_session(session_id)
        if s["credentials"] is None:
            return
        os.makedirs(self.token_dir, exist_ok=True)
        with open(self._token_path(session_id), 'w') as f:
            f.write(s["credentials"].to_json())

    def _load_credentials(self, session_id: str) -> bool:
        try:
            creds = Credentials.from_authorized_user_file(self._token_path(session_id))
            s = self._get_session(session_id)
            s["credentials"] = creds
            self._init_services(session_id)
            return True
        except (FileNotFoundError, Exception):
            return False

    def _init_services(self, session_id: str):
        from googleapiclient.discovery import build
        s = self._get_session(session_id)
        s["calendar_service"] = build('calendar', 'v3', credentials=s["credentials"])
        s["tasks_service"] = build('tasks', 'v1', credentials=s["credentials"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_event_time(event: dict) -> str:
        start = event['start'].get('dateTime', event['start'].get('date'))
        end = event['end'].get('dateTime', event['end'].get('date'))
        try:
            start_dt = dateutil_parser.parse(start)
            end_dt = dateutil_parser.parse(end)
            if 'T' in start:
                return f"{start_dt.strftime('%a %d %b %H:%M')} - {end_dt.strftime('%H:%M')}"
            return f"{start_dt.strftime('%d %b %Y')} (All Day)"
        except Exception:
            return "Time not specified"

    @staticmethod
    def _parse_time(time_str: str, reference=None):
        if reference is None:
            reference = datetime.now()
        try:
            import dateparser
            parsed = dateparser.parse(time_str, settings={
                'PREFER_DATES_FROM': 'future',
                'RELATIVE_BASE': reference,
                'TIMEZONE': 'Europe/Berlin'
            })
            if parsed is None:
                parsed = dateutil_parser.parse(time_str, fuzzy=True, default=reference)
            return parsed.isoformat()
        except Exception:
            return reference.isoformat() if isinstance(reference, datetime) else reference

    # ------------------------------------------------------------------
    # Calendar Operations
    # ------------------------------------------------------------------

    def _list_events(self, session_id: str, time_min: str | None = None,
                     time_max: str | None = None, max_results: int = 10,
                     calendar_id: str = "primary") -> dict:
        service = self._ensure_calendar(session_id)
        try:
            if time_min is None:
                time_min = datetime.utcnow().isoformat() + 'Z'
            if time_max is None:
                time_max = (datetime.utcnow() + timedelta(days=7)).isoformat() + 'Z'
            result = service.events().list(
                calendarId=calendar_id, timeMin=time_min, timeMax=time_max,
                singleEvents=True, orderBy='startTime', maxResults=max_results
            ).execute()
            events = []
            for ev in result.get('items', []):
                events.append({
                    "id": ev["id"],
                    "summary": ev.get("summary", "No title"),
                    "time": self._format_event_time(ev),
                    "location": ev.get("location", ""),
                    "description": ev.get("description", "")[:500],
                    "start": ev["start"].get("dateTime", ev["start"].get("date")),
                    "end": ev["end"].get("dateTime", ev["end"].get("date")),
                })
            return {"success": True, "events": events, "count": len(events)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_event(self, session_id: str, event_id: str,
                   calendar_id: str = "primary") -> dict:
        service = self._ensure_calendar(session_id)
        try:
            ev = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
            return {
                "success": True,
                "id": ev["id"],
                "summary": ev.get("summary", "No title"),
                "time": self._format_event_time(ev),
                "location": ev.get("location", ""),
                "description": ev.get("description", ""),
                "start": ev["start"].get("dateTime", ev["start"].get("date")),
                "end": ev["end"].get("dateTime", ev["end"].get("date")),
                "attendees": [a.get("email") for a in ev.get("attendees", [])],
                "html_link": ev.get("htmlLink", ""),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_event(self, session_id: str, summary: str, start: str, end: str,
                      description: str = "", location: str = "",
                      calendar_id: str = "primary") -> dict:
        service = self._ensure_calendar(session_id)
        try:
            start_parsed = self._parse_time(start)
            end_parsed = self._parse_time(end)
            event_body = {
                "summary": summary,
                "start": {"dateTime": start_parsed, "timeZone": "Europe/Berlin"},
                "end": {"dateTime": end_parsed, "timeZone": "Europe/Berlin"},
            }
            if description:
                event_body["description"] = description
            if location:
                event_body["location"] = location
            created = service.events().insert(
                calendarId=calendar_id, body=event_body
            ).execute()
            return {
                "success": True,
                "event_id": created["id"],
                "html_link": created.get("htmlLink", ""),
                "summary": summary,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _update_event(self, session_id: str, event_id: str,
                      summary: str | None = None, start: str | None = None,
                      end: str | None = None, description: str | None = None,
                      location: str | None = None,
                      calendar_id: str = "primary") -> dict:
        service = self._ensure_calendar(session_id)
        try:
            ev = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
            if summary is not None:
                ev["summary"] = summary
            if start is not None:
                ev["start"] = {"dateTime": self._parse_time(start), "timeZone": "Europe/Berlin"}
            if end is not None:
                ev["end"] = {"dateTime": self._parse_time(end), "timeZone": "Europe/Berlin"}
            if description is not None:
                ev["description"] = description
            if location is not None:
                ev["location"] = location
            updated = service.events().update(
                calendarId=calendar_id, eventId=event_id, body=ev
            ).execute()
            return {"success": True, "event_id": updated["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _delete_event(self, session_id: str, event_id: str,
                      calendar_id: str = "primary") -> dict:
        service = self._ensure_calendar(session_id)
        try:
            service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
            return {"success": True, "event_id": event_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _find_free_slots(self, session_id: str, duration_minutes: int = 30,
                         hours_ahead: int = 24, max_slots: int = 5) -> dict:
        service = self._ensure_calendar(session_id)
        try:
            now = datetime.now(UTC)
            end_time = now + timedelta(hours=hours_ahead)
            freebusy = service.freebusy().query(body={
                "timeMin": now.isoformat(),
                "timeMax": end_time.isoformat(),
                "items": [{"id": "primary"}]
            }).execute()
            busy = freebusy['calendars']['primary']['busy']

            slots = []
            current = now
            while current < end_time and len(slots) < max_slots:
                slot_end = current + timedelta(minutes=duration_minutes)
                if slot_end > end_time:
                    break
                is_free = all(
                    slot_end <= dateutil_parser.parse(b['start']) or
                    current >= dateutil_parser.parse(b['end'])
                    for b in busy
                )
                if is_free:
                    slots.append({
                        "start": current.isoformat(),
                        "end": slot_end.isoformat(),
                        "start_readable": current.strftime('%a %d %b %H:%M'),
                        "end_readable": slot_end.strftime('%H:%M'),
                        "duration_minutes": duration_minutes,
                    })
                    current = slot_end
                else:
                    current += timedelta(minutes=15)
            return {"success": True, "slots": slots, "count": len(slots)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Google Tasks Operations
    # ------------------------------------------------------------------

    def _list_task_lists(self, session_id: str) -> dict:
        service = self._ensure_tasks(session_id)
        try:
            result = service.tasklists().list(maxResults=100).execute()
            lists = [{"id": tl["id"], "title": tl["title"]}
                     for tl in result.get("items", [])]
            return {"success": True, "task_lists": lists}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _list_tasks(self, session_id: str, tasklist_id: str = "@default",
                    show_completed: bool = False, max_results: int = 50) -> dict:
        service = self._ensure_tasks(session_id)
        try:
            result = service.tasks().list(
                tasklist=tasklist_id, maxResults=max_results,
                showCompleted=show_completed
            ).execute()
            tasks = []
            for t in result.get("items", []):
                tasks.append({
                    "id": t["id"],
                    "title": t.get("title", ""),
                    "notes": t.get("notes", ""),
                    "due": t.get("due", ""),
                    "status": t.get("status", ""),
                    "completed": t.get("completed", ""),
                })
            return {"success": True, "tasks": tasks, "count": len(tasks)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_task(self, session_id: str, title: str, notes: str = "",
                     due: str = "", tasklist_id: str = "@default") -> dict:
        service = self._ensure_tasks(session_id)
        try:
            body: dict[str, Any] = {"title": title}
            if notes:
                body["notes"] = notes
            if due:
                parsed_due = self._parse_time(due)
                body["due"] = parsed_due
            created = service.tasks().insert(tasklist=tasklist_id, body=body).execute()
            return {"success": True, "task_id": created["id"], "title": title}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _complete_task(self, session_id: str, task_id: str,
                       tasklist_id: str = "@default") -> dict:
        service = self._ensure_tasks(session_id)
        try:
            task = service.tasks().get(tasklist=tasklist_id, task=task_id).execute()
            task["status"] = "completed"
            updated = service.tasks().update(
                tasklist=tasklist_id, task=task_id, body=task
            ).execute()
            return {"success": True, "task_id": updated["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _delete_task(self, session_id: str, task_id: str,
                     tasklist_id: str = "@default") -> dict:
        service = self._ensure_tasks(session_id)
        try:
            service.tasks().delete(tasklist=tasklist_id, task=task_id).execute()
            return {"success": True, "task_id": task_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _update_task(self, session_id: str, task_id: str,
                     title: str | None = None, notes: str | None = None,
                     due: str | None = None,
                     tasklist_id: str = "@default") -> dict:
        service = self._ensure_tasks(session_id)
        try:
            task = service.tasks().get(tasklist=tasklist_id, task=task_id).execute()
            if title is not None:
                task["title"] = title
            if notes is not None:
                task["notes"] = notes
            if due is not None:
                task["due"] = self._parse_time(due)
            updated = service.tasks().update(
                tasklist=tasklist_id, task=task_id, body=task
            ).execute()
            return {"success": True, "task_id": updated["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Toolkit registration
    # ------------------------------------------------------------------

    def get_tools(self, session_id: str) -> list[dict]:
        """
        Return list of tool dicts for agent registration.
        All tools are bound to the given session_id.
        """
        self._load_credentials(session_id)

        # --- Auth ---
        def calendar_auth_start() -> dict:
            """Start Google Calendar & Tasks OAuth2 authentication. Returns an authorization URL."""
            return self._auth_start(session_id)

        def calendar_auth_complete(authorization_code: str) -> dict:
            """Complete OAuth2 flow with the authorization code.
            Args:
                authorization_code: Code received after user authorization
            """
            return self._auth_complete(session_id, authorization_code)

        # --- Calendar ---
        def calendar_list_events(time_min: str = "", time_max: str = "",
                                 max_results: int = 10) -> dict:
            """List calendar events in a time range. Defaults to next 7 days.
            Args:
                time_min: Start time (ISO format or natural language, default: now)
                time_max: End time (ISO format or natural language, default: +7 days)
                max_results: Max events to return
            """
            t_min = self._parse_time(time_min) if time_min else None
            t_max = self._parse_time(time_max) if time_max else None
            return self._list_events(session_id, t_min, t_max, max_results)

        def calendar_get_event(event_id: str) -> dict:
            """Get full details of a calendar event.
            Args:
                event_id: The Google Calendar event ID
            """
            return self._get_event(session_id, event_id)

        def calendar_create_event(summary: str, start: str, end: str,
                                   description: str = "",
                                   location: str = "") -> dict:
            """Create a new calendar event.
            Args:
                summary: Event title
                start: Start time (ISO or natural language e.g. 'tomorrow 2pm')
                end: End time (ISO or natural language e.g. 'tomorrow 3pm')
                description: Event description
                location: Event location
            """
            return self._create_event(session_id, summary, start, end,
                                       description, location)

        def calendar_update_event(event_id: str, summary: str | None = None,
                                   start: str | None = None, end: str | None = None,
                                   description: str | None = None,
                                   location: str | None = None) -> dict:
            """Update an existing calendar event. Only provided fields are changed.
            Args:
                event_id: The Google Calendar event ID
                summary: New title (or None to keep)
                start: New start time (or None to keep)
                end: New end time (or None to keep)
                description: New description (or None to keep)
                location: New location (or None to keep)
            """
            return self._update_event(session_id, event_id, summary, start,
                                       end, description, location)

        def calendar_delete_event(event_id: str) -> dict:
            """Delete a calendar event.
            Args:
                event_id: The Google Calendar event ID
            """
            return self._delete_event(session_id, event_id)

        def calendar_find_free_slots(duration_minutes: int = 30,
                                      hours_ahead: int = 24,
                                      max_slots: int = 5) -> dict:
            """Find available time slots in the calendar.
            Args:
                duration_minutes: Required slot duration (default 30)
                hours_ahead: How far ahead to search (default 24h)
                max_slots: Max slots to return (default 5)
            """
            return self._find_free_slots(session_id, duration_minutes,
                                          hours_ahead, max_slots)

        # --- Tasks ---
        def tasks_list_tasklists() -> dict:
            """List all Google Tasks task lists."""
            return self._list_task_lists(session_id)

        def tasks_list(tasklist_id: str = "@default",
                       show_completed: bool = False,
                       max_results: int = 50) -> dict:
            """List tasks in a task list.
            Args:
                tasklist_id: Task list ID (default: '@default')
                show_completed: Include completed tasks
                max_results: Max tasks to return
            """
            return self._list_tasks(session_id, tasklist_id,
                                     show_completed, max_results)

        def tasks_create(title: str, notes: str = "", due: str = "",
                         tasklist_id: str = "@default") -> dict:
            """Create a new task.
            Args:
                title: Task title
                notes: Task notes/description
                due: Due date (ISO or natural language)
                tasklist_id: Task list ID (default: '@default')
            """
            return self._create_task(session_id, title, notes, due, tasklist_id)

        def tasks_complete(task_id: str, tasklist_id: str = "@default") -> dict:
            """Mark a task as completed.
            Args:
                task_id: The task ID
                tasklist_id: Task list ID (default: '@default')
            """
            return self._complete_task(session_id, task_id, tasklist_id)

        def tasks_update(task_id: str, title: str | None = None,
                         notes: str | None = None, due: str | None = None,
                         tasklist_id: str = "@default") -> dict:
            """Update an existing task. Only provided fields are changed.
            Args:
                task_id: The task ID
                title: New title (or None to keep)
                notes: New notes (or None to keep)
                due: New due date (or None to keep)
                tasklist_id: Task list ID (default: '@default')
            """
            return self._update_task(session_id, task_id, title, notes,
                                      due, tasklist_id)

        def tasks_delete(task_id: str, tasklist_id: str = "@default") -> dict:
            """Delete a task.
            Args:
                task_id: The task ID
                tasklist_id: Task list ID (default: '@default')
            """
            return self._delete_task(session_id, task_id, tasklist_id)

        return [
            # Auth
            {
                "tool_func": calendar_auth_start,
                "name": "calendar_auth_start",
                "category": ["google", "calendar", "auth"],
            },
            {
                "tool_func": calendar_auth_complete,
                "name": "calendar_auth_complete",
                "category": ["google", "calendar", "auth"],
            },
            # Calendar
            {
                "tool_func": calendar_list_events,
                "name": "calendar_list_events",
                "category": ["google", "calendar", "read"],
            },
            {
                "tool_func": calendar_get_event,
                "name": "calendar_get_event",
                "category": ["google", "calendar", "read"],
            },
            {
                "tool_func": calendar_create_event,
                "name": "calendar_create_event",
                "category": ["google", "calendar", "write"],
            },
            {
                "tool_func": calendar_update_event,
                "name": "calendar_update_event",
                "category": ["google", "calendar", "write"],
            },
            {
                "tool_func": calendar_delete_event,
                "name": "calendar_delete_event",
                "category": ["google", "calendar", "write"],
            },
            {
                "tool_func": calendar_find_free_slots,
                "name": "calendar_find_free_slots",
                "category": ["google", "calendar", "read"],
            },
            # Tasks
            {
                "tool_func": tasks_list_tasklists,
                "name": "tasks_list_tasklists",
                "category": ["google", "tasks", "read"],
            },
            {
                "tool_func": tasks_list,
                "name": "tasks_list",
                "category": ["google", "tasks", "read"],
            },
            {
                "tool_func": tasks_create,
                "name": "tasks_create",
                "category": ["google", "tasks", "write"],
            },
            {
                "tool_func": tasks_complete,
                "name": "tasks_complete",
                "category": ["google", "tasks", "write"],
            },
            {
                "tool_func": tasks_update,
                "name": "tasks_update",
                "category": ["google", "tasks", "write"],
            },
            {
                "tool_func": tasks_delete,
                "name": "tasks_delete",
                "category": ["google", "tasks", "write"],
            },
        ]

    def cleanup_session(self, session_id: str):
        """Remove session state (call on session end)."""
        self._sessions.pop(session_id, None)
