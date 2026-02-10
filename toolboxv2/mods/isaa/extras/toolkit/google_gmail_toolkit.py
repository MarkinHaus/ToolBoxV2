"""
Google Gmail Toolkit for Agent Sessions.

Design principles (matching Calendar/Tasks toolkit):
  - All tool responses are Markdown strings, not dicts/JSON
  - Emails referenced by #Index via internal IndexMap
  - Persistent OAuth2 credentials (survive shutdown)
  - CLI-based login (agent calls one function, never sees URL/code)
  - 2-part auth for external (Discord, web, Telegram) programmatic login
  - Clean session isolation
  - Token-efficient, human-readable output
"""

import base64
import logging
import os
import webbrowser
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


# ── IndexMap ─────────────────────────────────────────────────────────

class IndexMap:
    """Bidirectional #Index ↔ Google ID mapping."""

    def __init__(self):
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._next: int = 1

    def add(self, google_id: str) -> int:
        if google_id in self._id_to_idx:
            return self._id_to_idx[google_id]
        idx = self._next
        self._next += 1
        self._id_to_idx[google_id] = idx
        self._idx_to_id[idx] = google_id
        return idx

    def resolve(self, index: int) -> str | None:
        return self._idx_to_id.get(index)

    def clear(self):
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._next = 1


# ── Session State ────────────────────────────────────────────────────

class _Session:
    __slots__ = ("credentials", "service", "emails", "pending_flow")

    def __init__(self):
        self.credentials: Credentials | None = None
        self.service: Any = None
        self.emails: IndexMap = IndexMap()
        self.pending_flow: Any = None


# ── GmailToolkit ─────────────────────────────────────────────────────

class GmailToolkit:
    """Session-aware Gmail toolkit. Markdown-first, index-based."""

    def __init__(
        self,
        credentials_path: str = "/root/Toolboxv2/credentials.json",
        token_dir: str = "token",
    ):
        self.credentials_path = credentials_path
        self.token_dir = token_dir
        self._sessions: dict[str, _Session] = {}

    # ── internals ────────────────────────────────────────────────────

    def _s(self, sid: str) -> _Session:
        if sid not in self._sessions:
            self._sessions[sid] = _Session()
        return self._sessions[sid]

    def _token_path(self, sid: str) -> str:
        return os.path.join(self.token_dir, f"gmail_token_{sid}.json")

    def _save_credentials(self, sid: str):
        s = self._s(sid)
        if s.credentials is None:
            return
        os.makedirs(self.token_dir, exist_ok=True)
        with open(self._token_path(sid), "w") as f:
            f.write(s.credentials.to_json())

    def _load_credentials(self, sid: str) -> bool:
        path = self._token_path(sid)
        if not os.path.exists(path):
            return False
        try:
            creds = Credentials.from_authorized_user_file(path, SCOPES)
            if creds and creds.expired and creds.refresh_token:
                from google.auth.transport.requests import Request
                creds.refresh(Request())
                s = self._s(sid)
                s.credentials = creds
                self._save_credentials(sid)
            else:
                self._s(sid).credentials = creds
            self._init_service(sid)
            return True
        except Exception as e:
            logger.warning("Failed to load Gmail credentials for %s: %s", sid, e)
            return False

    def _init_service(self, sid: str):
        from googleapiclient.discovery import build
        s = self._s(sid)
        s.service = build("gmail", "v1", credentials=s.credentials)

    def _require_service(self, sid: str):
        s = self._s(sid)
        if s.service is None:
            raise RuntimeError("Not authenticated. Use `gmail_login` first.")
        return s.service

    def _create_flow(self):
        from google_auth_oauthlib.flow import Flow
        return Flow.from_client_secrets_file(
            self.credentials_path,
            scopes=SCOPES,
            redirect_uri="urn:ietf:wg:oauth:2.0:oob",
        )

    # ── Auth: CLI (single call, blocking) ────────────────────────────

    def _login_cli(self, sid: str) -> str:
        """Full CLI login: opens browser, waits for code, completes auth."""
        if self._load_credentials(sid):
            return "✅ Gmail already authenticated (persistent session)."

        flow = self._create_flow()
        auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")

        try:
            webbrowser.open(auth_url)
            print(f"\n🔗 Browser opened for Gmail login.\n   If not, visit:\n   {auth_url}\n")
        except Exception:
            print(f"\n🔗 Open this URL to authorize Gmail:\n   {auth_url}\n")

        code = input("📋 Paste authorization code: ").strip()
        if not code:
            return "❌ No authorization code provided."

        flow.fetch_token(code=code)
        s = self._s(sid)
        s.credentials = flow.credentials
        self._save_credentials(sid)
        self._init_service(sid)
        return "✅ Gmail authenticated and session saved."

    # ── Auth: External 2-part (Discord, Web, Telegram) ───────────────

    def _login_start(self, sid: str) -> str:
        """Part 1: Generate auth URL. Returns URL string for external delivery."""
        if self._load_credentials(sid):
            return "ALREADY_AUTHENTICATED"

        flow = self._create_flow()
        auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")
        self._s(sid).pending_flow = flow
        return auth_url

    def _login_complete(self, sid: str, code: str) -> str:
        """Part 2: Complete auth with code. Returns status markdown."""
        s = self._s(sid)
        flow = s.pending_flow
        if flow is None:
            return "❌ No pending login. Call `_login_start` first."

        try:
            flow.fetch_token(code=code)
            s.credentials = flow.credentials
            s.pending_flow = None
            self._save_credentials(sid)
            self._init_service(sid)
            return "✅ Gmail authenticated and session saved."
        except Exception as e:
            return f"❌ Auth failed: {e}"

    # ── Email Operations ─────────────────────────────────────────────

    def _list_emails(self, sid: str, query: str = "", max_results: int = 10) -> str:
        service = self._require_service(sid)
        s = self._s(sid)
        try:
            kwargs = {"userId": "me", "maxResults": max_results, "labelIds": ["INBOX"]}
            if query:
                kwargs["q"] = query

            results = service.users().messages().list(**kwargs).execute()
            messages = results.get("messages", [])[:max_results]

            if not messages:
                return "📭 No emails found."

            lines = [f"📬 **Inbox** ({len(messages)} emails)\n"]
            for msg in messages:
                data = service.users().messages().get(
                    userId="me", id=msg["id"], format="metadata"
                ).execute()
                headers = {h["name"]: h["value"] for h in data["payload"]["headers"]}
                idx = s.emails.add(msg["id"])
                unread = "🔵" if "UNREAD" in data.get("labelIds", []) else "  "
                lines.append(
                    f"{unread} **#{idx}** {headers.get('Subject', '(no subject)')}\n"
                    f"   From: {headers.get('From', '?')} · {headers.get('Date', '?')}\n"
                    f"   {data.get('snippet', '')[:100]}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"❌ List failed: {e}"

    def _read_email(self, sid: str, index: int) -> str:
        service = self._require_service(sid)
        s = self._s(sid)
        email_id = s.emails.resolve(index)
        if not email_id:
            return f"❌ Unknown email #{index}. List emails first."

        try:
            data = service.users().messages().get(
                userId="me", id=email_id, format="full"
            ).execute()
            headers = {h["name"]: h["value"] for h in data["payload"]["headers"]}

            body = self._extract_body(data)

            labels = data.get("labelIds", [])
            label_str = ", ".join(labels) if labels else "none"

            return (
                f"📧 **Email #{index}**\n\n"
                f"**From:** {headers.get('From', '?')}\n"
                f"**To:** {headers.get('To', '?')}\n"
                f"**Subject:** {headers.get('Subject', '(no subject)')}\n"
                f"**Date:** {headers.get('Date', '?')}\n"
                f"**Labels:** {label_str}\n\n"
                f"---\n\n{body}"
            )
        except Exception as e:
            return f"❌ Read failed: {e}"

    @staticmethod
    def _extract_body(data: dict) -> str:
        """Extract plain-text body from Gmail message payload."""
        parts = data.get("payload", {}).get("parts", [])
        if parts:
            for part in parts:
                if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                    return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
            # fallback: first part with data
            for part in parts:
                if part.get("body", {}).get("data"):
                    return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
        # single-part message
        raw = data.get("payload", {}).get("body", {}).get("data")
        if raw:
            return base64.urlsafe_b64decode(raw).decode("utf-8")
        return "(no body)"

    def _send_email(
        self, sid: str, to: str, subject: str, body: str,
        cc: str = "", bcc: str = "",
    ) -> str:
        service = self._require_service(sid)
        try:
            message = MIMEText(body)
            message["to"] = to
            message["subject"] = subject
            if cc:
                message["cc"] = cc
            if bcc:
                message["bcc"] = bcc
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            result = service.users().messages().send(
                userId="me", body={"raw": raw}
            ).execute()
            idx = self._s(sid).emails.add(result["id"])
            return f"✅ Email sent → **#{idx}**\n   To: {to}\n   Subject: {subject}"
        except Exception as e:
            return f"❌ Send failed: {e}"

    def _send_with_attachment(
        self, sid: str, to: str, subject: str, body: str,
        attachment_path: str, cc: str = "", bcc: str = "",
    ) -> str:
        service = self._require_service(sid)
        try:
            msg = MIMEMultipart()
            msg["to"] = to
            msg["subject"] = subject
            if cc:
                msg["cc"] = cc
            if bcc:
                msg["bcc"] = bcc
            msg.attach(MIMEText(body))

            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            filename = os.path.basename(attachment_path)
            part.add_header("Content-Disposition", f"attachment; filename={filename}")
            msg.attach(part)

            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            result = service.users().messages().send(
                userId="me", body={"raw": raw}
            ).execute()
            idx = self._s(sid).emails.add(result["id"])
            return (
                f"✅ Email with attachment sent → **#{idx}**\n"
                f"   To: {to}\n   Subject: {subject}\n   📎 {filename}"
            )
        except Exception as e:
            return f"❌ Send failed: {e}"

    def _search_emails(self, sid: str, query: str, max_results: int = 10) -> str:
        return self._list_emails(sid, query=query, max_results=max_results)

    def _modify_labels(
        self, sid: str, index: int,
        add_labels: str = "", remove_labels: str = "",
    ) -> str:
        service = self._require_service(sid)
        s = self._s(sid)
        email_id = s.emails.resolve(index)
        if not email_id:
            return f"❌ Unknown email #{index}. List emails first."

        try:
            body: dict[str, list[str]] = {}
            if add_labels:
                body["addLabelIds"] = [l.strip() for l in add_labels.split(",")]
            if remove_labels:
                body["removeLabelIds"] = [l.strip() for l in remove_labels.split(",")]
            service.users().messages().modify(
                userId="me", id=email_id, body=body
            ).execute()

            changes = []
            if add_labels:
                changes.append(f"+{add_labels}")
            if remove_labels:
                changes.append(f"-{remove_labels}")
            return f"✅ Email #{index} labels updated: {', '.join(changes)}"
        except Exception as e:
            return f"❌ Label update failed: {e}"

    def _mark_read(self, sid: str, index: int) -> str:
        return self._modify_labels(sid, index, remove_labels="UNREAD")

    def _mark_unread(self, sid: str, index: int) -> str:
        return self._modify_labels(sid, index, add_labels="UNREAD")

    def _archive(self, sid: str, index: int) -> str:
        return self._modify_labels(sid, index, remove_labels="INBOX")

    def _trash_email(self, sid: str, index: int) -> str:
        service = self._require_service(sid)
        s = self._s(sid)
        email_id = s.emails.resolve(index)
        if not email_id:
            return f"❌ Unknown email #{index}. List emails first."

        try:
            service.users().messages().trash(userId="me", id=email_id).execute()
            return f"🗑️ Email #{index} moved to trash."
        except Exception as e:
            return f"❌ Trash failed: {e}"

    def _reply(self, sid: str, index: int, body: str) -> str:
        """Reply to an email by #index. Preserves thread and subject."""
        service = self._require_service(sid)
        s = self._s(sid)
        email_id = s.emails.resolve(index)
        if not email_id:
            return f"❌ Unknown email #{index}. List emails first."

        try:
            original = service.users().messages().get(
                userId="me", id=email_id, format="metadata"
            ).execute()
            headers = {h["name"]: h["value"] for h in original["payload"]["headers"]}

            reply_to = headers.get("Reply-To", headers.get("From", ""))
            subject = headers.get("Subject", "")
            if not subject.lower().startswith("re:"):
                subject = f"Re: {subject}"
            thread_id = original.get("threadId")
            message_id = headers.get("Message-ID", "")

            msg = MIMEText(body)
            msg["to"] = reply_to
            msg["subject"] = subject
            if message_id:
                msg["In-Reply-To"] = message_id
                msg["References"] = message_id

            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            send_body: dict[str, Any] = {"raw": raw}
            if thread_id:
                send_body["threadId"] = thread_id

            result = service.users().messages().send(
                userId="me", body=send_body
            ).execute()
            idx = s.emails.add(result["id"])
            return f"✅ Reply sent → **#{idx}**\n   To: {reply_to}\n   Subject: {subject}"
        except Exception as e:
            return f"❌ Reply failed: {e}"

    # ── Tool Registration ────────────────────────────────────────────

    def get_tools(self, session_id: str) -> list[dict]:
        """
        Return agent-facing tools for this session.
        Auto-loads persistent credentials.
        All tools return markdown strings.
        """
        self._load_credentials(session_id)
        sid = session_id

        def gmail_login() -> str:
            """Login to Gmail via CLI. Opens browser, waits for auth code."""
            return self._login_cli(sid)

        def gmail_list(query: str = "", max_results: int = 10) -> str:
            """List inbox emails. Returns indexed list.
            Args:
                query: Gmail search query (e.g. 'is:unread', 'from:user@example.com')
                max_results: Max emails to return (default 10)
            """
            return self._list_emails(sid, query, max_results)

        def gmail_read(index: int) -> str:
            """Read full email content by #index.
            Args:
                index: Email index from gmail_list (e.g. 1 for #1)
            """
            return self._read_email(sid, index)

        def gmail_send(to: str, subject: str, body: str,
                       cc: str = "", bcc: str = "") -> str:
            """Send a plain-text email.
            Args:
                to: Recipient email address
                subject: Email subject
                body: Email body text
                cc: CC recipients (comma-separated)
                bcc: BCC recipients (comma-separated)
            """
            return self._send_email(sid, to, subject, body, cc, bcc)

        def gmail_send_with_attachment(
            to: str, subject: str, body: str,
            attachment_path: str, cc: str = "", bcc: str = "",
        ) -> str:
            """Send an email with file attachment.
            Args:
                to: Recipient email address
                subject: Email subject
                body: Email body text
                attachment_path: Local file path to attach
                cc: CC recipients (comma-separated)
                bcc: BCC recipients (comma-separated)
            """
            return self._send_with_attachment(sid, to, subject, body, attachment_path, cc, bcc)

        def gmail_search(query: str, max_results: int = 10) -> str:
            """Search emails. Returns indexed list.
            Args:
                query: Gmail search query (from:, to:, subject:, has:attachment, is:unread, newer_than:7d, etc.)
                max_results: Max results (default 10)
            """
            return self._search_emails(sid, query, max_results)

        def gmail_mark_read(index: int) -> str:
            """Mark email as read.
            Args:
                index: Email index (e.g. 1 for #1)
            """
            return self._mark_read(sid, index)

        def gmail_mark_unread(index: int) -> str:
            """Mark email as unread.
            Args:
                index: Email index (e.g. 1 for #1)
            """
            return self._mark_unread(sid, index)

        def gmail_archive(index: int) -> str:
            """Archive email (remove from inbox).
            Args:
                index: Email index (e.g. 1 for #1)
            """
            return self._archive(sid, index)

        def gmail_trash(index: int) -> str:
            """Move email to trash.
            Args:
                index: Email index (e.g. 1 for #1)
            """
            return self._trash_email(sid, index)

        def gmail_reply(index: int, body: str) -> str:
            """Reply to an email. Preserves thread and subject.
            Args:
                index: Email index to reply to (e.g. 1 for #1)
                body: Reply body text
            """
            return self._reply(sid, index, body)

        def gmail_modify_labels(
            index: int, add_labels: str = "", remove_labels: str = "",
        ) -> str:
            """Add/remove labels on an email.
            Args:
                index: Email index (e.g. 1 for #1)
                add_labels: Comma-separated labels to add (e.g. 'STARRED,IMPORTANT')
                remove_labels: Comma-separated labels to remove (e.g. 'INBOX')
            """
            return self._modify_labels(sid, index, add_labels, remove_labels)

        return [
            {"tool_func": gmail_login, "name": "gmail_login",
             "category": ["google", "gmail", "auth"]},
            {"tool_func": gmail_list, "name": "gmail_list",
             "category": ["google", "gmail", "read"]},
            {"tool_func": gmail_read, "name": "gmail_read",
             "category": ["google", "gmail", "read"]},
            {"tool_func": gmail_send, "name": "gmail_send",
             "category": ["google", "gmail", "write"]},
            {"tool_func": gmail_send_with_attachment, "name": "gmail_send_with_attachment",
             "category": ["google", "gmail", "write"]},
            {"tool_func": gmail_search, "name": "gmail_search",
             "category": ["google", "gmail", "read"]},
            {"tool_func": gmail_mark_read, "name": "gmail_mark_read",
             "category": ["google", "gmail", "write"]},
            {"tool_func": gmail_mark_unread, "name": "gmail_mark_unread",
             "category": ["google", "gmail", "write"]},
            {"tool_func": gmail_archive, "name": "gmail_archive",
             "category": ["google", "gmail", "write"]},
            {"tool_func": gmail_trash, "name": "gmail_trash",
             "category": ["google", "gmail", "write"]},
            {"tool_func": gmail_reply, "name": "gmail_reply",
             "category": ["google", "gmail", "write"]},
            {"tool_func": gmail_modify_labels, "name": "gmail_modify_labels",
             "category": ["google", "gmail", "write"]},
        ]

    # ── External Auth API (for Discord, Web, Telegram bots) ──────────

    def external_auth_start(self, session_id: str) -> str:
        """Returns auth URL string. Deliver this to user via your platform."""
        return self._login_start(session_id)

    def external_auth_complete(self, session_id: str, code: str) -> str:
        """Complete auth with code from user. Returns status markdown."""
        return self._login_complete(session_id, code)

    # ── Cleanup ──────────────────────────────────────────────────────

    def cleanup_session(self, session_id: str):
        """Remove in-memory session state (credentials persist on disk)."""
        self._sessions.pop(session_id, None)
