"""
Google Gmail Toolkit for Agent Sessions.

Session-isolated: Each agent session can have a different Google user.
Credentials are stored per session_id, so session 1 and session 2
can operate with different Gmail accounts simultaneously.
"""

import base64
import logging
import os
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)


class GmailToolkit:
    """Session-aware Gmail toolkit. Each session gets its own credentials/service."""

    def __init__(self, credentials_path: str = "/root/Toolboxv2/credentials.json",
                 token_dir: str = "token"):
        self.credentials_path = credentials_path
        self.token_dir = token_dir
        # Per-session state
        self._sessions: dict[str, dict[str, Any]] = {}

    def _get_session(self, session_id: str) -> dict[str, Any]:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "credentials": None,
                "service": None,
                "pending": {},
            }
        return self._sessions[session_id]

    def _token_path(self, session_id: str) -> str:
        return os.path.join(self.token_dir, f"gmail_token_{session_id}.json")

    def _ensure_service(self, session_id: str):
        s = self._get_session(session_id)
        if s["service"] is None:
            raise RuntimeError("Gmail not authenticated. Call gmail_auth_start first.")
        return s["service"]

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _auth_start(self, session_id: str) -> dict:
        """Generate OAuth2 authorization URL for this session."""
        from google_auth_oauthlib.flow import Flow

        SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
        flow = Flow.from_client_secrets_file(
            self.credentials_path,
            scopes=SCOPES,
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'
        )
        authorization_url, _ = flow.authorization_url(
            access_type='offline', prompt='consent'
        )
        s = self._get_session(session_id)
        s["pending"] = {"type": "auth", "flow_scopes": SCOPES}
        return {"success": True, "authorization_url": authorization_url}

    def _auth_complete(self, session_id: str, authorization_code: str) -> dict:
        """Complete OAuth2 flow with the code the user received."""
        from google_auth_oauthlib.flow import Flow

        s = self._get_session(session_id)
        SCOPES = s.get("pending", {}).get("flow_scopes",
                                          ['https://www.googleapis.com/auth/gmail.modify'])
        flow = Flow.from_client_secrets_file(
            self.credentials_path, scopes=SCOPES,
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'
        )
        flow.fetch_token(code=authorization_code)
        s["credentials"] = flow.credentials
        self._save_credentials(session_id)
        self._init_service(session_id)
        s["pending"] = {}
        return {"success": True, "message": "Gmail authenticated."}

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
            self._init_service(session_id)
            return True
        except (FileNotFoundError, Exception):
            return False

    def _init_service(self, session_id: str):
        from googleapiclient.discovery import build
        s = self._get_session(session_id)
        s["service"] = build('gmail', 'v1', credentials=s["credentials"])

    # ------------------------------------------------------------------
    # Gmail Operations
    # ------------------------------------------------------------------

    def _list_emails(self, session_id: str, query: str = "", max_results: int = 10) -> dict:
        """List emails from inbox, optionally filtered by query."""
        service = self._ensure_service(session_id)
        try:
            kwargs = {"userId": "me", "maxResults": max_results, "labelIds": ["INBOX"]}
            if query:
                kwargs["q"] = query
            results = service.users().messages().list(**kwargs).execute()
            emails = []
            for msg in results.get("messages", [])[:max_results]:
                data = service.users().messages().get(
                    userId="me", id=msg["id"], format="metadata"
                ).execute()
                headers = {h["name"]: h["value"] for h in data["payload"]["headers"]}
                emails.append({
                    "id": msg["id"],
                    "from": headers.get("From", "Unknown"),
                    "subject": headers.get("Subject", "No Subject"),
                    "date": headers.get("Date", "Unknown"),
                    "snippet": data.get("snippet", ""),
                    "unread": "UNREAD" in data.get("labelIds", []),
                })
            return {"success": True, "emails": emails, "count": len(emails)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_email(self, session_id: str, email_id: str) -> dict:
        """Get full email content by ID."""
        service = self._ensure_service(session_id)
        try:
            data = service.users().messages().get(
                userId="me", id=email_id, format="full"
            ).execute()
            headers = {h["name"]: h["value"] for h in data["payload"]["headers"]}
            body = ""
            parts = data.get("payload", {}).get("parts", [])
            if parts:
                for part in parts:
                    if part["mimeType"] == "text/plain":
                        body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                        break
            elif data.get("payload", {}).get("body", {}).get("data"):
                body = base64.urlsafe_b64decode(
                    data["payload"]["body"]["data"]
                ).decode("utf-8")
            return {
                "success": True,
                "id": email_id,
                "from": headers.get("From", "Unknown"),
                "to": headers.get("To", "Unknown"),
                "subject": headers.get("Subject", "No Subject"),
                "date": headers.get("Date", "Unknown"),
                "body": body,
                "snippet": data.get("snippet", ""),
                "labels": data.get("labelIds", []),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _send_email(self, session_id: str, to: str, subject: str, body: str,
                    cc: str = "", bcc: str = "") -> dict:
        """Send a plain-text email."""
        service = self._ensure_service(session_id)
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
            return {"success": True, "message_id": result["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _send_email_with_attachment(self, session_id: str, to: str, subject: str,
                                    body: str, attachment_path: str,
                                    cc: str = "", bcc: str = "") -> dict:
        """Send an email with a file attachment."""
        service = self._ensure_service(session_id)
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
            return {"success": True, "message_id": result["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _search_emails(self, session_id: str, query: str, max_results: int = 10) -> dict:
        """Search emails using Gmail query syntax."""
        return self._list_emails(session_id, query=query, max_results=max_results)

    def _modify_labels(self, session_id: str, email_id: str,
                       add_labels: list[str] | None = None,
                       remove_labels: list[str] | None = None) -> dict:
        """Add or remove labels from an email (e.g. mark read/unread, archive)."""
        service = self._ensure_service(session_id)
        try:
            body = {}
            if add_labels:
                body["addLabelIds"] = add_labels
            if remove_labels:
                body["removeLabelIds"] = remove_labels
            service.users().messages().modify(
                userId="me", id=email_id, body=body
            ).execute()
            return {"success": True, "email_id": email_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _trash_email(self, session_id: str, email_id: str) -> dict:
        """Move an email to trash."""
        service = self._ensure_service(session_id)
        try:
            service.users().messages().trash(userId="me", id=email_id).execute()
            return {"success": True, "email_id": email_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Toolkit registration
    # ------------------------------------------------------------------

    def get_tools(self, session_id: str) -> list[dict]:
        """
        Return list of tool dicts for agent registration.
        All tools are bound to the given session_id.
        Auto-loads saved credentials if available.
        """
        self._load_credentials(session_id)

        def gmail_auth_start() -> dict:
            """Start Gmail OAuth2 authentication. Returns an authorization URL the user must visit."""
            return self._auth_start(session_id)

        def gmail_auth_complete(authorization_code: str) -> dict:
            """Complete Gmail OAuth2 with the authorization code from Google.
            Args:
                authorization_code: The code received after user authorization
            """
            return self._auth_complete(session_id, authorization_code)

        def gmail_list(query: str = "", max_results: int = 10) -> dict:
            """List emails from inbox. Supports Gmail search operators.
            Args:
                query: Gmail search query (e.g. 'is:unread', 'from:user@example.com')
                max_results: Max emails to return (default 10)
            """
            return self._list_emails(session_id, query, max_results)

        def gmail_read(email_id: str) -> dict:
            """Read full email content by ID.
            Args:
                email_id: The Gmail message ID
            """
            return self._get_email(session_id, email_id)

        def gmail_send(to: str, subject: str, body: str,
                       cc: str = "", bcc: str = "") -> dict:
            """Send a plain-text email.
            Args:
                to: Recipient email address
                subject: Email subject
                body: Email body text
                cc: CC recipients (comma-separated)
                bcc: BCC recipients (comma-separated)
            """
            return self._send_email(session_id, to, subject, body, cc, bcc)

        def gmail_send_with_attachment(to: str, subject: str, body: str,
                                       attachment_path: str,
                                       cc: str = "", bcc: str = "") -> dict:
            """Send an email with a file attachment.
            Args:
                to: Recipient email address
                subject: Email subject
                body: Email body text
                attachment_path: Local file path to attach
                cc: CC recipients (comma-separated)
                bcc: BCC recipients (comma-separated)
            """
            return self._send_email_with_attachment(
                session_id, to, subject, body, attachment_path, cc, bcc
            )

        def gmail_search(query: str, max_results: int = 10) -> dict:
            """Search emails using Gmail query syntax.
            Args:
                query: Gmail search query (operators: from:, to:, subject:, has:attachment, is:unread, newer_than:7d, etc.)
                max_results: Max results (default 10)
            """
            return self._search_emails(session_id, query, max_results)

        def gmail_modify_labels(email_id: str,
                                add_labels: list[str] | None = None,
                                remove_labels: list[str] | None = None) -> dict:
            """Add/remove labels on an email. Use to mark read/unread, archive, etc.
            Args:
                email_id: The Gmail message ID
                add_labels: Labels to add (e.g. ['UNREAD'])
                remove_labels: Labels to remove (e.g. ['INBOX'] to archive)
            """
            return self._modify_labels(session_id, email_id, add_labels, remove_labels)

        def gmail_trash(email_id: str) -> dict:
            """Move an email to trash.
            Args:
                email_id: The Gmail message ID
            """
            return self._trash_email(session_id, email_id)

        return [
            {
                "tool_func": gmail_auth_start,
                "name": "gmail_auth_start",
                "category": ["google", "gmail", "auth"],
            },
            {
                "tool_func": gmail_auth_complete,
                "name": "gmail_auth_complete",
                "category": ["google", "gmail", "auth"],
            },
            {
                "tool_func": gmail_list,
                "name": "gmail_list",
                "category": ["google", "gmail", "read"],
            },
            {
                "tool_func": gmail_read,
                "name": "gmail_read",
                "category": ["google", "gmail", "read"],
            },
            {
                "tool_func": gmail_send,
                "name": "gmail_send",
                "category": ["google", "gmail", "write"],
            },
            {
                "tool_func": gmail_send_with_attachment,
                "name": "gmail_send_with_attachment",
                "category": ["google", "gmail", "write"],
            },
            {
                "tool_func": gmail_search,
                "name": "gmail_search",
                "category": ["google", "gmail", "read"],
            },
            {
                "tool_func": gmail_modify_labels,
                "name": "gmail_modify_labels",
                "category": ["google", "gmail", "write"],
            },
            {
                "tool_func": gmail_trash,
                "name": "gmail_trash",
                "category": ["google", "gmail", "write"],
            },
        ]

    def cleanup_session(self, session_id: str):
        """Remove session state (call on session end)."""
        self._sessions.pop(session_id, None)
