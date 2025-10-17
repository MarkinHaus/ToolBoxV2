import base64
import os
import time
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dateutil import parser

from toolboxv2.mods.isaa.base.AgentUtils import LLMMode

try:
    from whatsapp import Message, WhatsApp
except ImportError:
    print("NO Whatsapp installed")
    def WhatsApp():
        return None
    def Message():
        return None
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
try:
    from datetime import UTC, datetime, timedelta
except ImportError:
    from datetime import datetime, timedelta, timezone
    UTC = timezone.utc
from enum import Enum
from typing import Any, Optional

from google.oauth2.credentials import Credentials

from toolboxv2 import TBEF, get_app
from toolboxv2.mods.isaa import Tools
from toolboxv2.mods.WhatsAppTb.server import AppManager
from toolboxv2.mods.WhatsAppTb.utils import (
    ProgressMessenger,
    emoji_set_thermometer,
    emoji_set_work_phases,
)
from toolboxv2.utils.extras.blobs import BlobFile, BlobStorage
from toolboxv2.utils.system import FileCache


@dataclass
class WhClient:
    messenger: WhatsApp
    disconnect: Callable
    s_callbacks: Callable
    progress_messenger0: ProgressMessenger
    progress_messenger1: ProgressMessenger
    progress_messenger2: ProgressMessenger
    set_to: Callable
    to: str



class AssistantState(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"


class DocumentSystem:
    def __init__(self, storage: BlobStorage):
        self.storage = storage
        self.media_types = {
            'document': ['pdf', 'doc', 'docx', 'txt'],
            'image': ['jpg', 'jpeg', 'png', 'gif'],
            'video': ['mp4', 'mov', 'avi']
        }

    def list_documents(self, filter_type: str = None) -> list[dict]:
        """List all documents with metadata"""
        docs = []
        for blob_id in self.storage._get_all_blob_ids():
            with BlobFile(blob_id, 'r', self.storage) as f:
                metadata = f.read_json()
                if metadata:
                    docs.append({
                        'id': blob_id,
                        'name': metadata.get('filename', blob_id),
                        'type': metadata.get('type', 'document'),
                        'size': metadata.get('size', 0),
                        'modified': metadata.get('timestamp', ''),
                        'preview': metadata.get('preview', '')
                    })
        if filter_type:
            return [d for d in docs if d['type'] == filter_type]
        return docs

    def save_document(self, file_data: bytes, filename: str, file_type: str) -> str:
        """Save a document with metadata"""
        blob_id = self.storage._generate_blob_id()
        metadata = {
            'filename': filename,
            'type': file_type,
            'size': len(file_data),
            'timestamp': datetime.now().isoformat(),
            'preview': self._generate_preview(file_data, file_type)
        }

        with BlobFile(blob_id, 'w', self.storage) as f:
            f.write_json(metadata)
            f.write(file_data)
        return blob_id

    def delete_document(self, blob_id: str) -> bool:
        """Delete a document"""
        try:
            self.storage.delete_blob(blob_id)
            return True
        except Exception as e:
            logging.error(f"Delete failed: {str(e)}")
            return False

    def search_documents(self, query: str) -> list[dict]:
        """Search documents by filename or content"""
        results = []
        for doc in self.list_documents():
            if query.lower() in doc['name'].lower() or self._search_in_content(doc['id'], query):
                results.append(doc)
        return results

    def _generate_preview(self, data: bytes, file_type: str) -> str:
        """Generate preview based on file type"""
        if file_type in self.media_types['image']:
            return f"Image preview: {data[:100].hex()}"
        elif file_type in self.media_types['video']:
            return "Video preview unavailable"
        return data[:100].decode('utf-8', errors='ignore')

    def _search_in_content(self, blob_id: str, query: str) -> bool:
        """Search content within documents"""
        try:
            with BlobFile(blob_id, 'r', self.storage) as f:
                content = f.read().decode('utf-8', errors='ignore')
                return query.lower() in content.lower()
        except:
            return False

@dataclass
class WhatsAppAssistant:
    whc: WhClient
    isaa: 'Tools'
    agent: Optional['Agent'] = None
    credentials: Credentials | None = None
    state: AssistantState = AssistantState.OFFLINE

    # Service clients
    gmail_service: Any = None
    calendar_service: Any = None

    start_time: Any = None

    blob_docs_system: Any = None
    duration_minutes: int = 20
    credentials_path: str = "/root/Toolboxv2/credentials.json"
    # Progress messengers
    progress_messengers: dict[str, 'ProgressMessenger'] = field(default_factory=dict)
    buttons: dict[str, dict] = field(default_factory=dict)
    history: FileCache = field(default_factory=FileCache)

    pending_actions: dict[str, dict] = field(default_factory=dict)


    def __post_init__(self):

        self.start_time = datetime.now()
        self.processed_messages = set()
        self.message_lock = threading.Lock()
        self.audio_processor = None
        self.blob_docs_system = DocumentSystem(BlobStorage())
        self.stt = get_app().run_any(TBEF.AUDIO.STT_GENERATE,
                                     model="openai/whisper-small",
                                     row=False, device=1)

        self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {}

        self.load_credentials()
        self.setup_progress_messengers()
        self.setup_interaction_buttons()
        self.history = FileCache(folder=".data/WhatsAppAssistant")
        self.state = AssistantState.ONLINE

    async def generate_authorization_url(self, *a):
        """
        Generate an authorization URL for user consent

        :return: Authorization URL for the user to click and authorize access
        """
        from google_auth_oauthlib.flow import Flow
        # Define the scopes required for Gmail and Calendar
        SCOPES = [
            'https://www.googleapis.com/auth/gmail.modify',
            'https://www.googleapis.com/auth/calendar'
        ]

        # Create a flow instance to manage the OAuth 2.0 authorization process
        flow = Flow.from_client_secrets_file(
            self.credentials_path,
            scopes=SCOPES,
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'  # Use 'urn:ietf:wg:oauth:2.0:oob' for desktop apps
        )

        # Generate the authorization URL
        authorization_url, _ = flow.authorization_url(
            access_type='offline',  # Allows obtaining refresh token
            prompt='consent'  # Ensures user is always prompted for consent
        )
        self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {'type': 'auth',
                                                                              'step': 'awaiting_key'}
        return {
            'type': 'quick_reply',
            'text': f'Url to log in {authorization_url}',
            'options': {'cancel': '❌ Cancel Upload'}
        }

    def complete_authorization(self, message: Message):
        """
        Complete the authorization process using the authorization code

        :param authorization_code: Authorization code received from Google
        """
        from google_auth_oauthlib.flow import Flow
        authorization_code = message.content
        # Define the scopes required for Gmail and Calendar
        SCOPES = [
            'https://www.googleapis.com/auth/gmail.modify',
            'https://www.googleapis.com/auth/calendar'
        ]

        # Create a flow instance to manage the OAuth 2.0 authorization process
        flow = Flow.from_client_secrets_file(
            self.credentials_path,
            scopes=SCOPES,
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'
        )

        # Exchange the authorization code for credentials
        flow.fetch_token(code=authorization_code)
        self.credentials = flow.credentials

        # Save the credentials for future use
        self.save_credentials()

        # Initialize services
        self.init_services()
        return "Done"


    def save_credentials(self):
        """
        Save the obtained credentials to a file for future use
        """
        if not os.path.exists('token'):
            os.makedirs('token')

        with open('token/google_token.json', 'w') as token_file:
            token_file.write(self.credentials.to_json())


    def load_credentials(self):
        """
        Load previously saved credentials if available

        :return: Whether credentials were successfully loaded
        """
        try:
            self.credentials = Credentials.from_authorized_user_file('token/google_token.json')
            self.init_services()
            return True
        except FileNotFoundError:
            return False


    def init_services(self):
        """
        Initialize Gmail and Calendar services
        """
        from googleapiclient.discovery import build

        self.gmail_service = build('gmail', 'v1', credentials=self.credentials)
        self.calendar_service = build('calendar', 'v3', credentials=self.credentials)
        self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {}

    def setup_progress_messengers(self):
        """Initialize progress messengers for different types of tasks"""
        self.progress_messengers = {
            'task': self.whc.progress_messenger0,
            'email': self.whc.progress_messenger1,
            'calendar': self.whc.progress_messenger2
        }

    def setup_interaction_buttons(self):
        """Define WhatsApp interaction buttons for different functionalities"""
        self.buttons = {
            'menu': {
                'header': 'Digital Assistant',
                'body': 'Please select an option:',
                'footer': '-- + --',
                'action': {
                    'button': 'Menu',
                    'sections': [
                        {
                            'title': 'Main Functions',
                            'rows': [
                                {'id': 'agent', 'title': 'Agent Controls', 'description': 'Manage your AI assistant'},
                                {'id': 'email', 'title': 'Email Management', 'description': 'Handle your emails'},
                                {'id': 'calendar', 'title': 'Calendar', 'description': 'Manage your schedule'},
                                {'id': 'docs', 'title': 'Documents', 'description': 'Handle documents'},
                                {'id': 'system', 'title': 'System', 'description': 'System controls and metrics'}
                            ]
                        }
                    ]
                }
            },
            'agent': self._create_agent_controls_buttons(),
            'email': self._create_email_controls_buttons(),
            'calendar': self._create_calendar_controls_buttons(),
            'docs': self._create_docs_controls_buttons(),
            'system': self._create_system_controls_buttons()
        }

    @staticmethod
    def _create_agent_controls_buttons():
        return {
            'header': 'Agent Controls',
            'body': 'Manage your AI assistant:',
            'action': {
                'button': 'Select',
                'sections': [
                    {
                        'title': 'Basic Actions',
                        'rows': [
                            {'id': 'agent-task', 'title': 'Agent Task', 'description': 'Run the agent'},
                            {'id': 'start', 'title': 'Start Agent', 'description': 'Run taskstack in background'},
                            {'id': 'stop', 'title': 'Stop Agent', 'description': 'Stop taskstack execution'}
                        ]
                    },
                    {
                        'title': 'Advanced Actions',
                        'rows': [
                            {'id': 'system-task', 'title': 'System Task',
                             'description': 'Run the Isaa Reasoning Agent system'},
                            {'id': 'tasks', 'title': 'Task Stack', 'description': 'View and manage tasks'},
                            {'id': 'memory', 'title': 'Clear Memory', 'description': 'Reset agent memory'}
                        ]
                    }
                ]
            }
        }

    @staticmethod
    def _create_email_controls_buttons():
        return {
            'header': 'Email Management',
            'body': 'Handle your emails:',
            'action': {
                'button': 'Select',
                'sections': [
                    {
                        'title': 'Basic Actions',
                        'rows': [
                            {'id': 'check', 'title': 'Check Emails', 'description': 'View recent emails'},
                            {'id': 'send', 'title': 'Send Email', 'description': 'Compose new email'},
                            {'id': 'summary', 'title': 'Get Summary', 'description': 'Summarize emails'}
                        ]
                    },
                    {
                        'title': 'Advanced Actions',
                        'rows': [
                            {'id': 'search', 'title': 'Search', 'description': 'Search emails'}
                        ]
                    }
                ]
            }
        }

    @staticmethod
    def _create_calendar_controls_buttons():
        return {
            'header': 'Calendar Management',
            'body': 'Manage your schedule:',
            'action': {
                'button': 'Select',
                'sections': [
                    {
                        'title': 'Basic Actions',
                        'rows': [
                            {'id': 'today', 'title': 'Today\'s Events', 'description': 'View today\'s schedule'},
                            {'id': 'add', 'title': 'Add Event', 'description': 'Create new event'},
                            {'id': 'upcoming', 'title': 'Upcoming', 'description': 'View upcoming events'}
                        ]
                    },
                    {
                        'title': 'Advanced Actions',
                        'rows': [
                            {'id': 'find_slot', 'title': 'Find Time Slot', 'description': 'Find available time'}
                        ]
                    }
                ]
            }
        }

    @staticmethod
    def _create_docs_controls_buttons():
        return {
            'header': 'Document Management',
            'body': 'Handle your documents:',
            'action': {
                'button': 'Select',
                'sections': [
                    {
                        'title': 'Basic Actions',
                        'rows': [
                            {'id': 'upload', 'title': 'Upload', 'description': 'Add new document'},
                            {'id': 'list', 'title': 'List Documents', 'description': 'View all documents'},
                            {'id': 'search', 'title': 'Search', 'description': 'Search documents'}
                        ]
                    },
                    {
                        'title': 'Advanced Actions',
                        'rows': [
                            {'id': 'delete', 'title': 'Delete', 'description': 'Remove document'}
                        ]
                    }
                ]
            }
        }

    @staticmethod
    def _create_system_controls_buttons():
        return {
            'header': 'System Controls',
            'body': 'System management:',
            'action': {
                'button': 'Select',
                'sections': [
                    {
                        'title': 'Basic Actions',
                        'rows': [
                            {'id': 'status', 'title': 'System Status', 'description': 'View current status'},
                            {'id': 'restart', 'title': 'Restart', 'description': 'Restart system'},
                            {'id': 'connect', 'title': 'Connect', 'description': 'Connect to Google Calendar and Email'}
                        ]
                    }
                ]
            }
        }

    async def handle_message(self, message: 'Message'):
        """Main message handler for incoming WhatsApp messages"""

        # Deduplication check
        with self.message_lock:
            if message.id in self.processed_messages:
                return
            last_ts = time.time()
            print(last_ts)
            if len(self.processed_messages) > 0:
                m_id, last_ts = self.processed_messages.pop()
                self.processed_messages.add((m_id, last_ts))

            print("DUPLICATION P", message.data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {}).get('messages', [{}])[0].get('timestamp', 0) , last_ts)
            if float(message.data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {}).get('messages', [{}])[0].get('timestamp', 0)) < last_ts - 120:
                return
            self.processed_messages.add((message.id, time.perf_counter()))

        # Mark message as read
        message.mark_as_read()

        # Extract content and type
        content_type = message.type
        content = message.content

        print(f"message.content {content=} {content_type=} {message.data=}")

        try:
            if content_type == 'interactive':
                await self.handle_interactive(message)
            elif content_type == 'audio':
                await self.handle_audio_message(message)
            elif content_type in ['document', 'image', 'video']:
                response = await self.handle_media_message(message)
                self.save_reply(message, response)
            elif content_type == 'text':
                if content.lower() == "menu":
                    self.whc.messenger.send_button(
                        recipient_id=self.whc.progress_messenger0.recipient_phone,
                        button=self.buttons[content.lower()]
                    )
                else:
                    await self.helper_text(message)
            else:
                message.reply("Unsupported message type")
        #except Exception as e:
        #    logging.error(f"Message handling error: {str(e)}")
        #   message.reply("❌ Error processing request")
        finally:
            # Cleanup old messages (keep 1 hour history)
            with self.message_lock:
                self._clean_processed_messages()

    async def helper_text(self, message: 'Message', return_text=False):
        if not isinstance(message.content, str) and not len(message.content) > 0:
            content = self.whc.messenger.get_message(message.data)
            print(f"contents {content=}, {message.content=}")
            message.content = content
        self.history.set(message.id, message.content)
        if len(self.pending_actions[self.whc.progress_messenger0.recipient_phone].keys()) != 0:
            message.reply(
                f"Open Interaction : {json.dumps(self.pending_actions[self.whc.progress_messenger0.recipient_phone], indent=2)}")
            if self.pending_actions[self.whc.progress_messenger0.recipient_phone].get('type') == 'auth':
                res = self.complete_authorization(message)
                self.save_reply(message, res)
            res = await self.handle_calendar_actions(message)
            if res:
                self.save_reply(message, res)
                return
            res2 = await self.handle_email_actions(message)
            if res2:
                self.save_reply(message, res2)
                return
            await self.handle_agent_actions(message)
            return
        await self.handle_agent_actions(message)

    async def handle_interactive(self, message: Message):
        """Handle all interactive messages"""
        content = self.whc.messenger.get_interactive_response(message.data)
        if content.get("type") == "list_reply":
            await self.handle_button_interaction(content.get("list_reply"), message)
        elif content.get("type") == "button_reply":
            print(content)

    async def handle_audio_message(self, message: 'Message'):
        """Process audio messages with STT and TTS"""
        # Download audio
        progress = self.progress_messengers['task']
        stop_flag = threading.Event()
        # message_id = progress.send_initial_message(mode="loading")
        progress.message_id = message.id
        progress.start_loading_in_background(stop_flag)

        content = self.whc.messenger.get_audio(message.data)
        audio_file_name = self.whc.messenger.download_media(media_url=self.whc.messenger.query_media_url(media_id=content.get('id')), mime_type='audio/opus', file_path=".data/temp")
        print(f"audio_file_name {audio_file_name}")
        if audio_file_name is None:
            message.reply("Could not process audio file")
            stop_flag.set()
            return

        text = self.stt(audio_file_name)['text']
        if not text:
            message.reply("Could not process audio")
            stop_flag.set()
            return

        message.reply("Transcription :\n "+ text)
        message.content = text
        agent_res = await self.helper_text(message, return_text=True)

        if agent_res is not None:
            pass

        stop_flag.set()
        # Process text and get response
        # response = await self.process_input(text, message)

        # Convert response to audio
        #audio_file = self.audio_processor.tts(response)
        #audio_file = None # TODO
        #self.whc.messenger.send_audio(
        #    audio=audio_file,
        #    recipient_id=self.whc.progress_messenger0.recipient_phone,
        #)

    async def confirm(self, message: Message):
        status = self.pending_actions[self.whc.progress_messenger0.recipient_phone]
        if status.get('type') == "create_event":
            if status.get('step') == "confirm_envet":
                event = self._create_calendar_event(status.get('event_data'))
                self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {}
                return f"✅ Event created!\n{event.get('htmlLink')}"
            return "❌"
        elif status.get('type') == "compose_email":
            if status.get('step') == "confirm_email":
                # Send email
                result = self.gmail_service.users().messages().send(
                    userId='me',
                    body=self._build_email_draft(status['draft'])
                ).execute()
                self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {}
                return f"✅ Email sent! Message ID: {result['id']}"
            return "❌"
        return "❌ Done"

    async def cancel(self, *a):
        self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {}
        return "✅ cancel Done"

    async def handle_button_interaction(self, content: dict, message: Message):
        """Handle button click interactions"""
        button_id = content['id']

        # First check if it's a main menu button
        if button_id in self.buttons:
            self.whc.messenger.send_button(
                recipient_id=self.whc.progress_messenger0.recipient_phone,
                button=self.buttons[button_id]
            )
            return

        # Handle action buttons
        action_handlers = {
            # Agent controls
            'start': self.start_agent,
            'stop': self.stop_agent,
            'tasks': self.show_task_stack,
            'memory': self.clear_memory,
            'system-task': self.system_task,
            'agent-task': self.agent_task,

            # Email controls
            'check': self.check_emails,
            'send': self.start_email_compose,
            'summary': self.email_summary,
            'search': self.email_search,

            # Calendar controls
            'today': self.show_today_events,
            'add': self.start_event_create,
            'upcoming': self.show_upcoming_events,
            'find_slot': self.find_time_slot,

            # Document controls
            'upload': self.start_document_upload,
            'list': self.list_documents,
            'search_docs': self.search_documents,
            'delete': self.delete_document,

            # System controls
            'status': self.system_status,
            'restart': self.restart_system,
            'connect': self.generate_authorization_url,

            'cancel': self.cancel,
            'confirm': self.confirm,
        }
        if button_id in action_handlers:
            try:
                # Start progress indicator
                progress = self.progress_messengers['task']
                stop_flag = threading.Event()
                # message_id = progress.send_initial_message(mode="loading")
                progress.message_id = message.id
                progress.start_loading_in_background(stop_flag)

                # Execute handler

                result = await action_handlers[button_id](message)


                # Send result
                if isinstance(result, str):
                    self.save_reply(message, result)
                elif isinstance(result, dict):  # For structured responses
                    self.send_structured_response(result)

                stop_flag.set()
            finally:
                #except Exception as e:
                stop_flag.set()
            #    message.reply(f"❌ Error processing {button_id}: {str(e)}")
        elif 'event_' in button_id:
            res = await self.get_event_details(button_id.replace("event_", ''))
            if isinstance(res, str):
                self.save_reply(message, res)
                return
            for r in res:
                if isinstance(r, str):
                    self.save_reply(message, r)
                else:
                    self.whc.messenger.send_location(**r)

        elif 'email_' in button_id:
            res = await self.get_email_details(button_id.replace("email_", ''))
            self.save_reply(message, res)
        else:
            message.reply("⚠️ Unknown command")

    def send_structured_response(self, result: dict):
        """Send complex responses using appropriate WhatsApp features"""
        if result['type'] == 'list':
            self.whc.messenger.send_button(
                recipient_id=self.whc.progress_messenger0.recipient_phone,
                button={
                    'header': result.get('header', ''),
                    'body': result.get('body', ''),
                    'footer': result.get('footer', ''),
                    'action': {
                        'button': 'Action',
                        'sections': result['sections']
                    }
                }
            )
        elif result['type'] == 'quick_reply':
            self.whc.messenger.send_button(
                recipient_id=self.whc.progress_messenger0.recipient_phone,
                button={
                    'header': "Quick reply",
                    'body': result['text'],
                    'footer': '',
                    'action': {'button': 'Action', 'sections': [{
                        'title': 'View',
                        'rows': [{'id': k, 'title': v[:23]} for k, v in result['options'].items()]
                    }]}
                }
            )

        elif result['type'] == 'media':
            if result['media_type'] == 'image':
                self.whc.messenger.send_image(
                    image=result['url'],
                    recipient_id=self.whc.progress_messenger0.recipient_phone,
                    caption=result.get('caption', '')
                )
            elif result['media_type'] == 'document':
                self.whc.messenger.send_document(
                    document=result['url'],
                    recipient_id=self.whc.progress_messenger0.recipient_phone,
                    caption=result.get('caption', '')
                )

    async def clear_memory(self, message):
        self.agent.reset_context()
        self.agent.taskstack.tasks = []
        return "🧠 Memory cleared successfully"

    async def system_task(self, message):
        """Initiate email search workflow"""
        self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {
            'type': 'system',
            'step': 'await_query'
        }
        return {
            'type': 'quick_reply',
            'text': "Now prompt the 🧠ISAA-System 📝",
            'options': {'cancel': '❌ Cancel Search'}
        }

    async def agent_task(self, message):
        """Initiate email search workflow"""
        self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {
            'type': 'self-agent',
            'step': 'await_query'
        }
        return {
            'type': 'quick_reply',
            'text': "Now prompt the self-agent 📝",
            'options': {'cancel': '❌ Cancel Search'}
        }

    async def check_emails(self, message, query=""):
        """Improved email checking with WhatsApp API formatting"""
        if not self.gmail_service:
            return "⚠️ Gmail service not configured"

        try:
            results = self.gmail_service.users().messages().list(
                userId='me',
                maxResults=10,
                labelIds=['INBOX'],
                q=query
            ).execute()

            emails = []
            for msg in results.get('messages', [])[:10]:
                email_data = self.gmail_service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='metadata'
                ).execute()

                headers = {h['name']: h['value'] for h in email_data['payload']['headers']}
                emails.append({
                    'id': msg['id'],
                    'from': headers.get('From', 'Unknown'),
                    'subject': headers.get('Subject', 'No Subject'),
                    'date': headers.get('Date', 'Unknown'),
                    'snippet': email_data.get('snippet', ''),
                    'unread': 'UNREAD' in email_data.get('labelIds', [])
                })

            return {
                'type': 'list',
                'header': '📨 Recent Emails',
                'body': 'Tap to view full email',
                'footer': 'Email Manager',
                'sections': [{
                    'title': f"Inbox ({len(emails)} emails)",
                    'rows': [{
                        'id': f"email_{email['id']}",
                        'title': f"{'📬' if email['unread'] else '📭'} {email['subject']}"[:23],
                        'description': f"From: {email['from']}\n{email['snippet']}"[:45]
                    } for email in emails]
                }]
            }
        except Exception as e:
            return f"⚠️ Error fetching emails: {str(e)}"

    async def get_email_details(self, email_id):
        """Retrieve and format full email details"""
        if not self.gmail_service:
            return "⚠️ Gmail service not configured"

        try:
            email_data = self.gmail_service.users().messages().get(
                userId='me',
                id=email_id,
                format='full'
            ).execute()

            headers = {h['name']: h['value'] for h in email_data['payload']['headers']}
            body = ""
            for part in email_data.get('payload', {}).get('parts', []):
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break

            formatted_text = (
                f"📧 *Email Details*\n\n"
                f"From: {headers.get('From', 'Unknown')}\n"
                f"Subject: {headers.get('Subject', 'No Subject')}\n"
                f"Date: {headers.get('Date', 'Unknown')}\n\n"
                f"{body[:15000]}{'...' if len(body) > 15000 else ''}"
            )
            return  self.agent.mini_task(
                formatted_text , "system", "Summarize the email in bullet points with key details"
            )
        except Exception as e:
            return f"⚠️ Error fetching email: {str(e)}"

    async def email_summary(self, message):
        """Generate AI-powered email summaries"""
        try:
            messages = self.gmail_service.users().messages().list(
                userId='me',
                maxResults=3,
                labelIds=['INBOX']
            ).execute().get('messages', [])

            email_contents = []
            for msg in messages[:3]:
                email_data = self.gmail_service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='full'
                ).execute()
                email_contents.append(self._parse_email_content(email_data))

            summary = self.agent.mini_task(
                "\n\n".join(email_contents) , "system", "Summarize these emails in bullet points with key details:"
            )

            return f"📋 Email Summary:\n{summary}\n\n*Powered by AI*"
        except Exception as e:
            logging.error(f"Summary failed: {str(e)}")
            return f"❌ Could not generate summary: {str(e)}"

    async def email_search(self, message):
        """Initiate email search workflow"""
        self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {
            'type': 'email_search',
            'step': 'await_query'
        }
        return {
            'type': 'quick_reply',
            'text': "🔍 What would you like to search for?",
            'options': {'cancel': '❌ Cancel Search'}
        }

    async def start_email_compose(self, message):
        """Enhanced email composition workflow"""
        self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {
            'type': 'compose_email',
            'step': 'subject',
            'draft': {'attachments': []}
        }
        return {
            'type': 'quick_reply',
            'text': "📝 Let's compose an email\n\nSubject:",
            'options': {'cancel': '❌ Cancel Composition'}
        }

    async def handle_email_actions(self, message):
        """Handle multi-step email workflows"""
        user_state = self.pending_actions.get(self.whc.progress_messenger0.recipient_phone, {})

        if user_state.get('type') == 'compose_email':
            return await self._handle_email_composition(message, user_state)
        if user_state.get('type') == 'email_search':
            return await self.check_emails(message, self.agent.mini_task("""Conventire Pezise zu einer googel str only query using : Gmail Suchoperatoren!

Basis-Operatoren:
- from: Absender
- to: Empfänger
- subject: Betreff
- label: Gmail Label
- has:attachment Anhänge
- newer_than:7d Zeitfilter
- before: Datum vor
- after: Datum nach

Erweiterte Operatoren:
- in:inbox
- in:sent
- in:spam
- cc: Kopie
- bcc: Blindkopie
- is:unread
- is:read
- larger:10M Größenfilter
- smaller:5M
- filename:pdf Dateityp

Profi-Tipps:
- Kombinierbar mit UND/ODER
- Anführungszeichen für exakte Suche
- Negation mit -
 beispeile : 'Ungelesene Mails letzte Woche': -> 'is:unread newer_than:7d'

""", "user",message.content))


        return None

    async def _handle_email_composition(self, message, state):
        if state['step'] == 'subject':
            state['draft']['subject'] = message.content
            state['step'] = 'body'
            return {
                'type': 'quick_reply',
                'text': "✍️ Email body:",
                'options': {'attach': '📎 Add Attachment', 'send': '📤 Send Now'}
            }

        elif state['step'] == 'body':
            if message.content == 'attach':
                state['step'] = 'attachment'
                return "📎 Please send the file you want to attach"

            state['draft']['body'] = message.content
            state['step'] = 'confirm_email'
            return {
                'type': 'quick_reply',
                'text': f"📧 Ready to send?\n\nSubject: {state['draft']['subject']}\n\n{state['draft']['body']}",
                'options': {'confirm': '✅ Send', 'cancel': '❌ cancel'}
            }

        elif state['step'] == 'attachment':
            # Handle attachment upload
            file_type = message.type
            if file_type not in ['document', 'image']:
                return "❌ Unsupported file type"

            media_url = getattr(message, file_type).id
            media_data = self.whc.messenger.download_media(media_url=self.whc.messenger.query_media_url(media_id=media_url), mime_type=media_url.type, file_path=".data/temp")
            state['draft']['attachments'].append(media_data)
            state['step'] = 'body'
            return "📎 Attachment added! Add more or send the email"


    def _parse_email_content(self, email_data):
        """Extract readable content from email payload"""
        parts = email_data.get('payload', {}).get('parts', [])
        body = ""
        for part in parts:
            if part['mimeType'] == 'text/plain':
                body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
        return f"Subject: {email_data.get('subject', '')}\nFrom: {email_data.get('from', '')}\n\n{body}"

    def _build_email_draft(self, draft):
        """Create MIME message from draft data"""
        message = MIMEMultipart()
        message['to'] = draft.get('to', '')
        message['subject'] = draft['subject']
        message.attach(MIMEText(draft['body']))

        for attachment in draft['attachments']:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment')
            message.attach(part)

        return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

    def _get_email_subject(self, msg):
        headers = msg.get('payload', {}).get('headers', [])
        return next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')

    def _get_email_sender(self, msg):
        headers = msg.get('payload', {}).get('headers', [])
        return next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')

    def _get_email_snippet(self, msg):
        return msg.get('snippet', '')[:100] + '...'
    # Calendar Handlers

    # Calendar Functions
    def _format_event_time(self, event):
        """Improved time formatting for calendar events"""
        start = event['start'].get('dateTime', event['start'].get('date'))
        end = event['end'].get('dateTime', event['end'].get('date'))

        try:
            start_dt = parser.parse(start)
            end_dt = parser.parse(end)
            if 'T' in start:
                return f"{start_dt.strftime('%a %d %b %H:%M')} - {end_dt.strftime('%H:%M')}"
            return f"{start_dt.strftime('%d %b %Y')} (All Day)"
        except:
            return "Time not specified"

    async def get_event_details(self, event_id):
        """Retrieve and format calendar event details with location support"""
        if not self.calendar_service:
            return "⚠️ Calendar service not configured"

        try:
            event = self.calendar_service.events().get(
                calendarId='primary',
                eventId=event_id
            ).execute()

            response = [ (
                    f"📅 *Event Details*\n\n"
                    f"Title: {event.get('summary', 'No title')}\n"
                    f"Time: {self._format_event_time(event)}\n"
                    f"Location: {event.get('location', 'Not specified')}\n\n"
                    f"{event.get('description', 'No description')[:1000]}"
                )]

            if 'geo' in event:
                response.append({
                    'lat': float(event['geo']['latitude']),
                    'long': float(event['geo']['longitude']),
                    'name': event.get('location', 'Event Location'),
                    'address': event.get('location', ''),
                    'recipient_id': self.whc.progress_messenger0.recipient_phone
                })
            return response
        except Exception as e:
            return f"⚠️ Error fetching event: {str(e)}"

    async def show_today_events(self, message):
        """Show today's calendar events"""
        if not self.calendar_service:
            message.replay("service not online")

        now = datetime.utcnow().isoformat() + 'Z'
        end_of_day = (datetime.now() + timedelta(days=1)).replace(
            hour=0, minute=0, second=0).isoformat() + 'Z'

        events_result = self.calendar_service.events().list(
            calendarId='primary',
            timeMin=now,
            timeMax=end_of_day,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        return self._format_calendar_response(events, "Today's Events")

    # Updated Calendar List Handlers
    async def show_upcoming_events(self, message):
        """Show upcoming events with interactive support"""
        if not self.calendar_service:
            return "⚠️ Calendar service not configured"

        try:
            now = datetime.utcnow().isoformat() + 'Z'
            next_week = (datetime.now() + timedelta(days=7)).isoformat() + 'Z'

            events_result = self.calendar_service.events().list(
                calendarId='primary',
                timeMin=now,
                timeMax=next_week,
                singleEvents=True,
                orderBy='startTime',
                maxResults=10
            ).execute()

            events = events_result.get('items', [])
            return self._format_calendar_response(events, "Upcoming Events")
        except Exception as e:
            return f"⚠️ Error fetching events: {str(e)}"

    async def start_event_create(self, message):
        """Initiate event creation workflow"""
        self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {
            'type': 'create_event',
            'step': 'title',
            'event_data': {}
        }
        return {
            'type': 'quick_reply',
            'text': "Let's create an event! What's the title?",
            'options': {'cancel': '❌ Cancel'}
        }

    async def find_time_slot(self, message):
        """Find and display the next 5 available time slots with dynamic durations"""
        if not self.calendar_service:
            return "⚠️ Calendar service not configured"

        try:
            # Define the time range for the search (next 24 hours)
            now = datetime.now(UTC)
            end_time = now + timedelta(days=1)

            # FreeBusy Request
            freebusy_request = {
                "timeMin": now.isoformat(),
                "timeMax": end_time.isoformat(),
                "items": [{"id": 'primary'}]
            }

            freebusy_response = self.calendar_service.freebusy().query(body=freebusy_request).execute()
            busy_slots = freebusy_response['calendars']['primary']['busy']

            # Slot-Berechnung
            available_slots = self._calculate_efficient_slots(
                busy_slots,
                self.duration_minutes
            )

            # Format the response for WhatsApp
            return {
                'type': 'list',
                'header': "⏰ Available Time Slots",
                'body': "Tap to select a time slot",
                'footer': "Time Slot Finder",
                'sections': [{
                    'title': "Next 5 Available Slots",
                    'rows': [{
                        'id': f"slot_{slot['start'].timestamp()}",
                        'title': f"🕒 {slot['start'].strftime('%H:%M')} - {slot['end'].strftime('%H:%M')}",
                        'description': f"Duration: {slot['duration']}"
                    } for slot in available_slots[:5]]
                }]
            }
        except Exception as e:
            return f"⚠️ Error finding time slots: {str(e)}"

    def _calculate_efficient_slots(self, busy_slots, duration_minutes):
        """Effiziente Slot-Berechnung"""
        available_slots = []
        current = datetime.now(UTC)
        end_time = current + timedelta(days=1)

        while current < end_time:
            slot_end = current + timedelta(minutes=duration_minutes)

            if slot_end > end_time:
                break

            is_available = all(
                slot_end <= parser.parse(busy['start']) or
                current >= parser.parse(busy['end'])
                for busy in busy_slots
            )

            if is_available:
                available_slots.append({
                    'start': current,
                    'end': slot_end,
                    'duration': f"{duration_minutes} min"
                })
                current = slot_end
            else:
                current += timedelta(minutes=15)

        return available_slots

    async def handle_calendar_actions(self, message):
        """Handle calendar-related pending actions"""
        user_state = self.pending_actions.get(self.whc.progress_messenger0.recipient_phone, {})

        if user_state.get('type') == 'create_event':
            return await self._handle_event_creation(message, user_state)

        return None

    async def _handle_event_creation(self, message, state):
        step = state['step']
        event_data = state['event_data']

        if step == 'title':
            event_data['summary'] = message.content
            state['step'] = 'start_time'
            return "📅 When should it start? (e.g., 'tomorrow 2pm' or '2024-03-20 14:30')"

        elif step == 'start_time':
            event_data['start'] = self._parse_time(message.content)
            state['step'] = 'end_time'
            return "⏰ When should it end? (e.g., '3pm' or '2024-03-20 15:30')"

        elif step == 'end_time':
            event_data['end'] = self._parse_time(message.content, reference=event_data['start'])
            state['step'] = 'description'
            return "📝 Add a description (or type 'skip')"

        elif step == 'description':
            if message.content.lower() != 'skip':
                event_data['description'] = message.content
            state['step'] = 'confirm_envet'
            return self._create_confirmation_message(event_data)

    def _format_calendar_response(self, events, title):
        """Enhanced calendar formatting with interactive support"""
        if not events:
            return f"📅 No {title.lower()} found"

        return {
            'type': 'list',
            'header': title,
            'body': "Tap to view event details",
            "footer": "-- Calendar --",
            'sections': [{
                'title': f"{len(events)} Events",
                'rows': [{
                    'id': f"event_{event['id']}",
                    'title': f"📅 {event['summary']}"[:23],
                    'description': self._format_event_time(event)[:45]
                } for event in events[:5]]
            }]
        }

    def _parse_iso_to_readable(self, iso_str):
        """Convert ISO datetime to readable format"""
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime("%a %d %b %Y %H:%M")

    def _parse_time(self, time_str, reference=None):
        """
        Konvertiert natürliche Sprache zu präziser Datetime

        Unterstützt:
        - 'heute'
        - 'morgen'
        - 'in einer woche'
        - '10 uhr'
        - '10pm'
        - 'nächsten montag'
        """
        if reference is None:
            reference = datetime.now()

        try:
            import dateparser

            # Dateparser für flexibel Zeitparsing
            parsed_time = dateparser.parse(
                time_str,
                settings={
                    'PREFER_DATES_FROM': 'future',
                    'RELATIVE_BASE': reference,
                    'TIMEZONE': 'Europe/Berlin'
                }
            )

            if parsed_time is None:
                # Fallback auf dateutil wenn dateparser scheitert
                parsed_time = parser .parse(time_str, fuzzy=True, default=reference)

            return parsed_time

        except Exception as e:
            print(f"Zeitparsing-Fehler: {e}")
            return reference

    def _calculate_free_slots(self, start, end, busy_slots):
        """Calculate free time slots between busy periods"""
        # Implementation would calculate available windows
        return [{
            'start': "09:00",
            'end': "11:00",
            'duration': "2 hours"
        }]

    def _create_confirmation_message(self, event_data):
        """Create event confirmation message"""
        details = [
            f"📌 Title: {event_data['summary']}",
            f"🕒 Start: {self._parse_iso_to_readable(event_data['start'])}",
            f"⏰ End: {self._parse_iso_to_readable(event_data['end'])}",
            f"📝 Description: {event_data.get('description', 'None')}"
        ]
        return {
            'type': 'quick_reply',
            'text': "\n".join(details),
            'options': {'confirm': '✅ Confirm', 'cancel': '❌ Cancel'}
        }

    def _create_calendar_event(self, event_data):
        """Create event through Calendar API"""
        event = {
            'summary': event_data['summary'],
            'start': {'dateTime': event_data['start']},
            'end': {'dateTime': event_data['end']},
        }
        if 'description' in event_data:
            event['description'] = event_data['description']

        return self.calendar_service.events().insert(
            calendarId='primary',
            body=event
        ).execute()

    async def system_status(self, message):
        o = (datetime.now() - self.start_time)
        o.microseconds = 0
        status = {
            "🤖 Agent": "Online" if self.agent else "Offline",
            "📧 Email": "Connected" if self.gmail_service else "Disconnected",
            "📅 Calendar": "Connected" if self.calendar_service else "Disconnected",
            "📄 Documents": "Connected" if self.blob_docs_system else "Disconnected",
            "⏳ Uptime": f"{str(o.isoformat())}"
        }
        return "\n".join([f"{k}: {v}" for k, v in status.items()])

    async def restart_system(self, message):
        message.reply("🔄 System restart initiated...")
        time.sleep(1)
        await self.clear_memory(message)
        time.sleep(1)
        return  "✅ System restarted"

    # Updated document handlers
    async def list_documents(self, message, filter_type=None):
        docs = self.blob_docs_system.list_documents(filter_type)
        if len(docs) == 0:
            return "No docs found"
        else:
            return str(docs)
        return {
            'type': 'list',
            'body': 'Stored Documents',
            'action': {
                'sections': [{
                    'title': 'Your Documents',
                    'rows': [{
                        'id': doc['id'],
                        'title': f"{self._get_icon(doc['type'])} {doc['name']}"[:23],
                        'description': f"{doc['type'].title()} | {self._format_size(doc['size'])} | {doc['modified']}"[:29]
                    } for doc in docs[:10]]
                }]}
        }

    async def start_document_upload(self, message):
        """Initiate document upload workflow"""
        self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {'type': 'document', 'step': 'awaiting_file'}
        return {
            'type': 'quick_reply',
            'text': '📤 Send me the file you want to upload',
            'options': {'cancel': '❌ Cancel Upload'}
        }

    async def search_documents(self, message):
        """Initiate document search workflow"""
        self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {'type': 'search', 'step': 'awaiting_query'}
        return {
            'type': 'quick_reply',
            'text': '🔍 What are you looking for?',
            'options': {'cancel': '❌ Cancel Search'}
        }

    async def handle_media_message(self, message: 'Message'):
        """Handle document/image/video uploads"""
        user_state = self.pending_actions.get(self.whc.progress_messenger0.recipient_phone, {})

        if user_state.get('step') == 'awaiting_file':
            file_type = message.type
            if file_type not in ['document', 'image', 'video']:
                return "Unsupported file type"

            try:
                # Download media
                #media_url = message.document.url if hasattr(message, 'document') else \
                #    message.image.url if hasattr(message, 'image') else \
                #        message.video.url
                if file_type =='video':
                    content = self.whc.messenger.get_video(message.data)
                if file_type =='image':
                    content = self.whc.messenger.get_image(message.data)
                if file_type =='document':
                    content = self.whc.messenger.get_document(message.data)
                print("Media content:", content)
                media_data = self.whc.messenger.download_media(media_url=self.whc.messenger.query_media_url(media_id=content.get('id')),  mime_type=content.get('mime_type'), file_path='.data/temp')
                print("Media media_data:", media_data)
                # Save to blob storage
                filename = f"file_{file_type}_{datetime.now().isoformat()}_{content.get('sha256', '')}"
                blob_id = self.blob_docs_system.save_document(
                    open(media_data, 'rb').read(),
                    filename=filename,
                    file_type=file_type
                )

                self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {}
                return f"✅ File uploaded successfully!\nID: {blob_id}"

            except Exception as e:
                logging.error(f"Upload failed: {str(e)}")
                return f"❌ Failed to upload file Error : {str(e)}"

        return "No pending uploads"

    async def delete_document(self, message):
        """Delete document workflow"""
        docs = self.blob_docs_system.list_documents()
        return {
            'type': 'quick_reply',
            'text': 'Select document to delete:',
            'options': {doc['id']: doc['name'] for doc in docs[:5]},
            'handler': self._confirm_delete
        }

    async def _confirm_delete(self, doc_id, message):
        """Confirm deletion workflow"""
        doc = next((d for d in self.blob_docs_system.list_documents() if d['id'] == doc_id), None)
        if not doc:
            return "Document not found"

        if self.blob_docs_system.delete_document(doc_id):
            return f"✅ {doc['name']} deleted successfully"
        return "❌ Failed to delete document"

    # Helper methods
    def _get_icon(self, file_type: str) -> str:
        icons = {
            'document': '📄',
            'image': '🖼️',
            'video': '🎥'
        }
        return icons.get(file_type, '📁')

    def _format_size(self, size: int) -> str:
        if size < 1024:
            return f"{size}B"
        elif size < 1024 ** 2:
            return f"{size / 1024:.1f}KB"
        elif size < 1024 ** 3:
            return f"{size / (1024 ** 2):.1f}MB"
        return f"{size / (1024 ** 3):.1f}GB"

    # Utility Methods

    def _clean_processed_messages(self):
        """Clean old messages from processed cache"""
        now = time.time()
        self.processed_messages = {
            msg_id for msg_id, timestamp in self.processed_messages
            if now - timestamp < 3600  # 1 hour retention
        }

    def send_email(self, to, subject, body):
        """Actual email sending function to be called by agent"""
        if not self.gmail_service:
            return False

        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        self.gmail_service.users().messages().send(
            userId='me',
            body={'raw': encoded_message}
        ).execute()
        return True

    async def start_agent(self, *a):
        """Start the agent in background mode"""
        if self.agent:
            self.agent.run_in_background()
            return True
        return False

    async def stop_agent(self, *b):
        """Stop the currently running agent"""
        if self.agent:
            self.agent.stop()
            return True
        return False

    async def show_task_stack(self, *a):
        """Display current task stack"""
        if self.agent and len(self.agent.taskstack.tasks) > 0:
            tasks = self.agent.taskstack.tasks
            return self.agent.mini_task("\n".join([f"Task {t.id}: {t.description}" for t in tasks]), "system", "Format to nice and clean whatsapp format")
        return "No tasks in stack"

    def run(self):
        """Start the WhatsApp assistant"""
        try:
            self.state = AssistantState.ONLINE
            # Send welcome message

            mas = self.whc.messenger.create_message(
                content="Digital Assistant is online! Send /help for available commands.",to=self.whc.progress_messenger0.recipient_phone,
            ).send(sender=0)
            mas_id = mas.get("messages", [{}])[0].get("id")
            print(mas_id)

        except Exception as e:
            logging.error(f"Assistant error: {str(e)}")
            self.state = AssistantState.OFFLINE
            raise

    async def handle_agent_actions(self, message):
        user_state = self.pending_actions.get(self.whc.progress_messenger0.recipient_phone, {})
        def helper():

            stop_flag = threading.Event()
            try:
                progress = self.progress_messengers['task']
                # message_id = progress.send_initial_message(mode="loading")
                progress.message_id = message.id
                progress.start_loading_in_background(stop_flag)
                res = message.content
                print(message.data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {}).get('messages', [{}])[0].get(
                    'context'))
                if context := message.data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {}).get('messages', [{}])[0].get(
                    'context'):
                    context_str = f"Context : source {'USER' if context.get('from') in self.whc.progress_messenger0.recipient_phone else 'AGENT'}"
                    cd = self.history.get(context.get('id'))
                    context_str += "\n" + (cd if cd is not None else "The ref Message is not in the history")
                    res += "\n" + context_str
                if user_state.get('type') == 'system':
                    res = self.isaa.run(res)
                    self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {}
                elif user_state.get('type') == 'self-agent':
                    res = self.agent.run(res)
                    self.pending_actions[self.whc.progress_messenger0.recipient_phone] = {}
                self.agent.mode = LLMMode(
                    name="Chatter",
                    description="whatsapp Chat LLM",
                    system_msg="Response precise and short style using whatsapp syntax!",
                    post_msg=None
                )
                response = self.agent.mini_task(res, "user", persist=True)
                self.save_reply(message, response)
            except Exception as e:
                stop_flag.set()
                message.reply("❌ Error in agent "+str(e))
            finally:
                self.agent.mode = None
                stop_flag.set()
        threading.Thread(target=helper, daemon=True).start()

    def save_reply(self, message, content):
        res = message.reply(content)
        res_id = res.get("messages", [{}])[0].get("id")
        if res_id is not None:
            self.history.set(res_id, content)
        else:
            print(f"No ID to add to history: {res}")

def connect(app, phone_number_id):
    app.config_fh.one_way_hash(phone_number_id, "WhatsappAppManager",
                                     AppManager.pepper)

    messenger, s_callbacks = AppManager().online("main")

    emoji_set_loading = ["🔄", "🌀", "⏳", "⌛", "🔃"]  # Custom Loading Emoji Set
    progress_messenger0 = ProgressMessenger(messenger, "", emoji_set=emoji_set_loading)
    progress_messenger1 = ProgressMessenger(messenger, "", emoji_set=emoji_set_thermometer)
    progress_messenger2 = ProgressMessenger(messenger, "", emoji_set=emoji_set_work_phases)
    whc = WhClient(messenger=messenger,
                   s_callbacks=s_callbacks,
                   disconnect=AppManager().offline("main"),
                   progress_messenger0=progress_messenger0,
                   progress_messenger1=progress_messenger1,
                   progress_messenger2=progress_messenger2,
                   to="",
                   set_to=lambda :None
                   )

    def set_to(to:str):
        progress_messenger0.recipient_phone = to
        progress_messenger1.recipient_phone = to
        progress_messenger2.recipient_phone = to
        whc.to = to

    whc.set_to = set_to


    #step_flag = threading.Event()
    #message_id = progress_messenger0.send_initial_message(mode="progress")
    #print(progress_messenger0.max_steps)
    #progress_messenger0.start_progress_in_background(step_flag=step_flag)
    #for i in range(progress_messenger0.max_steps):
    #    time.sleep(2)
    #    step_flag.set()
    # Simulate work, then stop loading
    # time.sleep(10)  # Simulate work duration
    # stop_flag.set()

    # stop_flag = threading.Event()
    # message_id = progress_messenger0.send_initial_message(mode="loading")
    # progress_messenger0.start_loading_in_background(stop_flag)

    # Simulate work, then stop loading
    # time.sleep(10)  # Simulate work duration
    # stop_flag.set()

    return whc

def runner(app, phone_number_id, to):

    whc = connect(app, phone_number_id)
    whc.set_to(to)
    # setup
    isaa = app.get_mod("isaa")

    self_agent = isaa.get_agent("self")

    waa = WhatsAppAssistant(whc=whc, isaa=isaa, agent=self_agent, credentials=None)
    whc.s_callbacks(waa.handle_message, print)
    waa.run()

    return waa

# @dataclass is full inplent do not tuch only for help !!!
    # class WhClient:
    #     messenger: WhatsApp
    #     disconnect: Callable
    #     s_callbacks: Callable
    #     progress_messenger0: ProgressMessenger
    #     progress_messenger1: ProgressMessenger
    #     progress_messenger2: ProgressMessenger


    # @dataclass
    # class Task:
    #     id: str
    #     description: str
    #     priority: int
    #     estimated_complexity: float  # Range 0.0 to 1.0
    #     time_sensitivity: float  # Range 0.0 to 1.0
    #     created_at: datetime

    #   class TaskStack:
    #     def __init__(self):
    #         self.tasks: List[Task] = []
    #         self.current_task: Optional[Task] = None
    #         ...
    #
    #     def add_task(self, task: Task):
    #         ...
    #
    #     def _sort_tasks(self):
    #         ....
    #
    #     def get_next_task(self) -> Optional[Task]:
    #         ...
    #
    #     def remove_task(self, task_id: str):
    #         ...
    #
    #     def emtpy(self):
    #         return len(self.tasks) == 0


    #  class AgentState(Enum):
    #    IDLE = "idle"
    #    RUNNING = "running"
    #    STOPPED = "stopped"

    # @dataclass
    # class TaskStatus:
    #     task_id: str
    #     status: str  # queued, running, completed, error
    #     progress: float  # Range 0.0 to 1.0
    #     result: Optional[Any] = None
    #     error: Optional[str] = None

    # set up base client


"""


    @dataclass
    class Agent:
        amd: AgentModelData = field(default_factory=AgentModelData)

        stream: bool = field(default=False) # set Flase
        messages: List[Dict[str, str]] = field(default_factory=list)

        max_history_length: int = field(default=10)
        similarity_threshold: int = field(default=75)

        verbose: bool = field(default=logger.level == logging.DEBUG) # must be Tro for Clabbacks ( print_verbose )

        stream_function: Callable[[str], bool or None] = field(default_factory=print) # (live strem callback do not use ..

        taskstack: Optional[TaskStack] = field(default_factory=TaskStack)
        status_dict: Dict[str, TaskStatus] = field(default_factory=dict)
        state: AgentState = AgentState.IDLE
        world_model: Dict[str, str] = field(default_factory=dict)

        post_callback: Optional[Callable] = field(default=None) # gets calls d wit the final result str
        progress_callback: Optional[Callable] = field(default=None) # gets onlled wit an status object

        mode: Optional[LLMMode or ModeController] = field(default=None) # add an inteface to sent the modes ( isaa controller for modes :  @dataclass
                                            class ControllerManager:
                                                controllers: Dict[str, ModeController] = field(default_factory=dict)

                                                def rget(self, llm_mode: LLMMode, name: str = None):
                                                    if name is None:
                                                        name = llm_mode.name
                                                    if not self.registered(name):
                                                        self.add(name, llm_mode)
                                                    return self.get(name)

                                                def list_names(self):
                                                    return list(self.controllers.keys())
            # avalabel with isaa.controller

        last_result: Optional[Dict[str, Any]] = field(default=None) # opionl

        def show_world_model(self):
            if not self.world_model:
                return "balnk"
            return "Key <> Value\n" + "\n".join([f"{k} <> {v}" for k, v in self.world_model.items()])

        def flow_world_model(self, query):
                #.... add dircet information to the agents word modell
                pass

        def run_in_background(self):
            ""Start a task in background mode""
           # ... start fuction of the agent non blocking !

        def stop(self):
            self._stop_event.set()
            # dos not sop instante


        def run(self, user_input_or_task: str or Task, with_memory=None, with_functions=None, max_iterations=3, **kwargs):
            # ... entry pont for the Agent, uses callbacks  self.progress_callback with an status object
            # dos not send the final completet stae only inf run in backrund not a dirct task from the user -> returns agent autup as string / noot good for dircte viw for the user ..
          # ...
            #update_progress()

            #self.status_dict[task.id].status = "completed"
            #self.status_dict[task.id].result = out
            #self.status_dict[task.id].progress = 1.0
            #return out

        def _to_task(self, query:str) -> Task:
            ## creat a task from a str input
            # return task

        def invoke(self, user_input, with_mem=False, max_iterations=3, **kwargs):
            # run in decret agentcy mode + sepace mode for the user to utilyse
            #except Exception as e:
            #    return str(e).split("Result:")[-1]

        def mini_task(self, user_task, task_from="user", mini_task=None, message=None):
            ## mini task fro the agent sutch as refactoring the autu to whastapp syle syntax and make it perises and optimest for minimlistk conrret relavent chat Asisstatnt

        def format_class(self, format_class, task, re_try=4, gen=None):
           # tasks a BasModel Cass as input and a str string and retus a  full version for the model example :
            prompt = f"Determen if to change the current world model ##{self.show_world_model()}## basd on the new informaiton :" + query

            class WorldModelAdaption(BaseModel):
                ""world model adaption action['remove' or 'add' or ' change' or None] ;
                key from the existing word model or new one ;
                informations changed or added""
                action: Optional[str] = field(default=None)
                key: Optional[str] = field(default=None)
                informations: Optional[str] = field(default=None)

            model_action = self.format_class(WorldModelAdaption, prompt)

        def reset_context(self):
            self.messages = []
            self.world_model = {}


        def reset_memory(self):
            self.content_memory.text = ""
            self.messages = []

"""

    # long runing tasks
    # config stop task
    #   list lask stak , stask list
    #   reoderd / cancel / edit tasks
    #   stop agent | run in background
    #  toggels spesific insights from task execution

    # Emails
    #   Summary of N Keywort
    #   Last 5 Emals /
    #   Send email with attachment

    # Docs
    #   input .txt .pdf. png .jpg .csv .mp3 .mp4
    #   save and add or remove from agent knolage base
    #   list in nive form
    #   agent space for interaction

    # Kalender
    #   list day scope
    #   show nex event
    #   add event
    #   remove events
    #   agent get-, add-, list-events

    # System
    #   Metrics
    #   Online since

    # stt speech to text ->
        #     talk_generate = app.run_any(TBEF.AUDIO.STT_GENERATE,
#                                 model="openai/whisper-small",
#                                 row=True, device=1)
# audio_data: bytes =
#             text: str = talk_generate(audio_data)['text']

    # tts ->
    # filepaths: bytes = app.run_any(TBEF.AUDIO.SPEECH, text="hallo das ist ein test", voice_index=0,
#                                             use_cache=False,
#                                             provider='piper',
#                                             config={'play_local': False, 'model_name': 'kathleen'},
#                                             local=False,
#                                             save=True) # simply convert  to str ic neddet witch .decode()

    # give tasks
    #   inputs text audio docs
    #
    #   commands /config /emails /Kalender /docs /System
    #   flows -> config Interaction Button
    #               - Edit Agent | Edit Tasks
    #               - start stop agent clear temp memory | show Taskstakc (edit, roder, remove singel tasks) , show status_dict (show one task as progress par ( toogel on / off ) show result in a nice format mardown auto to -> whatsapp
    #               - crate new agent / switch bewenn agent and (dircht call isaa.run) for norma tasks
    #   flows -> emails Interaction Button
    #               - send | recive
    #               - send with all detais and atachemts | filter for (Time, context, emaladress) -> convert to summary send as audio / gent spesific insigt and send location / get sepsific insind and save event to calender + user aproval !

    #   flows -> Kalender Interaction Button
    #               - Get latest evnt and send -> location -> summary , audio summary
    #               - gent Day / nDays overwiy and send -> summary , audio summary
    #               - find optimual spoot for x event using perrafances and exsiting events
    #               - adding an event
    #   flows -> docs Interaction Button
    #               - add remove view
    #   flows -> System Interaction Button
    #               - show Nice Texte view , restart
    #   flows -> Default text or audio recived
    #               - tirgger active agent or (issa.run)


    # return results
    # message = Message(instance=yourclient, id="MESSAGEID")
    #   - make as red
    #       - to see if isystem is online
    #            message.mark_as_read()
    #   - react to user massage
    #       message.react("👍")
    #   - show inital welcome and explantion
    #       message.send_button
    #   - reply to user massage
    #       message.reply("Hello world!")
    #   - reply to agent massage
    #       message.reply("Hello world!")
    #   - show progrss bar
    #       step_flag = threading.Event()
    #     message_id = progress_messenger0.send_initial_message(mode="progress")
    #     print(progress_messenger0.max_steps)
    #     progress_messenger0.start_progress_in_background(step_flag=step_flag)
    #     for i in range(progress_messenger0.max_steps):
    #         time.sleep(2)
    #         step_flag.set()
    #   - shwo loding state
    #        # stop_flag = threading.Event()
    #     message_id = progress_messenger0.send_initial_message(mode="loading")
    #     # progress_messenger0.start_loading_in_background(stop_flag)
    #
    #     # Simulate work, then stop loading
    #     # time.sleep(10)  # Simulate work duration
    #     # stop_flag.set()
    #   - send audio
    #       messenger.send_audio(
    #         audio="https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
    #         recipient_id="255757xxxxxx",
    #         sender=0,
    #     )
    #   - send image
    #       media_id = messenger.upload_media(
    #         media='path/to/media',
    #     )['id']
    # >>> messenger.send_image(
    #         image=media_id,
    #         recipient_id="255757xxxxxx",
    #         link=False
    #         sender=0,
    #     )
    #   - send location
    #       messenger.send_location(
    #         lat=1.29,
    #         long=103.85,
    #         name="Singapore",
    #         address="Singapore",
    #         recipient_id="255757xxxxxx",
    #         sender=0,
    #     )

    # exaple button .send_button(
    #         recipient_id="255757xxxxxx",
    #         button={
    #             "header": "Header Testing",
    #             "body": "Body Testing",
    #             "footer": "Footer Testing",
    #             "action": {
    #                 "button": "Button Testing",
    #                 "sections": [
    #                     {
    #                         "title": "iBank",
    #                         "rows": [
    #                             {"id": "row 1", "title": "Send Money", "description": ""},
    #                             {
    #                                 "id": "row 2",
    #                                 "title": "Withdraw money",
    #                                 "description": "",
    #                             },
    #                         ],
    #                     }
    #                 ],
    #             },
    #         },
    #         sender=0,
    #     )


    # tools for the agent :        tools = {**tools, **{
    #
    #             "saveDataToMemory": {"func": ad_data, "description": "tool to save data to memory,"
    #                                                                  " write the data as specific"
    #                                                                  " and accurate as possible."},

