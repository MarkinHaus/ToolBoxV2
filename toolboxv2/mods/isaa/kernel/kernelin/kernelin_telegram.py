"""
ProA Kernel Telegram Interface
================================

Production-ready Telegram interface for the ProA Kernel with:
- Auto-persistence (save/load on start/stop)
- Full media support (photos, documents, voice messages)
- Voice message transcription (Groq Whisper)
- Inline keyboard support
- Per-user agent instances (self-{username})
- Proactive notifications (morning brief, evening summary, reminders)
- Scheduled tasks and deadline tracking
- Unified user mapping with Discord

Installation:
-------------
pip install python-telegram-bot[job-queue] groq

Environment Variables:
----------------------
TELEGRAM_BOT_TOKEN=your_bot_token_here
GROQ_API_KEY=your_groq_api_key (for voice transcription)

Commands:
---------
/start - Initialize bot and register user
/status - Show kernel status
/capture [text] - Quick capture idea/note
/tasks - Show scheduled tasks
/focus [project] - Set current focus/context
/brief - Get morning brief now
/summary - Get evening summary now
/help - Show all commands

Proactive Features:
-------------------
- Morning Brief (configurable time, default 8:00)
- Evening Summary (configurable time, default 22:00)
- Deadline Reminders (48h, 24h, 2h before)
- Task Notifications
- Custom Scheduled Messages
"""

import asyncio
import os
import sys
import time
import json
import tempfile
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from telegram import (
        Update,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        ReplyKeyboardMarkup,
        ReplyKeyboardRemove,
        Message,
        Voice,
        Document,
        PhotoSize
    )
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        ContextTypes,
        filters,
        JobQueue
    )
    from telegram.constants import ParseMode, ChatAction
    TELEGRAM_SUPPORT = True
except ImportError:
    print("⚠️ python-telegram-bot not installed. Install with: pip install python-telegram-bot[job-queue]")
    TELEGRAM_SUPPORT = False

# Check for Groq API (voice transcription)
try:
    from groq import Groq
    GROQ_SUPPORT = True
except ImportError:
    GROQ_SUPPORT = False
    print("⚠️ Groq not installed. Voice transcription disabled. Install with: pip install groq")

# Check for Obsidian tools
OBSIDIAN_SUPPORT = False
try:
    from toolboxv2.mods.isaa.kernel.kernelin.tools.obsidian_tools import ObsidianKernelTools
    OBSIDIAN_SUPPORT = True
except ImportError:
    print("⚠️ Obsidian tools not available")

from toolboxv2 import App, get_app
from toolboxv2.mods.isaa.kernel.instace import Kernel
from toolboxv2.mods.isaa.kernel.types import Signal as KernelSignal, SignalType, KernelConfig, IOutputRouter


# ===== USER AGENT MAPPING =====

@dataclass
class UserAgentMapping:
    """Maps platform user IDs to agent instances"""
    telegram_id: str
    discord_id: Optional[str] = None
    agent_name: str = ""  # e.g., "self-markin"
    display_name: str = ""
    registered_at: float = field(default_factory=time.time)
    preferences: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.agent_name and self.display_name:
            # Auto-generate agent name from display name
            self.agent_name = f"self-{self.display_name.lower().replace(' ', '_')}"


class UserMappingStore:
    """Persistent storage for user-agent mappings"""

    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.mappings: Dict[str, UserAgentMapping] = {}  # telegram_id -> mapping
        self.discord_to_telegram: Dict[str, str] = {}  # discord_id -> telegram_id
        self._load()

    def _load(self):
        """Load mappings from file"""
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    for telegram_id, mapping_data in data.get('mappings', {}).items():
                        self.mappings[telegram_id] = UserAgentMapping(**mapping_data)
                        if mapping_data.get('discord_id'):
                            self.discord_to_telegram[mapping_data['discord_id']] = telegram_id
                print(f"✓ Loaded {len(self.mappings)} user mappings")
            except Exception as e:
                print(f"⚠️ Error loading user mappings: {e}")

    def _save(self):
        """Save mappings to file"""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'mappings': {
                    tid: {
                        'telegram_id': m.telegram_id,
                        'discord_id': m.discord_id,
                        'agent_name': m.agent_name,
                        'display_name': m.display_name,
                        'registered_at': m.registered_at,
                        'preferences': m.preferences
                    }
                    for tid, m in self.mappings.items()
                }
            }
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving user mappings: {e}")

    def register_user(self, telegram_id: str, display_name: str, discord_id: str = None) -> UserAgentMapping:
        """Register or update a user mapping"""
        if telegram_id in self.mappings:
            # Update existing
            mapping = self.mappings[telegram_id]
            if discord_id:
                mapping.discord_id = discord_id
                self.discord_to_telegram[discord_id] = telegram_id
            if display_name:
                mapping.display_name = display_name
        else:
            # Create new
            mapping = UserAgentMapping(
                telegram_id=telegram_id,
                discord_id=discord_id,
                display_name=display_name
            )
            self.mappings[telegram_id] = mapping
            if discord_id:
                self.discord_to_telegram[discord_id] = telegram_id

        self._save()
        return mapping

    def get_by_telegram(self, telegram_id: str) -> Optional[UserAgentMapping]:
        """Get mapping by Telegram ID"""
        return self.mappings.get(telegram_id)

    def get_by_discord(self, discord_id: str) -> Optional[UserAgentMapping]:
        """Get mapping by Discord ID"""
        telegram_id = self.discord_to_telegram.get(discord_id)
        if telegram_id:
            return self.mappings.get(telegram_id)
        return None

    def link_discord(self, telegram_id: str, discord_id: str) -> bool:
        """Link a Discord ID to an existing Telegram user"""
        if telegram_id not in self.mappings:
            return False

        self.mappings[telegram_id].discord_id = discord_id
        self.discord_to_telegram[discord_id] = telegram_id
        self._save()
        return True


# ===== TELEGRAM OUTPUT ROUTER =====

class TelegramOutputRouter(IOutputRouter):
    """Telegram-specific output router"""

    def __init__(self, bot_app: 'Application', groq_client: 'Groq' = None):
        self.bot_app = bot_app
        self.groq_client = groq_client
        self.user_chats: Dict[str, int] = {}  # user_id -> chat_id
        self.active_chats: Dict[int, dict] = {}  # chat_id -> chat_info

    async def send_response(self, user_id: str, content: str, role: str = "assistant", metadata: dict = None):
        """Send agent response to Telegram user"""
        try:
            chat_id = self.user_chats.get(user_id)
            if not chat_id:
                print(f"⚠️ No chat found for user {user_id}")
                return

            # Split long messages (Telegram limit: 4096 chars)
            chunks = self._split_message(content, max_length=4000)

            for i, chunk in enumerate(chunks):
                # Use Markdown formatting
                try:
                    await self.bot_app.bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
                        parse_mode=ParseMode.MARKDOWN
                    )
                except Exception:
                    # Fallback to plain text if markdown fails
                    await self.bot_app.bot.send_message(
                        chat_id=chat_id,
                        text=chunk
                    )

                # Small delay between chunks
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.3)

        except Exception as e:
            print(f"❌ Error sending Telegram response: {e}")

    async def send_notification(self, user_id: str, content: str, priority: int = 5, metadata: dict = None):
        """Send notification to Telegram user"""
        try:
            chat_id = self.user_chats.get(user_id)
            if not chat_id:
                return

            # Add notification emoji based on priority
            if priority >= 8:
                prefix = "🚨 *URGENT*\n\n"
            elif priority >= 6:
                prefix = "⚠️ *Important*\n\n"
            else:
                prefix = "🔔 "

            await self.bot_app.bot.send_message(
                chat_id=chat_id,
                text=prefix + content,
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            print(f"❌ Error sending Telegram notification: {e}")

    async def send_error(self, user_id: str, error: str, metadata: dict = None):
        """Send error message to Telegram user"""
        try:
            chat_id = self.user_chats.get(user_id)
            if not chat_id:
                return

            await self.bot_app.bot.send_message(
                chat_id=chat_id,
                text=f"❌ *Error*\n\n{error}",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            print(f"❌ Error sending Telegram error: {e}")

    async def send_media(self, user_id: str, file_path: str = None, url: str = None,
                        caption: str = None) -> Dict[str, Any]:
        """Send media to Telegram user"""
        try:
            chat_id = self.user_chats.get(user_id)
            if not chat_id:
                return {"success": False, "error": "No chat found"}

            if file_path:
                with open(file_path, 'rb') as f:
                    # Detect file type
                    if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        msg = await self.bot_app.bot.send_photo(
                            chat_id=chat_id,
                            photo=f,
                            caption=caption
                        )
                    elif file_path.lower().endswith(('.mp3', '.ogg', '.wav')):
                        msg = await self.bot_app.bot.send_audio(
                            chat_id=chat_id,
                            audio=f,
                            caption=caption
                        )
                    else:
                        msg = await self.bot_app.bot.send_document(
                            chat_id=chat_id,
                            document=f,
                            caption=caption
                        )
                return {"success": True, "message_id": msg.message_id}

            elif url:
                # Send URL as message
                text = f"📎 {caption}\n{url}" if caption else url
                msg = await self.bot_app.bot.send_message(chat_id=chat_id, text=text)
                return {"success": True, "message_id": msg.message_id}

            return {"success": False, "error": "No file or URL provided"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _split_message(self, content: str, max_length: int = 4000) -> List[str]:
        """Split long message into chunks"""
        if len(content) <= max_length:
            return [content]

        chunks = []
        current = ""

        for para in content.split('\n\n'):
            if len(current) + len(para) + 2 > max_length:
                if current:
                    chunks.append(current.strip())
                current = para
            else:
                current += ('\n\n' if current else '') + para

        if current:
            while len(current) > max_length:
                split_index = current[:max_length].rfind(' ')
                if split_index == -1:
                    split_index = max_length
                chunks.append(current[:split_index].strip())
                current = current[split_index:].strip()
            chunks.append(current.strip())

        return chunks


# ===== PROACTIVE SCHEDULER =====

@dataclass
class ScheduledTask:
    """A scheduled proactive task"""
    id: str
    user_id: str
    task_type: str  # "morning_brief", "evening_summary", "reminder", "deadline"
    scheduled_time: datetime
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    recurring: bool = False
    recurrence_rule: str = ""  # e.g., "daily", "weekly", "weekdays"


class ProactiveScheduler:
    """Manages proactive notifications and scheduled tasks"""

    def __init__(self, kernel: 'TelegramKernel'):
        self.kernel = kernel
        self.tasks: Dict[str, ScheduledTask] = {}
        self.user_schedules: Dict[str, Dict[str, str]] = {}  # user_id -> {morning_time, evening_time}

    def set_user_schedule(self, user_id: str, morning_time: str = "08:00", evening_time: str = "22:00"):
        """Set user's preferred schedule times"""
        self.user_schedules[user_id] = {
            "morning_time": morning_time,
            "evening_time": evening_time
        }

    async def schedule_morning_brief(self, user_id: str):
        """Schedule daily morning brief for user"""
        schedule = self.user_schedules.get(user_id, {"morning_time": "08:00"})
        hour, minute = map(int, schedule["morning_time"].split(":"))

        # Schedule with job queue
        job_queue = self.kernel.bot_app.job_queue

        # Remove existing job if any
        job_name = f"morning_brief_{user_id}"
        current_jobs = job_queue.get_jobs_by_name(job_name)
        for job in current_jobs:
            job.schedule_removal()

        # Schedule new job
        job_queue.run_daily(
            self._send_morning_brief,
            time=datetime.now().replace(hour=hour, minute=minute, second=0).time(),
            data={"user_id": user_id},
            name=job_name
        )
        print(f"✓ Scheduled morning brief for user {user_id} at {hour:02d}:{minute:02d}")

    async def schedule_evening_summary(self, user_id: str):
        """Schedule daily evening summary for user"""
        schedule = self.user_schedules.get(user_id, {"evening_time": "22:00"})
        hour, minute = map(int, schedule["evening_time"].split(":"))

        job_queue = self.kernel.bot_app.job_queue

        # Remove existing job if any
        job_name = f"evening_summary_{user_id}"
        current_jobs = job_queue.get_jobs_by_name(job_name)
        for job in current_jobs:
            job.schedule_removal()

        # Schedule new job
        job_queue.run_daily(
            self._send_evening_summary,
            time=datetime.now().replace(hour=hour, minute=minute, second=0).time(),
            data={"user_id": user_id},
            name=job_name
        )
        print(f"✓ Scheduled evening summary for user {user_id} at {hour:02d}:{minute:02d}")

    async def _send_morning_brief(self, context: ContextTypes.DEFAULT_TYPE):
        """Send morning brief to user"""
        user_id = context.job.data["user_id"]

        try:
            # Generate brief using agent
            brief_prompt = """Generate a morning brief including:
1. A motivational greeting
2. Summary of pending tasks/reminders
3. Today's priorities based on context
4. Weather reminder (generic)
5. One productivity tip

Keep it concise and actionable."""

            signal = KernelSignal(
                type=SignalType.SYSTEM_EVENT,
                id=user_id,
                content=brief_prompt,
                metadata={
                    "interface": "telegram",
                    "proactive": True,
                    "task_type": "morning_brief"
                }
            )
            await self.kernel.kernel.process_signal(signal)

        except Exception as e:
            print(f"❌ Error sending morning brief: {e}")

    async def _send_evening_summary(self, context: ContextTypes.DEFAULT_TYPE):
        """Send evening summary to user"""
        user_id = context.job.data["user_id"]

        try:
            summary_prompt = """Generate an evening summary including:
1. What was accomplished today
2. Pending items for tomorrow
3. Learning/insights from today's interactions
4. Suggestions for tomorrow
5. A positive closing note

Keep it reflective and helpful."""

            signal = KernelSignal(
                type=SignalType.SYSTEM_EVENT,
                id=user_id,
                content=summary_prompt,
                metadata={
                    "interface": "telegram",
                    "proactive": True,
                    "task_type": "evening_summary"
                }
            )
            await self.kernel.kernel.process_signal(signal)

        except Exception as e:
            print(f"❌ Error sending evening summary: {e}")

    async def schedule_reminder(self, user_id: str, content: str, remind_at: datetime,
                               reminder_id: str = None) -> str:
        """Schedule a one-time reminder"""
        if reminder_id is None:
            reminder_id = f"reminder_{user_id}_{int(time.time())}"

        job_queue = self.kernel.bot_app.job_queue

        # Calculate delay
        delay = (remind_at - datetime.now()).total_seconds()
        if delay <= 0:
            delay = 1  # Send immediately if time has passed

        job_queue.run_once(
            self._send_reminder,
            when=delay,
            data={"user_id": user_id, "content": content},
            name=reminder_id
        )

        self.tasks[reminder_id] = ScheduledTask(
            id=reminder_id,
            user_id=user_id,
            task_type="reminder",
            scheduled_time=remind_at,
            content=content
        )

        return reminder_id

    async def _send_reminder(self, context: ContextTypes.DEFAULT_TYPE):
        """Send reminder to user"""
        user_id = context.job.data["user_id"]
        content = context.job.data["content"]

        await self.kernel.output_router.send_notification(
            user_id=user_id,
            content=f"⏰ *Reminder*\n\n{content}",
            priority=7
        )

        # Remove from tasks
        job_name = context.job.name
        if job_name in self.tasks:
            del self.tasks[job_name]

    def get_user_tasks(self, user_id: str) -> List[ScheduledTask]:
        """Get all scheduled tasks for a user"""
        return [t for t in self.tasks.values() if t.user_id == user_id]

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        if task_id not in self.tasks:
            return False

        job_queue = self.kernel.bot_app.job_queue
        jobs = job_queue.get_jobs_by_name(task_id)
        for job in jobs:
            job.schedule_removal()

        del self.tasks[task_id]
        return True


# ===== TELEGRAM KERNEL =====

class TelegramKernel:
    """Telegram-based ProA Kernel with proactive capabilities"""

    def __init__(
        self,
        agent,
        app: App,
        bot_token: str,
        instance_id: str = "telegram",
        auto_save_interval: int = 300
    ):
        """
        Initialize Telegram Kernel

        Args:
            agent: FlowAgent instance
            app: ToolBoxV2 App instance
            bot_token: Telegram bot token from @BotFather
            instance_id: Instance identifier
            auto_save_interval: Auto-save interval in seconds
        """
        if not TELEGRAM_SUPPORT:
            raise ImportError("python-telegram-bot not installed")

        self.agent = agent
        self.app = app
        self.instance_id = instance_id
        self.auto_save_interval = auto_save_interval
        self.running = False
        self.save_path = self._get_save_path()

        # Initialize Groq client for voice transcription
        self.groq_client = None
        if GROQ_SUPPORT:
            groq_api_key = os.getenv('GROQ_API_KEY')
            if groq_api_key:
                self.groq_client = Groq(api_key=groq_api_key)
                print("✓ Groq Whisper enabled for voice transcription")

        # Build Telegram application
        self.bot_app = Application.builder().token(bot_token).build()

        # Initialize output router
        self.output_router = TelegramOutputRouter(self.bot_app, self.groq_client)

        # Initialize kernel
        config = KernelConfig(
            heartbeat_interval=30.0,
            idle_threshold=600.0,
            proactive_cooldown=120.0,
            max_proactive_per_hour=8
        )
        self.kernel = Kernel(
            agent=agent,
            config=config,
            output_router=self.output_router
        )

        # User mapping store
        self.user_mapping = UserMappingStore(self.save_path.parent / "user_mappings.json")

        # Proactive scheduler
        self.proactive_scheduler = ProactiveScheduler(self)

        # Initialize Obsidian tools if vault path configured
        self.obsidian_tools = None
        vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
        if vault_path and OBSIDIAN_SUPPORT:
            vault_path_obj = Path(vault_path)
            if vault_path_obj.exists():
                self.obsidian_tools = ObsidianKernelTools(vault_path, agent_id="telegram")
                print(f"✓ Obsidian vault connected: {vault_path}")
            else:
                print(f"⚠️ Obsidian vault path does not exist: {vault_path}")
        elif vault_path and not OBSIDIAN_SUPPORT:
            print(f"⚠️ OBSIDIAN_VAULT_PATH set but Obsidian tools not available")

        # Admin whitelist
        self.admin_users: set = set()  # Will be populated with first user

        # Setup handlers
        self._setup_handlers()

        print(f"✓ Telegram Kernel initialized (instance: {instance_id})")
        print(f"  Voice transcription: {'✅' if self.groq_client else '❌'}")
        print(f"  Obsidian vault: {'✅' if self.obsidian_tools else '❌'}")

    def _get_save_path(self) -> Path:
        """Get save file path"""
        save_dir = Path(self.app.data_dir) / 'Agents' / 'kernel' / self.agent.amd.name / 'telegram'
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"telegram_kernel_{self.instance_id}.pkl"

    def _setup_handlers(self):
        """Setup Telegram command and message handlers"""

        # Command handlers
        self.bot_app.add_handler(CommandHandler("start", self._cmd_start))
        self.bot_app.add_handler(CommandHandler("help", self._cmd_help))
        self.bot_app.add_handler(CommandHandler("status", self._cmd_status))
        self.bot_app.add_handler(CommandHandler("capture", self._cmd_capture))
        self.bot_app.add_handler(CommandHandler("tasks", self._cmd_tasks))
        self.bot_app.add_handler(CommandHandler("focus", self._cmd_focus))
        self.bot_app.add_handler(CommandHandler("brief", self._cmd_brief))
        self.bot_app.add_handler(CommandHandler("summary", self._cmd_summary))
        self.bot_app.add_handler(CommandHandler("remind", self._cmd_remind))
        self.bot_app.add_handler(CommandHandler("schedule", self._cmd_schedule))
        self.bot_app.add_handler(CommandHandler("link", self._cmd_link))
        self.bot_app.add_handler(CommandHandler("reset", self._cmd_reset))
        self.bot_app.add_handler(CommandHandler("context", self._cmd_context))

        # Obsidian commands
        self.bot_app.add_handler(CommandHandler("note", self._cmd_note))
        self.bot_app.add_handler(CommandHandler("vsearch", self._cmd_vsearch))
        self.bot_app.add_handler(CommandHandler("vault", self._cmd_vault_stats))
        self.bot_app.add_handler(CommandHandler("daily", self._cmd_daily))
        self.bot_app.add_handler(CommandHandler("vault_config", self._cmd_vault_config))

        # Message handlers
        self.bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        self.bot_app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))
        self.bot_app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        self.bot_app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))

        # Callback query handler (for inline keyboards)
        self.bot_app.add_handler(CallbackQueryHandler(self._handle_callback))

        # Error handler
        self.bot_app.add_error_handler(self._error_handler)

    # ===== COMMAND HANDLERS =====

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command - Register user"""
        user = update.effective_user
        chat_id = update.effective_chat.id
        user_id = str(user.id)

        # Register user
        mapping = self.user_mapping.register_user(
            telegram_id=user_id,
            display_name=user.first_name or user.username or "User"
        )

        # Register chat for output
        self.output_router.user_chats[user_id] = chat_id

        # First user becomes admin
        if not self.admin_users:
            self.admin_users.add(user_id)
            print(f"🔒 First user {user.first_name} ({user_id}) registered as admin")

        # Schedule proactive features
        await self.proactive_scheduler.schedule_morning_brief(user_id)
        await self.proactive_scheduler.schedule_evening_summary(user_id)

        # Welcome message
        welcome = f"""🤖 *Welcome to ToolBox Brain, {user.first_name}!*

I'm your personal AI assistant with proactive capabilities.

*Your Profile:*
• Agent: `{mapping.agent_name}`
• Telegram ID: `{user_id}`

*What I can do:*
• 💬 Chat naturally - just send me a message
• 🎤 Process voice messages
• 📸 Analyze images
• ⏰ Send reminders and notifications
• 🌅 Morning briefs (8:00)
• 🌙 Evening summaries (22:00)

*Quick Commands:*
/capture [text] - Quick capture
/tasks - View scheduled tasks
/remind [time] [text] - Set reminder
/help - Full command list

Let's get started! What can I help you with?"""

        await update.message.reply_text(welcome, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """📚 *ToolBox Brain Commands*

*Basic:*
/start - Initialize and register
/status - Show kernel status
/help - This help message

*Capture & Focus:*
/capture [text] - Quick capture idea/note
/focus [project] - Set current focus
/context - Show your context/memory

*Scheduling:*
/tasks - Show scheduled tasks
/remind [time] [text] - Set reminder
  Examples:
  `/remind 30m Check email`
  `/remind 2h Call back`
  `/remind 14:00 Meeting`
/schedule - Manage proactive schedule

*Proactive:*
/brief - Get morning brief now
/summary - Get evening summary now

*Obsidian Vault:*
/capture [text] - Quick capture to daily note
/note Title | Content - Create new note
/vsearch [query] - Search vault
/vault - Vault statistics
/daily [date] - Get daily note
/vault\\_config - Show configuration

*Account:*
/link [discord_id] - Link Discord account
/reset - Reset your data

*Tips:*
• Send voice messages for transcription
• Send images for analysis
• Just chat naturally for anything else!"""

        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        user_id = str(update.effective_user.id)
        status = self.kernel.to_dict()
        mapping = self.user_mapping.get_by_telegram(user_id)

        status_text = f"""🤖 *Kernel Status*

*State:* {status['state']}
*Running:* {'✅' if status['running'] else '❌'}

*Your Profile:*
• Agent: `{mapping.agent_name if mapping else 'Not registered'}`
• Discord Linked: {'✅' if mapping and mapping.discord_id else '❌'}

*Metrics:*
• Signals Processed: {status['metrics']['signals_processed']}
• Memories: {status['memory']['total_memories']}
• Learning Records: {status['learning']['total_records']}
• Scheduled Tasks: {len(self.proactive_scheduler.get_user_tasks(user_id))}

*Capabilities:*
• Voice Transcription: {'✅' if self.groq_client else '❌'}
• Proactive Notifications: ✅"""

        await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_capture(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /capture command - Quick capture (to Obsidian if configured, else to kernel)"""
        user_id = str(update.effective_user.id)

        if not context.args:
            help_text = "💡 *Quick Capture*\n\nUsage: `/capture Your idea or note here #optional #tags`"
            if self.obsidian_tools:
                help_text += "\n\n📝 This adds an entry to today's Daily Note in Obsidian."
            await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
            return

        content = " ".join(context.args)

        # If Obsidian is configured, capture to vault
        if self.obsidian_tools:
            result = await self.obsidian_tools.capture(content)
            if result["success"]:
                tags_str = " ".join([f"`#{t}`" for t in result["tags"]]) if result["tags"] else ""
                msg = f"✅ *Captured!*\n\n_{result['captured']}_"
                if tags_str:
                    msg += f"\n\nTags: {tags_str}"
                msg += f"\n📄 `{result['daily_note']}`"
                await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
            else:
                await update.message.reply_text(f"❌ Capture failed: {result.get('error')}")
            return

        # Fallback: Send to kernel as capture signal
        signal = KernelSignal(
            type=SignalType.USER_INPUT,
            id=user_id,
            content=f"[CAPTURE] {content}",
            metadata={
                "interface": "telegram",
                "capture": True,
                "timestamp": time.time()
            }
        )

        await self.kernel.process_signal(signal)
        await update.message.reply_text(f"✅ Captured: _{content}_", parse_mode=ParseMode.MARKDOWN)

    async def _cmd_tasks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tasks command"""
        user_id = str(update.effective_user.id)
        tasks = self.proactive_scheduler.get_user_tasks(user_id)

        if not tasks:
            await update.message.reply_text("📋 No scheduled tasks.\n\nUse `/remind` to create reminders!")
            return

        task_text = "📋 *Your Scheduled Tasks*\n\n"
        for task in sorted(tasks, key=lambda t: t.scheduled_time):
            time_str = task.scheduled_time.strftime("%Y-%m-%d %H:%M")
            task_text += f"• `{task.task_type}` - {time_str}\n  {task.content[:50]}...\n\n"

        await update.message.reply_text(task_text, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_focus(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /focus command"""
        user_id = str(update.effective_user.id)

        if not context.args:
            # Show current focus
            current_focus = "Not set"
            if hasattr(self.kernel.agent, 'variable_manager'):
                current_focus = self.kernel.agent.variable_manager.get(
                    f'user.{user_id}.current_focus'
                ) or "Not set"

            await update.message.reply_text(
                f"🎯 *Current Focus:* {current_focus}\n\n"
                "Set focus with: `/focus ProjectName`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        focus = " ".join(context.args)

        # Store focus in agent variables
        if hasattr(self.kernel.agent, 'variable_manager'):
            self.kernel.agent.variable_manager.set(f'user.{user_id}.current_focus', focus)

        await update.message.reply_text(f"🎯 Focus set to: *{focus}*", parse_mode=ParseMode.MARKDOWN)

    async def _cmd_brief(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /brief command - Get morning brief now"""
        user_id = str(update.effective_user.id)

        await update.message.reply_text("☀️ Generating your brief...")

        # Trigger morning brief
        await self.proactive_scheduler._send_morning_brief(
            type('Context', (), {'job': type('Job', (), {'data': {'user_id': user_id}})()})()
        )

    async def _cmd_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /summary command - Get evening summary now"""
        user_id = str(update.effective_user.id)

        await update.message.reply_text("🌙 Generating your summary...")

        await self.proactive_scheduler._send_evening_summary(
            type('Context', (), {'job': type('Job', (), {'data': {'user_id': user_id}})()})()
        )

    async def _cmd_remind(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /remind command"""
        user_id = str(update.effective_user.id)

        if len(context.args) < 2:
            await update.message.reply_text(
                "⏰ *Set a Reminder*\n\n"
                "Usage: `/remind [time] [message]`\n\n"
                "Examples:\n"
                "• `/remind 30m Check email`\n"
                "• `/remind 2h Call back`\n"
                "• `/remind 14:00 Team meeting`\n"
                "• `/remind tomorrow Buy groceries`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        time_str = context.args[0].lower()
        content = " ".join(context.args[1:])

        # Parse time
        remind_at = self._parse_time(time_str)
        if not remind_at:
            await update.message.reply_text(
                "❌ Could not parse time. Use formats like:\n"
                "• `30m`, `2h`, `1d`\n"
                "• `14:00`, `tomorrow`"
            )
            return

        # Schedule reminder
        reminder_id = await self.proactive_scheduler.schedule_reminder(
            user_id=user_id,
            content=content,
            remind_at=remind_at
        )

        time_display = remind_at.strftime("%Y-%m-%d %H:%M")
        await update.message.reply_text(
            f"✅ Reminder set!\n\n"
            f"⏰ *When:* {time_display}\n"
            f"📝 *What:* {content}",
            parse_mode=ParseMode.MARKDOWN
        )

    async def _cmd_schedule(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /schedule command - Manage proactive schedule"""
        user_id = str(update.effective_user.id)

        if not context.args:
            # Show current schedule
            schedule = self.proactive_scheduler.user_schedules.get(user_id, {
                "morning_time": "08:00",
                "evening_time": "22:00"
            })

            keyboard = [
                [
                    InlineKeyboardButton("🌅 Change Morning", callback_data=f"schedule_morning_{user_id}"),
                    InlineKeyboardButton("🌙 Change Evening", callback_data=f"schedule_evening_{user_id}")
                ],
                [InlineKeyboardButton("✅ Keep Current", callback_data="schedule_keep")]
            ]

            await update.message.reply_text(
                f"⏰ *Your Proactive Schedule*\n\n"
                f"🌅 Morning Brief: *{schedule['morning_time']}*\n"
                f"🌙 Evening Summary: *{schedule['evening_time']}*\n\n"
                "Tap to change:",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.MARKDOWN
            )
            return

        # Parse schedule update
        if len(context.args) >= 2:
            schedule_type = context.args[0].lower()
            new_time = context.args[1]

            if schedule_type in ["morning", "brief"]:
                self.proactive_scheduler.set_user_schedule(user_id, morning_time=new_time)
                await self.proactive_scheduler.schedule_morning_brief(user_id)
                await update.message.reply_text(f"✅ Morning brief set to *{new_time}*",
                                               parse_mode=ParseMode.MARKDOWN)
            elif schedule_type in ["evening", "summary"]:
                self.proactive_scheduler.set_user_schedule(user_id, evening_time=new_time)
                await self.proactive_scheduler.schedule_evening_summary(user_id)
                await update.message.reply_text(f"✅ Evening summary set to *{new_time}*",
                                               parse_mode=ParseMode.MARKDOWN)

    async def _cmd_link(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /link command - Link Discord account"""
        user_id = str(update.effective_user.id)

        if not context.args:
            await update.message.reply_text(
                "🔗 *Link Discord Account*\n\n"
                "Usage: `/link YOUR_DISCORD_ID`\n\n"
                "Find your Discord ID:\n"
                "1. Enable Developer Mode in Discord settings\n"
                "2. Right-click your name → Copy ID",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        discord_id = context.args[0]

        if self.user_mapping.link_discord(user_id, discord_id):
            await update.message.reply_text(
                f"✅ Discord account linked!\n\n"
                f"Discord ID: `{discord_id}`\n\n"
                "Your context is now shared between Telegram and Discord.",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text("❌ Failed to link account. Try /start first.")

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset command"""
        user_id = str(update.effective_user.id)

        keyboard = [
            [
                InlineKeyboardButton("🗑️ Reset All", callback_data=f"reset_all_{user_id}"),
                InlineKeyboardButton("❌ Cancel", callback_data="reset_cancel")
            ]
        ]

        await update.message.reply_text(
            "⚠️ *Reset Your Data*\n\n"
            "This will delete:\n"
            "• All your memories\n"
            "• Learning preferences\n"
            "• Scheduled tasks\n\n"
            "This cannot be undone!",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )

    async def _cmd_context(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /context command - Show user context"""
        user_id = str(update.effective_user.id)

        # Get user data
        mapping = self.user_mapping.get_by_telegram(user_id)
        memories = self.kernel.memory_store.user_memories.get(user_id, [])
        prefs = self.kernel.learning_engine.preferences.get(user_id)
        tasks = self.proactive_scheduler.get_user_tasks(user_id)

        # Current focus
        current_focus = "Not set"
        if hasattr(self.kernel.agent, 'variable_manager'):
            current_focus = self.kernel.agent.variable_manager.get(
                f'user.{user_id}.current_focus'
            ) or "Not set"

        context_text = f"""🧠 *Your Context*

*Agent:* `{mapping.agent_name if mapping else 'Unknown'}`
*Focus:* {current_focus}

*Data:*
• Memories: {len(memories)}
• Preferences: {'✅ Learned' if prefs else '❌ Not yet'}
• Scheduled Tasks: {len(tasks)}
• Discord Linked: {'✅' if mapping and mapping.discord_id else '❌'}

*Recent Activity:*
"""

        # Add recent memories preview
        if memories:
            recent_memories = list(memories)[-3:]
            for mem_id in recent_memories:
                mem = self.kernel.memory_store.memories.get(mem_id)
                if mem:
                    preview = mem.content[:40] + "..." if len(mem.content) > 40 else mem.content
                    context_text += f"• {mem.memory_type.value}: _{preview}_\n"
        else:
            context_text += "_No memories yet_\n"

        await update.message.reply_text(context_text, parse_mode=ParseMode.MARKDOWN)

    # ===== OBSIDIAN COMMANDS =====

    async def _cmd_note(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Create a new note. Usage: /note Title | Content"""
        if not self.obsidian_tools:
            await update.message.reply_text("❌ Obsidian vault not configured.")
            return

        if not context.args:
            await update.message.reply_text(
                "📝 *Create Note*\n\n"
                "Usage: `/note Title | Optional content`\n\n"
                "Creates a new note in the Inbox folder.",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        # Parse title and content (separated by |)
        full_text = " ".join(context.args)
        if "|" in full_text:
            parts = full_text.split("|", 1)
            title = parts[0].strip()
            content = parts[1].strip() if len(parts) > 1 else ""
        else:
            title = full_text
            content = ""

        result = await self.obsidian_tools.create_note(title, content)

        if result["success"]:
            await update.message.reply_text(
                f"📝 *Note Created*\n\n"
                f"*{result['title']}*\n"
                f"Path: `{result['path']}`",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(f"❌ Failed: {result.get('error')}")

    async def _cmd_vsearch(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Search vault. Usage: /vsearch query"""
        if not self.obsidian_tools:
            await update.message.reply_text("❌ Obsidian vault not configured.")
            return

        if not context.args:
            await update.message.reply_text("💡 Usage: `/vsearch your query here`", parse_mode=ParseMode.MARKDOWN)
            return

        query = " ".join(context.args)
        result = await self.obsidian_tools.search(query)

        if result["success"] and result["results"]:
            msg = f"🔍 *Search: {query}*\n"
            msg += f"Found {result['count']} results\n\n"

            for r in result["results"][:5]:
                snippet = r["snippet"][:80] + "..." if len(r["snippet"]) > 80 else r["snippet"]
                msg += f"• *{r['title']}*\n  `{r['path']}`\n  {snippet}\n\n"

            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(f"🔍 No results for: {query}")

    async def _cmd_vault_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show vault statistics"""
        if not self.obsidian_tools:
            await update.message.reply_text("❌ Obsidian vault not configured.")
            return

        result = await self.obsidian_tools.get_graph_stats()

        if result["success"]:
            stats = result["stats"]
            msg = "📊 *Vault Statistics*\n\n"
            msg += f"📝 Notes: {stats['total_notes']}\n"
            msg += f"🔗 Links: {stats['total_links']}\n"
            msg += f"🏝️ Orphans: {stats['orphan_notes']}\n"
            msg += f"📊 Avg Links: {stats['average_links']:.1f}\n\n"

            if result["top_linked"]:
                msg += "*Most Linked:*\n"
                for n in result["top_linked"][:3]:
                    msg += f"• {n['title']} ({n['backlinks']})\n"
                msg += "\n"

            if result["top_tags"]:
                msg += "*Top Tags:*\n"
                for t in result["top_tags"][:5]:
                    msg += f"• #{t['tag']} ({t['count']})\n"

            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")

    async def _cmd_daily(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get daily note. Usage: /daily or /daily 2024-01-15"""
        if not self.obsidian_tools:
            await update.message.reply_text("❌ Obsidian vault not configured.")
            return

        date_str = context.args[0] if context.args else None
        result = await self.obsidian_tools.get_daily(date_str)

        if result["success"]:
            content = result["content"]
            # Truncate for Telegram (4096 char limit)
            if len(content) > 3500:
                content = content[:3500] + "\n\n_...truncated..._"

            msg = f"📅 *Daily Note*\n\n```\n{content}\n```\n\n_{result['path']}_"

            # Split if still too long
            if len(msg) > 4000:
                await update.message.reply_text(f"📅 *{result['path']}*", parse_mode=ParseMode.MARKDOWN)
                await update.message.reply_text(content[:4000])
            else:
                await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")

    async def _cmd_vault_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show vault configuration"""
        vault_path = os.getenv("OBSIDIAN_VAULT_PATH", "Not configured")

        msg = "⚙️ *Vault Configuration*\n\n"
        msg += f"*Path:* `{vault_path}`\n"
        msg += f"*Status:* {'✅ Connected' if self.obsidian_tools else '❌ Not connected'}\n"
        msg += f"*Support:* {'✅ Available' if OBSIDIAN_SUPPORT else '❌ Not installed'}\n"

        if self.obsidian_tools:
            result = await self.obsidian_tools.get_graph_stats()
            if result["success"]:
                msg += f"*Notes:* {result['stats']['total_notes']}\n"

        msg += "\n*How to Configure:*\n"
        msg += "Set environment variable:\n"
        msg += "`OBSIDIAN_VAULT_PATH=/your/vault/path`"

        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    # ===== MESSAGE HANDLERS =====

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages"""
        user = update.effective_user
        user_id = str(user.id)
        chat_id = update.effective_chat.id
        content = update.message.text

        # Ensure user is registered
        if not self.user_mapping.get_by_telegram(user_id):
            await self._cmd_start(update, context)
            return

        # Register chat
        self.output_router.user_chats[user_id] = chat_id

        # Send typing indicator
        await update.message.chat.send_action(ChatAction.TYPING)

        # Get user context
        mapping = self.user_mapping.get_by_telegram(user_id)
        telegram_context = self._get_telegram_context(update)

        # Inject context
        if hasattr(self.kernel.agent, 'variable_manager'):
            self.kernel.agent.variable_manager.set(
                f'telegram.current_context.{user_id}',
                telegram_context
            )

        # Send to kernel
        signal = KernelSignal(
            type=SignalType.USER_INPUT,
            id=user_id,
            content=content,
            metadata={
                "interface": "telegram",
                "chat_id": chat_id,
                "message_id": update.message.message_id,
                "user_name": user.first_name,
                "agent_name": mapping.agent_name if mapping else None,
                "telegram_context": telegram_context
            }
        )

        await self.kernel.process_signal(signal)

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages - transcribe and process"""
        user = update.effective_user
        user_id = str(user.id)
        chat_id = update.effective_chat.id

        if not self.groq_client:
            await update.message.reply_text("❌ Voice transcription not available.")
            return

        # Register chat
        self.output_router.user_chats[user_id] = chat_id

        await update.message.reply_text("🎤 Transcribing...")

        try:
            # Download voice file
            voice = update.message.voice
            file = await voice.get_file()

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as tmp:
                await file.download_to_drive(tmp.name)
                tmp_path = tmp.name

            # Transcribe
            with open(tmp_path, 'rb') as audio_file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    response_format="json"
                )

            # Clean up
            os.unlink(tmp_path)

            text = transcription.text.strip()

            if not text or len(text) < 2:
                await update.message.reply_text("🤷 Could not transcribe. Please try again.")
                return

            # Show transcription
            await update.message.reply_text(f"📝 _{text}_", parse_mode=ParseMode.MARKDOWN)

            # Send typing for response
            await update.message.chat.send_action(ChatAction.TYPING)

            # Process as message
            mapping = self.user_mapping.get_by_telegram(user_id)
            signal = KernelSignal(
                type=SignalType.USER_INPUT,
                id=user_id,
                content=text,
                metadata={
                    "interface": "telegram",
                    "chat_id": chat_id,
                    "voice_message": True,
                    "agent_name": mapping.agent_name if mapping else None
                }
            )

            await self.kernel.process_signal(signal)

        except Exception as e:
            await update.message.reply_text(f"❌ Error transcribing: {e}")

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo messages"""
        user_id = str(update.effective_user.id)
        chat_id = update.effective_chat.id

        self.output_router.user_chats[user_id] = chat_id

        # Get largest photo
        photo = update.message.photo[-1]
        file = await photo.get_file()

        caption = update.message.caption or "Analyze this image"

        # Download photo
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        await update.message.chat.send_action(ChatAction.TYPING)

        # Send to kernel with image
        signal = KernelSignal(
            type=SignalType.USER_INPUT,
            id=user_id,
            content=f"[IMAGE] {caption}",
            metadata={
                "interface": "telegram",
                "chat_id": chat_id,
                "image_path": tmp_path,
                "has_image": True
            }
        )

        await self.kernel.process_signal(signal)

        # Clean up later
        asyncio.get_event_loop().call_later(60, lambda: os.unlink(tmp_path) if os.path.exists(tmp_path) else None)

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document uploads"""
        user_id = str(update.effective_user.id)
        chat_id = update.effective_chat.id

        self.output_router.user_chats[user_id] = chat_id

        doc = update.message.document
        caption = update.message.caption or f"Process this file: {doc.file_name}"

        await update.message.reply_text(f"📄 Received: _{doc.file_name}_", parse_mode=ParseMode.MARKDOWN)

        # Download if small enough
        if doc.file_size < 10 * 1024 * 1024:  # 10MB limit
            file = await doc.get_file()

            with tempfile.NamedTemporaryFile(suffix=Path(doc.file_name).suffix, delete=False) as tmp:
                await file.download_to_drive(tmp.name)
                tmp_path = tmp.name

            signal = KernelSignal(
                type=SignalType.USER_INPUT,
                id=user_id,
                content=f"[DOCUMENT: {doc.file_name}] {caption}",
                metadata={
                    "interface": "telegram",
                    "document_path": tmp_path,
                    "document_name": doc.file_name,
                    "document_type": doc.mime_type
                }
            )

            await self.kernel.process_signal(signal)
        else:
            await update.message.reply_text("❌ File too large (max 10MB)")

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()

        data = query.data

        if data.startswith("reset_all_"):
            user_id = data.replace("reset_all_", "")

            # Perform reset
            if user_id in self.kernel.memory_store.user_memories:
                self.kernel.memory_store.user_memories[user_id] = []
            if user_id in self.kernel.learning_engine.preferences:
                del self.kernel.learning_engine.preferences[user_id]

            # Cancel tasks
            for task in self.proactive_scheduler.get_user_tasks(user_id):
                await self.proactive_scheduler.cancel_task(task.id)

            await query.edit_message_text("✅ All your data has been reset.")

        elif data == "reset_cancel":
            await query.edit_message_text("❌ Reset cancelled.")

        elif data == "schedule_keep":
            await query.edit_message_text("✅ Schedule unchanged.")

    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        print(f"❌ Telegram error: {context.error}")

        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An error occurred. Please try again."
            )

    # ===== HELPER METHODS =====

    def _get_telegram_context(self, update: Update) -> dict:
        """Get Telegram-specific context"""
        user = update.effective_user
        chat = update.effective_chat

        return {
            "user_id": str(user.id),
            "user_name": user.first_name,
            "username": user.username,
            "chat_id": chat.id,
            "chat_type": chat.type,
            "is_private": chat.type == "private",
            "platform": "telegram"
        }

    def _parse_time(self, time_str: str) -> Optional[datetime]:
        """Parse time string to datetime"""
        now = datetime.now()

        # Relative time: 30m, 2h, 1d
        if time_str.endswith('m'):
            minutes = int(time_str[:-1])
            return now + timedelta(minutes=minutes)
        elif time_str.endswith('h'):
            hours = int(time_str[:-1])
            return now + timedelta(hours=hours)
        elif time_str.endswith('d'):
            days = int(time_str[:-1])
            return now + timedelta(days=days)

        # Absolute time: HH:MM
        if ':' in time_str:
            try:
                hour, minute = map(int, time_str.split(':'))
                target = now.replace(hour=hour, minute=minute, second=0)
                if target <= now:
                    target += timedelta(days=1)
                return target
            except:
                pass

        # Keywords
        if time_str == 'tomorrow':
            return now + timedelta(days=1)

        return None

    # ===== LIFECYCLE =====

    async def start(self):
        """Start the Telegram kernel"""
        self.running = True

        # Load previous state
        if self.save_path.exists():
            print("📂 Loading previous Telegram session...")
            await self.kernel.load_from_file(str(self.save_path))

        # Start kernel
        await self.kernel.start()

        # Inject kernel prompt
        self.kernel.inject_kernel_prompt_to_agent()

        # Start auto-save loop
        asyncio.create_task(self._auto_save_loop())

        # Initialize and start Telegram bot
        await self.bot_app.initialize()
        await self.bot_app.start()
        await self.bot_app.updater.start_polling(drop_pending_updates=True)

        print(f"✓ Telegram Kernel started (instance: {self.instance_id})")
        print(f"  Bot: @{(await self.bot_app.bot.get_me()).username}")

    async def stop(self):
        """Stop the Telegram kernel"""
        if not self.running:
            return

        self.running = False
        print("💾 Saving Telegram session...")

        # Save state
        await self.kernel.save_to_file(str(self.save_path))

        # Stop kernel
        await self.kernel.stop()

        # Stop Telegram bot
        await self.bot_app.updater.stop()
        await self.bot_app.stop()
        await self.bot_app.shutdown()

        print("✓ Telegram Kernel stopped")

    async def _auto_save_loop(self):
        """Auto-save kernel state"""
        while self.running:
            await asyncio.sleep(self.auto_save_interval)
            if self.running:
                await self.kernel.save_to_file(str(self.save_path))
                print(f"💾 Auto-saved Telegram kernel at {datetime.now().strftime('%H:%M:%S')}")


# ===== MODULE REGISTRATION =====

Name = "isaa.KernelTelegram"
version = "1.0.0"
app = get_app(Name)
export = app.tb

_kernel_instance: Optional[TelegramKernel] = None


@export(mod_name=Name, version=version, initial=True)
async def init_kernel_telegram(app: App):
    """Initialize the Telegram Kernel module"""
    global _kernel_instance

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not bot_token:
        return {
            "success": False,
            "error": "Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable"
        }

    # Get ISAA and create agent
    isaa = app.get_mod("isaa")
    builder = isaa.get_agent_builder("TelegramKernelAssistant")
    builder.with_system_message(
        "You are a helpful Telegram assistant with proactive capabilities. "
        "You can send reminders, morning briefs, and evening summaries. "
        "Keep responses concise for mobile reading. Use Telegram markdown formatting."
    )

    await isaa.register_agent(builder)
    agent = await isaa.get_agent("TelegramKernelAssistant")

    _kernel_instance = TelegramKernel(agent, app, bot_token=bot_token)
    await _kernel_instance.start()

    return {"success": True, "info": "KernelTelegram initialized"}


@export(mod_name=Name, version=version)
async def stop_kernel_telegram():
    """Stop the Telegram kernel"""
    global _kernel_instance

    if _kernel_instance:
        await _kernel_instance.stop()
        _kernel_instance = None

    return {"success": True, "info": "KernelTelegram stopped"}


async def main():
    """Standalone run"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ Set TELEGRAM_BOT_TOKEN environment variable")
        return

    await init_kernel_telegram(get_app())

    # Keep running
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
