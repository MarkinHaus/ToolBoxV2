"""
Unified Kernel Runner for ProA Kernel
Version: 1.0.0

Orchestrates:
- ProA Kernel (the brain)
- Discord Transport (transport layer)
- Telegram Transport (transport layer)
- Identity Mapping (unified user IDs)
- Multi-Channel Output Router

Usage:
    python unified_runner.py --discord-token XXX --telegram-token YYY --agent-name self
"""

import argparse
import asyncio
import json
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from toolboxv2 import get_app
from toolboxv2.mods.isaa.kernel.instace import Kernel
from toolboxv2.mods.isaa.kernel.types import (
    KernelConfig, KernelState, IOutputRouter
)
from toolboxv2.mods.isaa.kernel.models import MultiChannelRouter

# Import transports
from toolboxv2.mods.isaa.kernel.kernelin.kernelin_discord  import (
    DiscordTransport, DiscordConfig, DiscordOutputRouter,
    create_discord_transport
)
from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import (
    TelegramTransport, TelegramConfig, TelegramOutputRouter,
    create_telegram_transport
)

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent


# =============================================================================
# IDENTITY MAPPING
# =============================================================================

@dataclass
class UserIdentity:
    """Unified user identity across platforms"""
    primary_id: str  # The unified ID used by kernel
    discord_id: Optional[int] = None
    telegram_id: Optional[int] = None
    display_name: str = ""
    preferences: dict = field(default_factory=dict)


class IdentityStore:
    """
    Manages user identity mapping across platforms.

    Example mapping:
    {
        "discord:123456": "user:primary",
        "telegram:789012": "user:primary"
    }

    This allows the same user to interact via Discord and Telegram
    while maintaining a single session/context.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self._identities: dict[str, UserIdentity] = {}
        self._platform_map: dict[str, str] = {}  # platform:id -> primary_id

        if self.storage_path and self.storage_path.exists():
            self._load()

    def register_user(
        self,
        primary_id: str,
        discord_id: Optional[int] = None,
        telegram_id: Optional[int] = None,
        display_name: str = ""
    ) -> UserIdentity:
        """Register a unified user identity"""
        identity = UserIdentity(
            primary_id=primary_id,
            discord_id=discord_id,
            telegram_id=telegram_id,
            display_name=display_name
        )

        self._identities[primary_id] = identity

        if discord_id:
            self._platform_map[f"discord:{discord_id}"] = primary_id
        if telegram_id:
            self._platform_map[f"telegram:{telegram_id}"] = primary_id

        self._save()
        return identity

    def link_discord(self, primary_id: str, discord_id: int):
        """Link Discord account to existing user"""
        if primary_id in self._identities:
            self._identities[primary_id].discord_id = discord_id
            self._platform_map[f"discord:{discord_id}"] = primary_id
            self._save()

    def link_telegram(self, primary_id: str, telegram_id: int):
        """Link Telegram account to existing user"""
        if primary_id in self._identities:
            self._identities[primary_id].telegram_id = telegram_id
            self._platform_map[f"telegram:{telegram_id}"] = primary_id
            self._save()

    def resolve(self, platform_key: str) -> Optional[str]:
        """Resolve platform:id to primary user ID"""
        return self._platform_map.get(platform_key)

    def get_identity(self, primary_id: str) -> Optional[UserIdentity]:
        """Get user identity by primary ID"""
        return self._identities.get(primary_id)

    def get_platform_map(self) -> dict[str, str]:
        """Get the full platform mapping for transport layers"""
        return self._platform_map.copy()

    def _save(self):
        """Save to disk"""
        if not self.storage_path:
            return

        data = {
            "identities": {
                pid: {
                    "primary_id": u.primary_id,
                    "discord_id": u.discord_id,
                    "telegram_id": u.telegram_id,
                    "display_name": u.display_name,
                    "preferences": u.preferences
                }
                for pid, u in self._identities.items()
            },
            "platform_map": self._platform_map
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load from disk"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            for pid, udata in data.get("identities", {}).items():
                self._identities[pid] = UserIdentity(**udata)

            self._platform_map = data.get("platform_map", {})
        except Exception as e:
            print(f"[IdentityStore] Failed to load: {e}")


# =============================================================================
# UNIFIED OUTPUT ROUTER
# =============================================================================

class UnifiedOutputRouter(IOutputRouter):
    """
    Routes outputs to the appropriate platform based on user state.

    Features:
    - Routes to user's current/preferred platform
    - Supports multi-platform delivery (notify on all)
    - Handles platform-specific formatting
    """

    def __init__(
        self,
        discord_router: Optional[DiscordOutputRouter] = None,
        telegram_router: Optional[TelegramOutputRouter] = None,
        identity_store: Optional[IdentityStore] = None
    ):
        self.discord = discord_router
        self.telegram = telegram_router
        self.identity_store = identity_store

        # Track user's last active platform
        self._user_platforms: dict[str, str] = {}  # user_id -> "discord" | "telegram"

    def set_user_platform(self, user_id: str, platform: str):
        """Set user's active platform"""
        self._user_platforms[user_id] = platform

    def get_user_platform(self, user_id: str) -> Optional[str]:
        """Get user's current platform"""
        return self._user_platforms.get(user_id)

    async def send_response(
        self,
        user_id: str,
        content: str,
        role: str = "assistant",
        metadata: dict = None
    ):
        """Send response to user on their active platform"""
        metadata = metadata or {}
        source = metadata.get("source", self._user_platforms.get(user_id))

        if source == "discord" and self.discord:
            await self.discord.send_response(user_id, content, role, metadata)
        elif source == "telegram" and self.telegram:
            await self.telegram.send_response(user_id, content, role, metadata)
        else:
            # Fallback: try both
            if self.discord:
                try:
                    await self.discord.send_response(user_id, content, role, metadata)
                except Exception:
                    pass
            if self.telegram:
                try:
                    await self.telegram.send_response(user_id, content, role, metadata)
                except Exception:
                    pass

    async def send_notification(
        self,
        user_id: str,
        content: str,
        priority: int = 5,
        metadata: dict = None
    ):
        """
        Send notification to user.

        For high priority (>=8), sends to all platforms.
        For normal priority, sends to active platform only.
        """
        metadata = metadata or {}

        if priority >= 8:
            # High priority: notify on all available platforms
            if self.discord:
                try:
                    await self.discord.send_notification(user_id, content, priority, metadata)
                except Exception:
                    pass
            if self.telegram:
                try:
                    await self.telegram.send_notification(user_id, content, priority, metadata)
                except Exception:
                    pass
        else:
            # Normal priority: active platform only
            platform = metadata.get("source", self._user_platforms.get(user_id))

            if platform is None and '_' in user_id:
                platform = user_id.split('_')[0]

            if platform == "discord" and self.discord:
                await self.discord.send_notification(user_id, content, priority, metadata)
            elif platform == "telegram" and self.telegram:
                await self.telegram.send_notification(user_id, content, priority, metadata)
            else:
                await self.telegram.send_notification(user_id, content, priority, metadata) if self.telegram else None
                await self.discord.send_notification(user_id, content, priority, metadata) if self.discord else None


# =============================================================================
# UNIFIED KERNEL RUNNER
# =============================================================================

class UnifiedKernelRunner:
    """
    Orchestrates the Kernel with Discord and Telegram transports.

    Architecture:
    - Single Kernel instance (the brain)
    - Multiple transport layers (Discord, Telegram)
    - Unified output router
    - Shared identity store
    """

    def __init__(
        self,
        agent: 'FlowAgent',
        discord_token: Optional[str] = None,
        telegram_token: Optional[str] = None,
        discord_admin_ids: list[int] = None,
        telegram_admin_ids: list[int] = None,
        identity_store_path: Optional[str] = None,
        kernel_config: Optional[KernelConfig] = None
    ):
        self.agent = agent

        # Identity store
        self.identity_store = IdentityStore(identity_store_path)
        identity_map = self.identity_store.get_platform_map()

        # Transports
        self.discord_transport: Optional[DiscordTransport] = None
        self.telegram_transport: Optional[TelegramTransport] = None

        if discord_token:
            self.discord_transport = create_discord_transport(
                kernel=None,  # Will be set later
                token=discord_token,
                admin_ids=discord_admin_ids or [],
                identity_map=identity_map
            )

        if telegram_token:
            self.telegram_transport = create_telegram_transport(
                kernel=None,  # Will be set later
                token=telegram_token,
                admin_ids=telegram_admin_ids or [],
                identity_map=identity_map
            )

        # Unified output router
        self.output_router = UnifiedOutputRouter(
            discord_router=self.discord_transport.get_router() if self.discord_transport else None,
            telegram_router=self.telegram_transport.get_router() if self.telegram_transport else None,
            identity_store=self.identity_store
        )

        # Kernel
        self.kernel = Kernel(
            agent=agent,
            config=kernel_config or KernelConfig(),
            output_router=self.output_router
        )

        # Inject kernel into transports
        if self.discord_transport:
            self.discord_transport.kernel = self.kernel
        if self.telegram_transport:
            self.telegram_transport.kernel = self.kernel

        # State
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self):
        """Start all components"""
        print("=" * 60)
        print("UNIFIED KERNEL RUNNER")
        print("=" * 60)
        print(f"Agent: {self.agent.amd.name}")
        print(f"Discord: {'Enabled' if self.discord_transport else 'Disabled'}")
        print(f"Telegram: {'Enabled' if self.telegram_transport else 'Disabled'}")
        print("=" * 60)

        self._running = True

        # Start kernel
        print("[Runner] Starting Kernel...")
        await self.kernel.start()

        # Start transports
        if self.discord_transport:
            print("[Runner] Starting Discord transport...")
            self._tasks.append(
                asyncio.create_task(self._run_discord())
            )

        if self.telegram_transport:
            print("[Runner] Starting Telegram transport...")
            self._tasks.append(
                asyncio.create_task(self._run_telegram())
            )

        print("[Runner] All components started!")
        print("=" * 60)

    async def _run_discord(self):
        """Run Discord transport (wrapped for error handling)"""
        try:
            await self.discord_transport.start()
        except Exception as e:
            print(f"[Runner] Discord error: {e}")

    async def _run_telegram(self):
        """Run Telegram transport (wrapped for error handling)"""
        try:
            await self.telegram_transport.start()
        except Exception as e:
            print(f"[Runner] Telegram error: {e}")

    async def stop(self):
        """Stop all components"""
        print("[Runner] Stopping all components...")
        self._running = False

        # Stop transports
        if self.discord_transport:
            try:
                await self.discord_transport.stop()
            except Exception as e:
                print(f"[Runner] Discord stop error: {e}")

        if self.telegram_transport:
            try:
                await self.telegram_transport.stop()
            except Exception as e:
                print(f"[Runner] Telegram stop error: {e}")

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        # Stop kernel
        await self.kernel.stop()

        print("[Runner] All components stopped.")

    async def run_forever(self):
        """Run until interrupted"""
        await self.start()

        try:
            while self._running and self.kernel.state == KernelState.RUNNING:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    def get_status(self) -> dict:
        """Get status of all components"""
        return {
            "kernel": self.kernel.get_status(),
            "discord": "running" if self.discord_transport else "disabled",
            "telegram": "running" if self.telegram_transport else "disabled",
            "identities": len(self.identity_store._identities)
        }


# =============================================================================
# CLI
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="ProA Kernel Unified Runner")

    # Agent
    parser.add_argument("--agent-name", default="self", help="Agent name to load")

    # Discord
    parser.add_argument("--discord-token", help="Discord bot token")
    parser.add_argument("--discord-admins", type=str, help="Discord admin IDs (comma-separated)")

    # Telegram
    parser.add_argument("--telegram-token", help="Telegram bot token")
    parser.add_argument("--telegram-admins", type=str, help="Telegram admin IDs (comma-separated)")

    # Identity
    parser.add_argument("--identity-store", help="Path to identity store JSON")

    args = parser.parse_args()

    # Parse admin IDs
    discord_admin_ids = []
    if args.discord_admins:
        discord_admin_ids = [int(x.strip()) for x in args.discord_admins.split(",")]

    telegram_admin_ids = []
    if args.telegram_admins:
        telegram_admin_ids = [int(x.strip()) for x in args.telegram_admins.split(",")]

    # Get tokens from env if not provided
    discord_token = args.discord_token or os.environ.get("DISCORD_BOT_TOKEN")
    telegram_token = args.telegram_token or os.environ.get("TELEGRAM_BOT_TOKEN")

    if not discord_token and not telegram_token:
        print("Error: At least one of --discord-token or --telegram-token must be provided")
        print("       (or set DISCORD_BOT_TOKEN / TELEGRAM_BOT_TOKEN environment variables)")
        return

    # Load agent
    print(f"[Main] Loading agent '{args.agent_name}'...")
    try:
        app = get_app()
        isaa = app.get_mod("isaa")
        agent = await isaa.get_agent(args.agent_name)

        if not agent:
            print(f"Error: Agent '{args.agent_name}' not found")
            return
    except Exception as e:
        print(f"Error loading agent: {e}")
        return

    # Create runner
    runner = UnifiedKernelRunner(
        agent=agent,
        discord_token=discord_token,
        telegram_token=telegram_token,
        discord_admin_ids=discord_admin_ids,
        telegram_admin_ids=telegram_admin_ids,
        identity_store_path=args.identity_store
    )

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def handle_signal():
        print("\n[Main] Received shutdown signal...")
        asyncio.create_task(runner.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    # Run
    try:
        await runner.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("[Main] Goodbye!")



import asyncio
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Agent Configuration
    "agent_name": "self",  # Your FlowAgent name

    # Discord Configuration
    "discord": {
        "enabled": True,
        "token": os.environ.get("DISCORD_BOT_TOKEN", ""),
        "admin_ids": [
            int(os.environ.get("DISCORD_AID", "0"))
            # Add your Discord user IDs here
            # Example: 123456789012345678
        ],
        "voice_enabled": True,
        "voice_language": "de",  # Whisper transcription language
        "tts_provider": "local",  # "local", "elevenlabs"
    },

    # Telegram Configuration
    "telegram": {
        "enabled": True,
        "token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "admin_ids": [
            int(os.environ.get("TELEGRAM_AID", "0"))
            # Add your Telegram user IDs here
            # Example: 1234567890
        ],
        "voice_language": "de",
    },

    # Identity Mapping
    # Maps platform IDs to unified user IDs
    # This allows same user to use both platforms with shared context
    "identity_map": {
        f"discord:{os.environ.get("DISCORD_AID", "0")}": "user:markin",
        f"telegram:{os.environ.get("TELEGRAM_AID", "0")}": "user:markin",
    },

    # Kernel Configuration
    "kernel": {
        "heartbeat_interval": 1.0,
        "signal_timeout": 0.5,
        "max_signal_queue_size": 100,
    },

    # Storage paths
    "storage": {
        "identity_store": "./data/identity_store.json",
        "kernel_state": "./data/kernel_state/",
    }
}


# =============================================================================
# SETUP HELPERS
# =============================================================================

def validate_config():
    """
    Validate
    configuration
    """
    errors = []

    if CONFIG["discord"]["enabled"]:
        if not CONFIG["discord"]["token"]:
            errors.append("Discord enabled but no token provided")
        if not CONFIG["discord"]["admin_ids"]:
            errors.append("Discord enabled but no admin IDs - anyone can use the bot!")

    if CONFIG["telegram"]["enabled"]:
        if not CONFIG["telegram"]["token"]:
            errors.append("Telegram enabled but no token provided")
        if not CONFIG["telegram"]["admin_ids"]:
            errors.append("Telegram enabled but no admin IDs - anyone can use the bot!")

    if not CONFIG["discord"]["enabled"] and not CONFIG["telegram"]["enabled"]:
        errors.append("At least one transport (Discord or Telegram) must be enabled")

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        print("⚠️  Warning: GROQ_API_KEY not set - voice transcription will not work")

    return errors


def setup_directories():
    """
    Create
    necessary
    directories
    """
    for key, path in CONFIG["storage"].items():
        p = Path(path)
        if p.suffix:  # It's a file
            p.parent.mkdir(parents=True, exist_ok=True)
        else:  # It's a directory
            p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MAIN
# =============================================================================

async def run():
    """
    Run
    the
    unified
    kernel
    """
    print("=" * 60)
    print("ProA Kernel - Transport Configuration")
    print("=" * 60)

    # Validate
    errors = validate_config()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  ❌ {err}")
        return

    # Setup directories
    setup_directories()

    # Import here to avoid import errors if toolboxv2 not installed
    from toolboxv2 import get_app
    from toolboxv2.mods.isaa.kernel.instace import Kernel
    from toolboxv2.mods.isaa.kernel.types import KernelConfig
    from toolboxv2.mods.isaa.kernel.kernelin.kernelin_discord import DiscordConfig
    from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import TelegramConfig

    # Load agent
    print(f"\n[Setup] Loading agent '{CONFIG['agent_name']}'...")
    app = get_app()
    isaa = app.get_mod("isaa")
    agent = await isaa.get_agent(CONFIG["agent_name"])

    if not agent:
        print(f"❌ Agent '{CONFIG['agent_name']}' not found!")
        return

    print(f"✓ Agent loaded: {agent.amd.name}")

    # Setup identity store
    identity_store = IdentityStore(CONFIG["storage"]["identity_store"])

    # Pre-register users from config
    for platform_key, primary_id in CONFIG["identity_map"].items():
        if platform_key.startswith("discord:"):
            discord_id = int(platform_key.split(":")[1])
            identity_store.link_discord(primary_id, discord_id)
        elif platform_key.startswith("telegram:"):
            telegram_id = int(platform_key.split(":")[1])
            identity_store.link_telegram(primary_id, telegram_id)

    # Create runner
    runner = UnifiedKernelRunner(
        agent=agent,
        discord_token=CONFIG["discord"]["token"] if CONFIG["discord"]["enabled"] else None,
        telegram_token=CONFIG["telegram"]["token"] if CONFIG["telegram"]["enabled"] else None,
        discord_admin_ids=CONFIG["discord"]["admin_ids"],
        telegram_admin_ids=CONFIG["telegram"]["admin_ids"],
        identity_store_path=CONFIG["storage"]["identity_store"],
        kernel_config=KernelConfig(**CONFIG["kernel"])
    )

    # Display status
    print("\n[Setup] Configuration:")
    print(f"  Discord: {'✓ Enabled' if CONFIG['discord']['enabled'] else '✗ Disabled'}")
    if CONFIG["discord"]["enabled"]:
        print(f"    - Admins: {CONFIG['discord']['admin_ids']}")
        print(f"    - Voice: {'✓' if CONFIG['discord']['voice_enabled'] else '✗'}")

    print(f"  Telegram: {'✓ Enabled' if CONFIG['telegram']['enabled'] else '✗ Disabled'}")
    if CONFIG["telegram"]["enabled"]:
        print(f"    - Admins: {CONFIG['telegram']['admin_ids']}")

    print(f"  Identity mappings: {len(CONFIG['identity_map'])}")
    print("")

    # Run
    try:
        await runner.run_forever()
    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(run())

