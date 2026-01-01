#!/usr/bin/env python3
"""
Unified Kernel Runner
=====================

Starts both Discord and Telegram kernels with shared agent pool.
Each user gets their own agent instance (self-{username}).

Usage:
------
    python run_unified_kernels.py

    # Or with environment variables:
    DISCORD_BOT_TOKEN=xxx TELEGRAM_BOT_TOKEN=yyy python run_unified_kernels.py

Environment Variables:
----------------------
    DISCORD_BOT_TOKEN     - Discord bot token
    TELEGRAM_BOT_TOKEN    - Telegram bot token from @BotFather
    GROQ_API_KEY          - Groq API key for voice transcription
    ELEVENLABS_API_KEY    - ElevenLabs API key for TTS (optional)

Architecture:
-------------
    ┌─────────────────────────────────────────────────────────────┐
    │                    Shared Components                        │
    │         (Memory Store, Learning Engine, Scheduler)          │
    ├─────────────────────────────────────────────────────────────┤
    │                    Agent Pool                               │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
    │  │ self-markin  │  │ self-samuel  │  │ self-daniil  │       │
    │  └──────────────┘  └──────────────┘  └──────────────┘       │
    ├──────────────┬──────────────────────────────────────────────┤
    │   Discord    │              Telegram                        │
    │   Kernel     │              Kernel                          │
    └──────────────┴──────────────────────────────────────────────┘
"""

import asyncio
import os
import sys
import signal
from pathlib import Path
from datetime import datetime

# Add toolboxv2 to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from toolboxv2 import App, get_app


class UnifiedKernelRunner:
    """Manages both Discord and Telegram kernels"""

    def __init__(self):
        self.app = get_app("UnifiedKernels")
        self.discord_kernel = None
        self.telegram_kernel = None
        self.running = False

        # Check tokens
        self.discord_token = os.getenv("DISCORD_BOT_TOKEN")
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")

        if not self.discord_token and not self.telegram_token:
            print("❌ No bot tokens configured!")
            print("   Set at least one of:")
            print("   - DISCORD_BOT_TOKEN")
            print("   - TELEGRAM_BOT_TOKEN")
            sys.exit(1)

    async def start(self):
        """Start all configured kernels"""
        self.running = True

        print("\n" + "="*60)
        print("🚀 UNIFIED KERNEL RUNNER")
        print("="*60)
        print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Discord: {'✅ Configured' if self.discord_token else '❌ Not configured'}")
        print(f"   Telegram: {'✅ Configured' if self.telegram_token else '❌ Not configured'}")
        print("="*60 + "\n")

        tasks = []

        # Start Discord Kernel
        if self.discord_token:
            try:
                from toolboxv2.mods.isaa.kernel.kernelin.kernelin_discord import (
                    init_kernel_discord, DiscordKernel
                )

                print("🎮 Starting Discord Kernel...")
                result = await init_kernel_discord(self.app)

                if result.get("success"):
                    print("✅ Discord Kernel started!")
                else:
                    print(f"❌ Discord Kernel failed: {result.get('error')}")

            except ImportError as e:
                print(f"⚠️ Discord Kernel not available: {e}")
            except Exception as e:
                print(f"❌ Discord Kernel error: {e}")

        # Start Telegram Kernel
        if self.telegram_token:
            try:
                from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import (
                    init_kernel_telegram, TelegramKernel
                )

                print("📱 Starting Telegram Kernel...")
                result = await init_kernel_telegram(self.app)

                if result.get("success"):
                    print("✅ Telegram Kernel started!")
                else:
                    print(f"❌ Telegram Kernel failed: {result.get('error')}")

            except ImportError as e:
                print(f"⚠️ Telegram Kernel not available: {e}")
            except Exception as e:
                print(f"❌ Telegram Kernel error: {e}")

        print("\n" + "="*60)
        print("🟢 All kernels running! Press Ctrl+C to stop.")
        print("="*60 + "\n")

        # Keep running
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def stop(self):
        """Stop all kernels"""
        self.running = False

        print("\n⏹️ Stopping kernels...")

        # Stop Discord
        if self.discord_token:
            try:
                from toolboxv2.mods.isaa.kernel.kernelin.kernelin_discord import stop_kernel_discord
                await stop_kernel_discord()
                print("✅ Discord Kernel stopped")
            except Exception as e:
                print(f"⚠️ Error stopping Discord: {e}")

        # Stop Telegram
        if self.telegram_token:
            try:
                from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import stop_kernel_telegram
                await stop_kernel_telegram()
                print("✅ Telegram Kernel stopped")
            except Exception as e:
                print(f"⚠️ Error stopping Telegram: {e}")

        print("\n👋 Goodbye!")


async def main():
    """Main entry point"""
    runner = UnifiedKernelRunner()

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        asyncio.create_task(runner.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        await runner.start()
    except KeyboardInterrupt:
        await runner.stop()


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        sys.exit(1)

    # Run
    asyncio.run(main())
