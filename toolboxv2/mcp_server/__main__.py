#!/usr/bin/env python3
"""
ToolBoxV2 MCP Server - CLI Entry Point
======================================
Production-ready CLI with STDIO and HTTP modes
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Optional

# Ensure we can import local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_banner():
    """Print server banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║           ToolBoxV2 MCP Server v3.0.0                        ║
║           Production Ready | Facade & Workers Pattern        ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner, file=sys.stderr)


def print_config_example(api_key: str, host: str = "127.0.0.1", port: int = 8765):
    """Print configuration examples."""
    stdio_config = {
        "mcpServers": {
            "toolboxv2": {
                "command": "tb",
                "args": ["mcp"],
                "env": {"MCP_API_KEY": api_key},
            }
        }
    }

    http_config = {
        "mcpServers": {
            "toolboxv2-http": {
                "url": f"http://{host}:{port}/mcp",
                "transport": "http",
                "headers": {"Authorization": f"Bearer {api_key}"},
            }
        }
    }

    print("\n📋 STDIO Configuration (Claude Desktop, Cursor, etc.):", file=sys.stderr)
    print(json.dumps(stdio_config, indent=2), file=sys.stderr)

    print("\n🌐 HTTP Configuration (Web clients):", file=sys.stderr)
    print(json.dumps(http_config, indent=2), file=sys.stderr)


async def cmd_setup(args):
    """Run setup wizard."""
    from .models import MCPConfig
    from .managers import APIKeyManager

    print_banner()
    print("🧙 Setup Wizard\n", file=sys.stderr)

    config = MCPConfig()
    api_keys = APIKeyManager(config.api_keys_file)

    # Check for existing keys
    await api_keys.load()
    existing_keys = await api_keys.list_keys()

    if not existing_keys:
        print("📝 No API keys found. Generating default admin key...", file=sys.stderr)
        api_key, info = await api_keys.generate_key("default_admin")
        print(f"\n🔑 Your API Key: {api_key}", file=sys.stderr)
        print("⚠️  Save this key securely - it won't be shown again!\n", file=sys.stderr)
        print_config_example(api_key)
    else:
        print(f"✅ Found {len(existing_keys)} existing API key(s)", file=sys.stderr)
        print(
            "\nTo generate a new key, run: tb mcp --generate-key <name>",
            file=sys.stderr,
        )

    print("\n✅ Setup complete!", file=sys.stderr)
    print("   Run 'tb mcp' to start the server.", file=sys.stderr)


async def cmd_generate_key(args):
    """Generate a new API key."""
    from .models import MCPConfig
    from .managers import APIKeyManager

    config = MCPConfig()
    api_keys = APIKeyManager(config.api_keys_file)

    permissions = (
        args.permissions if args.permissions else ["read", "write", "execute", "admin"]
    )

    api_key, info = await api_keys.generate_key(args.generate_key, permissions)

    print(f"\n✓ API Key Generated:", file=sys.stderr)
    print(f"  Name: {info.name}", file=sys.stderr)
    print(f"  Key: {api_key}", file=sys.stderr)
    print(f"  Permissions: {', '.join(info.permissions)}", file=sys.stderr)
    print(f"\n⚠️  Store this key securely - it won't be shown again!", file=sys.stderr)

    print_config_example(api_key, args.host, args.port)

    api_keys.close()


async def cmd_list_keys(args):
    """List all API keys."""
    from .models import MCPConfig
    from .managers import APIKeyManager

    config = MCPConfig()
    api_keys = APIKeyManager(config.api_keys_file)

    keys = await api_keys.list_keys()

    print(f"\n📋 API Keys ({len(keys)}):\n", file=sys.stderr)

    for key_hash, info in keys.items():
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(info["created"]))
        print(f"  • {info['name']}", file=sys.stderr)
        print(f"    Permissions: {', '.join(info['permissions'])}", file=sys.stderr)
        print(f"    Created: {created}", file=sys.stderr)
        print(f"    Usage: {info['usage_count']} calls", file=sys.stderr)
        print("", file=sys.stderr)

    api_keys.close()


async def cmd_revoke_key(args):
    """Revoke an API key."""
    from .models import MCPConfig
    from .managers import APIKeyManager

    config = MCPConfig()
    api_keys = APIKeyManager(config.api_keys_file)

    success = await api_keys.revoke(args.revoke_key)

    if success:
        print(f"✓ Key '{args.revoke_key}' revoked successfully", file=sys.stderr)
    else:
        print(f"✗ Key '{args.revoke_key}' not found", file=sys.stderr)

    api_keys.close()


async def cmd_config(args):
    """Show server configuration."""
    from .models import MCPConfig

    config = MCPConfig()

    print("\n📋 Server Configuration:\n", file=sys.stderr)
    print(json.dumps(config.to_dict(), indent=2), file=sys.stderr)

    print("\n💡 Example MCP client configuration:", file=sys.stderr)
    print_config_example("<YOUR_API_KEY>", config.http_host, config.http_port)


async def cmd_run(args):
    """Run the MCP server."""
    from .models import MCPConfig, ServerMode
    from .server import ToolBoxV2MCPServer

    # Build config from args
    config = MCPConfig(
        server_mode=ServerMode.HTTP if args.mode == "http" else ServerMode.STDIO,
        http_host=args.host,
        http_port=args.port,
        require_auth=not args.no_auth,
        enable_python=not args.disable_python,
        enable_docs=not args.disable_docs,
        enable_flows=not args.disable_flows,
        silent_mode=args.silent,
    )

    # Print banner for HTTP mode (STDIO must be silent)
    if args.mode == "http" and not args.silent:
        print_banner()
        print(
            f"🚀 Starting HTTP server on http://{config.http_host}:{config.http_port}",
            file=sys.stderr,
        )
        print(
            f"   Auth: {'Enabled' if config.require_auth else 'Disabled'}",
            file=sys.stderr,
        )
        print(
            f"   Features: Python={config.enable_python}, Docs={config.enable_docs}, Flows={config.enable_flows}",
            file=sys.stderr,
        )
        print("", file=sys.stderr)

    # Create and run server
    server = ToolBoxV2MCPServer(config)

    try:
        if config.server_mode == ServerMode.HTTP:
            await server.run_http()
        else:
            await server.run_stdio()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user", file=sys.stderr)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ToolBoxV2 MCP Server - Production Ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='tb mcp',
        epilog="""
Examples:
  # Start in STDIO mode (for Claude Desktop, Cursor, etc.)
  tb mcp

  # Start in HTTP mode
  tb mcp --mode http
  tb mcp --mode http --port 8080 --no-auth

  # API Key Management
  tb mcp --setup
  tb mcp --generate-key admin
  tb mcp --generate-key reader --permissions read
  tb mcp --list-keys
  tb mcp --revoke-key admin

  # Show configuration
  tb mcp --config
""",
    )

    # Server mode
    parser.add_argument(
        "--mode",
        choices=["stdio", "http"],
        default="stdio",
        help="Server mode: stdio (default) or http",
    )

    # HTTP options
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="HTTP server host (default: 127.0.0.1 for security)",
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="HTTP server port (default: 8765)"
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable API key authentication (NOT RECOMMENDED)",
    )

    # Feature toggles
    parser.add_argument(
        "--disable-python", action="store_true", help="Disable Python execution tool"
    )
    parser.add_argument(
        "--disable-docs", action="store_true", help="Disable documentation tools"
    )
    parser.add_argument("--disable-flows", action="store_true", help="Disable flow tools")

    # Output control
    parser.add_argument(
        "--silent", action="store_true", help="Silent mode (minimal output)"
    )

    # Management commands
    parser.add_argument("--setup", action="store_true", help="Run setup wizard")
    parser.add_argument(
        "--generate-key",
        type=str,
        metavar="NAME",
        help="Generate new API key with given name",
    )
    parser.add_argument(
        "--permissions",
        nargs="+",
        choices=["read", "write", "execute", "admin"],
        help="Permissions for generated key (default: all)",
    )
    parser.add_argument("--list-keys", action="store_true", help="List all API keys")
    parser.add_argument(
        "--revoke-key", type=str, metavar="NAME", help="Revoke API key by name"
    )
    parser.add_argument("--config", action="store_true", help="Show server configuration")

    args = parser.parse_args()

    # Handle management commands
    if args.setup:
        await cmd_setup(args)
    elif args.generate_key:
        await cmd_generate_key(args)
    elif args.list_keys:
        await cmd_list_keys(args)
    elif args.revoke_key:
        await cmd_revoke_key(args)
    elif args.config:
        await cmd_config(args)
    else:
        # Run server
        try:
            await cmd_run(args)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
