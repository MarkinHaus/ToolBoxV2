#!/usr/bin/env python3
"""
ContainerManager CLI - Command Line Interface

Usage:
    python -m toolboxv2.mods.ContainerManager.cli create <user_id> [type]
    python -m toolboxv2.mods.ContainerManager.cli list [user_id]
    python -m toolboxv2.mods.ContainerManager.cli delete <container_id>
    python -m toolboxv2.mods.ContainerManager.cli logs <container_id>
    python -m toolboxv2.mods.ContainerManager.cli exec <container_id> <command>
    python -m toolboxv2.mods.ContainerManager.cli generate-key
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from toolboxv2 import get_app
from . import (
    create_container,
    list_containers,
    delete_container,
    start_container,
    stop_container,
    restart_container,
    container_logs,
    container_exec,
    generate_admin_key,
    get_container_ssh_info,
    add_ssh_key_to_container,
    list_ssh_containers, start_ui,
)

ADMIN_KEY = os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me")


async def cmd_create(args):
    """Create a new container"""
    if not args.user_id:
        print("❌ Error: user_id is required")
        return 1

    result = await create_container(
        app=get_app(),
        container_type=args.type or "cli_v4",
        user_id=args.user_id,
        container_name=args.name or None,
        admin_key=ADMIN_KEY,
        image=args.image or None,
        command=args.command or None,
        memory_limit=args.memory or None,
        cpu_limit=args.cpu or None,
    )

    if result.is_error():
        print(f"❌ Error: {result.info}")
        return 1

    data = result.get()
    print(f"✅ Container created successfully!")
    print(f"   ID: {data['container_id']}")
    print(f"   Name: {data['container_name']}")
    print(f"   Type: {args.type or 'cli_v4'}")
    print(f"   Port: {data['port']}")
    print(f"   URL: {data['url']}")
    print(f"   Status: {data['status']}")
    return 0


async def cmd_list(args):
    """List containers"""
    result = await list_containers(
        app=get_app(),
        user_id=args.user_id or None,
        admin_key=ADMIN_KEY,
        all=args.all or False
    )

    if result.is_error():
        print(f"❌ Error: {result.info}")
        return 1

    containers = result.get().get("containers", [])

    if not containers:
        print("📭 No containers found")
        return 0

    print(f"\n📋 Found {len(containers)} container(s):\n")

    for c in containers:
        status_icon = "🟢" if c["status"] == "running" else "⚫"
        print(f"{status_icon} {c['container_id']}")
        print(f"   Name: {c['container_name']}")
        print(f"   Type: {c['container_type']}")
        print(f"   User: {c['user_id']}")
        print(f"   Port: {c['port']}")
        print(f"   URL: {c['url']}")
        print(f"   Created: {c.get('created_at', 'N/A')}")
        print()

    return 0


async def cmd_delete(args):
    """Delete a container"""
    if not args.container_id:
        print("❌ Error: container_id is required")
        return 1

    result = await delete_container(
        app=get_app(),
        container_id=args.container_id,
        admin_key=ADMIN_KEY,
        force=args.force or False
    )

    if result.is_error():
        print(f"❌ Error: {result.info}")
        return 1

    print(f"✅ {result.get().get('message')}")
    return 0


async def cmd_start(args):
    """Start a container"""
    if not args.container_id:
        print("❌ Error: container_id is required")
        return 1

    result = await start_container(
        app=get_app(),
        container_id=args.container_id,
        admin_key=ADMIN_KEY
    )

    if result.is_error():
        print(f"❌ Error: {result.info}")
        return 1

    print(f"✅ Container started")
    print(f"   Status: {result.get().get('status')}")
    return 0


async def cmd_stop(args):
    """Stop a container"""
    if not args.container_id:
        print("❌ Error: container_id is required")
        return 1

    result = await stop_container(
        app=get_app(),
        container_id=args.container_id,
        admin_key=ADMIN_KEY
    )

    if result.is_error():
        print(f"❌ Error: {result.info}")
        return 1

    print(f"✅ Container stopped")
    print(f"   Status: {result.get().get('status')}")
    return 0


async def cmd_restart(args):
    """Restart a container"""
    if not args.container_id:
        print("❌ Error: container_id is required")
        return 1

    result = await restart_container(
        app=get_app(),
        container_id=args.container_id,
        admin_key=ADMIN_KEY
    )

    if result.is_error():
        print(f"❌ Error: {result.info}")
        return 1

    print(f"✅ Container restarted")
    print(f"   Status: {result.get().get('status')}")
    return 0


async def cmd_logs(args):
    """Show container logs"""
    if not args.container_id:
        print("❌ Error: container_id is required")
        return 1

    result = await container_logs(
        app=get_app(),
        container_id=args.container_id,
        admin_key=ADMIN_KEY,
        tail=args.tail or 100,
        follow=False  # CLI doesn't support follow yet
    )

    if result.is_error():
        print(f"❌ Error: {result.info}")
        return 1

    logs = result.get().get("logs", "")
    print(f"\n📋 Logs for {args.container_id}:\n")
    print(logs)
    return 0


async def cmd_exec(args):
    """Execute command in container"""
    if not args.container_id or not args.command:
        print("❌ Error: container_id and command are required")
        return 1

    result = await container_exec(
        app=get_app(),
        container_id=args.container_id,
        admin_key=ADMIN_KEY,
        command=args.command,
        timeout=args.timeout or 60
    )

    if result.is_error():
        print(f"❌ Error: {result.info}")
        return 1

    data = result.get()
    print(f"\n🔧 Executed: {args.command}")
    print(f"   Exit Code: {data.get('exit_code')}")
    print(f"\nOutput:\n{data.get('output')}")
    return 0


async def cmd_generate_key(args):
    """Generate a new admin key"""
    result = await generate_admin_key(app=get_app(), name=args.name or "admin")
    if result.is_error():
        print(f"❌ Error: {result.info}")
        return 1
    return 0


async def cmd_ssh(args):
    """SSH into a container"""
    result = await get_container_ssh_info(
        app=get_app(),
        container_id=args.container_id,
        admin_key=ADMIN_KEY
    )
    if result.is_error():
        print(f"❌ Error: {result.info}")
        return 1

    info = result.get()
    print(f"\n{'='*60}")
    print(f"🔑 SSH ACCESS: {info['container_name']}")
    print(f"{'='*60}")
    print(f"Container: {info['container_id']}")
    print(f"User: {info['user_id']}")
    print(f"SSH Port: {info['ssh_port']}")
    print(f"Server: {info['server_ip']}")
    print(f"Username: {info['username']}")
    print(f"\n📋 Connect with:")
    print(f"   ssh -p {info['ssh_port']} cli@{info['server_ip']}")
    print(f"{'='*60}\n")

    # Try to open SSH directly
    import shutil
    if shutil.which("ssh"):
        import subprocess
        try:
            print("🔗 Opening SSH connection... (Exit with 'Ctrl+b, d' or 'exit')")
            ssh_cmd = [
                "ssh",
                "-p", str(info['ssh_port']),
                f"cli@{info['server_ip']}"
            ]
            subprocess.run(ssh_cmd)
        except KeyboardInterrupt:
            print("\n✋ SSH connection closed")
    else:
        print("⚠️  SSH client not found. Please install OpenSSH.")

    return 0


async def cmd_add_ssh_key(args):
    """Add SSH key to a container"""
    if not args.ssh_key:
        print("❌ Error: ssh_key is required")
        return 1

    result = await add_ssh_key_to_container(
        app=get_app(),
        container_id=args.container_id,
        ssh_public_key=args.ssh_key,
        admin_key=ADMIN_KEY
    )
    if result.is_error():
        print(f"❌ Error: {result.info}")
        return 1

    data = result.get()
    print(f"✅ {data['message']}")
    print(f"   Container: {data['container_id']}")
    print(f"   Connect: {data['ssh_connection']}")
    return 0


async def cmd_list_ssh(args):
    """List containers with SSH access"""
    result = await list_ssh_containers(
        app=get_app(),
        user_id=args.user_id or None,
        admin_key=ADMIN_KEY
    )
    if result.is_error():
        print(f"❌ Error: {result.info}")
        return 1

    data = result.get()
    containers = data.get("containers", [])

    if not containers:
        print("📭 No SSH-enabled containers found")
        return 0

    print(f"\n🔑 Found {len(containers)} SSH-enabled container(s):\n")

    for c in containers:
        status_icon = "🟢" if c.get("status") == "running" else "⚫"
        print(f"{status_icon} {c['container_id']}")
        print(f"   Name: {c['container_name']}")
        print(f"   User: {c['user_id']}")
        print(f"   SSH Port: {c['ssh_port']}")
        print(f"   Connect: {c['connection_string']}")
        print()

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="container-manager",
        description="ContainerManager CLI - Manage Docker containers"
    )

    parser.add_argument("--admin-key", default=None, help="Admin key (or set CONTAINER_ADMIN_KEY env var)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # create
    create_parser = subparsers.add_parser("create", help="Create a new container")
    create_parser.add_argument("user_id", help="User ID from CloudM Auth")
    create_parser.add_argument("type", nargs="?", default="cli_v4", help="Container type (cli_v4, project_dev, preview_server, custom)")
    create_parser.add_argument("--name", help="Container name")
    create_parser.add_argument("--image", help="Docker image")
    create_parser.add_argument("--command", help="Container command")
    create_parser.add_argument("--memory", help="Memory limit (e.g., 512m, 1g)")
    create_parser.add_argument("--cpu", help="CPU limit (e.g., 0.5, 1.0)")

    # list
    list_parser = subparsers.add_parser("list", help="List containers")
    list_parser.add_argument("user_id", nargs="?", help="User ID (optional, shows all if admin)")
    list_parser.add_argument("--all", action="store_true", help="List all containers (admin only)")

    # delete
    delete_parser = subparsers.add_parser("delete", help="Delete a container")
    delete_parser.add_argument("container_id", help="Container ID")
    delete_parser.add_argument("--force", action="store_true", help="Force delete running container")

    # start
    start_parser = subparsers.add_parser("start", help="Start a stopped container")
    start_parser.add_argument("container_id", help="Container ID")

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop a running container")
    stop_parser.add_argument("container_id", help="Container ID")

    # restart
    restart_parser = subparsers.add_parser("restart", help="Restart a container")
    restart_parser.add_argument("container_id", help="Container ID")

    # logs
    logs_parser = subparsers.add_parser("logs", help="Show container logs")
    logs_parser.add_argument("container_id", help="Container ID")
    logs_parser.add_argument("--tail", type=int, default=100, help="Number of lines to show")

    # exec
    exec_parser = subparsers.add_parser("exec", help="Execute command in container")
    exec_parser.add_argument("container_id", help="Container ID")
    exec_parser.add_argument("command", help="Command to execute")
    exec_parser.add_argument("--timeout", type=int, default=60, help="Command timeout")

    # generate-key
    key_parser = subparsers.add_parser("generate-key", help="Generate a new admin key")
    key_parser.add_argument("--name", default="admin", help="Key name")

    # ssh
    ssh_parser = subparsers.add_parser("ssh", help="SSH into a container")
    ssh_parser.add_argument("container_id", help="Container ID")

    # add-ssh-key
    add_ssh_parser = subparsers.add_parser("add-ssh-key", help="Add SSH key to container")
    add_ssh_parser.add_argument("container_id", help="Container ID")
    add_ssh_parser.add_argument("ssh_key", help="SSH Public Key (ssh-ed25519 AAAA...)")

    # list-ssh
    list_ssh_parser = subparsers.add_parser("list-ssh", help="List SSH-enabled containers")
    list_ssh_parser.add_argument("user_id", nargs="?", help="Filter by User ID (optional)")
    # list-ssh
    st_ui_parser = subparsers.add_parser("ui", help="start st ui")
    st_ui_parser.add_argument("host", nargs="?", help="host ip default=localhost", default="localhost")
    st_ui_parser.add_argument("port", nargs="?", help="port default=8510", default=8510)

    args = parser.parse_args()

    # Override admin key from args
    global ADMIN_KEY
    if args.admin_key:
        ADMIN_KEY = args.admin_key

    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    commands = {
        "create": cmd_create,
        "list": cmd_list,
        "delete": cmd_delete,
        "start": cmd_start,
        "ui": lambda x:start_ui(host=args.host, port=args.prot),
        "restart": cmd_restart,
        "logs": cmd_logs,
        "exec": cmd_exec,
        "generate-key": cmd_generate_key,
        "ssh": cmd_ssh,
        "add-ssh-key": cmd_add_ssh_key,
        "list-ssh": cmd_list_ssh,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        return asyncio.run(cmd_func(args))

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
