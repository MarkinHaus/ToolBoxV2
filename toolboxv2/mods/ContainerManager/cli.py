#!/usr/bin/env python3
"""
ContainerManager CLI

Usage:
    # Admin
    python -m toolboxv2.mods.ContainerManager.cli create <user_id> [type]
    python -m toolboxv2.mods.ContainerManager.cli list [user_id] [--all]
    python -m toolboxv2.mods.ContainerManager.cli delete <container_id> [--force]
    python -m toolboxv2.mods.ContainerManager.cli start|stop|restart <container_id>
    python -m toolboxv2.mods.ContainerManager.cli logs <container_id> [--tail N]
    python -m toolboxv2.mods.ContainerManager.cli exec <container_id> <command>
    python -m toolboxv2.mods.ContainerManager.cli add-ssh-key <container_id> <key>
    python -m toolboxv2.mods.ContainerManager.cli list-ssh [user_id]
    python -m toolboxv2.mods.ContainerManager.cli generate-key

    # User
    python -m toolboxv2.mods.ContainerManager.cli setup
    python -m toolboxv2.mods.ContainerManager.cli connect <container_id>
    python -m toolboxv2.mods.ContainerManager.cli register-key <public_key>
    python -m toolboxv2.mods.ContainerManager.cli my-ssh
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

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
    list_ssh_containers,
    register_ssh_key,
    get_my_ssh_info,
    start_ui,
)

ADMIN_KEY = os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me")

# SSH key locations (ContainerManager-native, no DockSH dependency)
SSH_DIR = Path.home() / ".ssh"
CM_KEY_FILE = SSH_DIR / "cm_id_ed25519"
CM_CFG_FILE = Path.home() / ".cm_client.json"


# ============================================================================
# USER COMMANDS — SSH setup and connect live here now
# ============================================================================

async def cmd_setup(args):
    """Generate SSH key for container access."""
    SSH_DIR.mkdir(mode=0o700, exist_ok=True)

    if CM_KEY_FILE.exists():
        print(f"[info] Key already exists: {CM_KEY_FILE}")
    else:
        print("Generating SSH key for ContainerManager access...")
        try:
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-f", str(CM_KEY_FILE), "-N", ""],
                check=True, stdout=subprocess.DEVNULL
            )
            print(f"✓ Key generated: {CM_KEY_FILE}")
        except FileNotFoundError:
            print("Error: 'ssh-keygen' not found. Please install OpenSSH.")
            return 1

    pub_key = CM_KEY_FILE.with_suffix(".pub").read_text().strip()
    print("\n" + "=" * 60)
    print("YOUR PUBLIC KEY — send this to your admin or use register-key:")
    print("=" * 60)
    print(pub_key)
    print("=" * 60)
    print(f"\nPrivate key stored at: {CM_KEY_FILE}")
    print("\nNext steps:")
    print("  1. Send the public key above to your admin, OR")
    print("  2. Run: cm register-key \"<paste key>\"  (if you have an auth token)")
    return 0


async def cmd_register_key(args):
    """Register own SSH public key via CloudM auth — no admin needed."""
    key = args.public_key
    if not key:
        # Try reading from generated key file
        if CM_KEY_FILE.with_suffix(".pub").exists():
            key = CM_KEY_FILE.with_suffix(".pub").read_text().strip()
            print(f"[info] Using key from {CM_KEY_FILE.with_suffix('.pub')}")
        else:
            print("Error: no public key provided and no key file found. Run 'setup' first.")
            return 1

    result = await register_ssh_key(app=get_app(), ssh_public_key=key)
    if result.is_error():
        print(f"Error: {result.info}")
        return 1

    data = result.get()
    print(f"✓ SSH key registered")
    print(f"  Port:    {data['ssh_port']}")
    print(f"  Server:  {data['server_ip']}")
    print(f"  Connect: {data['connection_string']}")

    # Persist connection info
    CM_CFG_FILE.write_text(json.dumps({
        "host": data["server_ip"],
        "port": data["ssh_port"]
    }))
    print(f"\nConnection saved. Run 'cm connect' to connect.")
    return 0


async def cmd_my_ssh(args):
    """Show SSH connection info for own containers."""
    result = await get_my_ssh_info(app=get_app())
    if result.is_error():
        print(f"Error: {result.info}")
        return 1

    data = result.get()
    containers = data.get("containers", [])
    if not containers:
        print("No containers assigned to your account.")
        return 0

    server_ip = data.get("server_ip", "?")
    for c in containers:
        ssh_icon = "🔑" if c.get("ssh_enabled") else "  "
        status_icon = "●" if c.get("status") == "running" else "○"
        print(f"{status_icon} {ssh_icon} {c['container_id']} [{c['container_type']}]")
        print(f"     HTTP: {c['http_url']}")
        if c.get("ssh_enabled"):
            print(f"     SSH:  {c['connection_string']}")
        print()
    return 0


async def cmd_connect(args):
    """SSH into a container using the stored CM key."""
    if not CM_KEY_FILE.exists():
        print("No SSH key found. Run 'setup' first.")
        return 1

    host, port = None, None

    # If container_id given, fetch info from CM
    if hasattr(args, "container_id") and args.container_id:
        result = await get_container_ssh_info(
            app=get_app(),
            container_id=args.container_id,
            admin_key=ADMIN_KEY
        )
        if result.is_error():
            # Maybe it's a user-auth call
            result = await get_my_ssh_info(app=get_app())
            if result.is_error():
                print(f"Error: {result.info}")
                return 1
            containers = result.get().get("containers", [])
            match = next((c for c in containers if c["container_id"].startswith(args.container_id)), None)
            if not match or not match.get("ssh_enabled"):
                print(f"No SSH-enabled container found for '{args.container_id}'")
                return 1
            host = result.get()["server_ip"]
            port = match["ssh_port"]
        else:
            info = result.get()
            host, port = info["server_ip"], info["ssh_port"]
        # Save for next time
        CM_CFG_FILE.write_text(json.dumps({"host": host, "port": port}))

    elif CM_CFG_FILE.exists():
        cfg = json.loads(CM_CFG_FILE.read_text())
        host, port = cfg.get("host"), cfg.get("port")

    if not host or not port:
        print("Usage: cm connect <container_id>")
        print("       or run 'cm register-key' first to save connection info")
        return 1

    print(f"Connecting to {host}:{port} ...")
    ssh_cmd = [
        "ssh",
        "-i", str(CM_KEY_FILE),
        "-p", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=" + str(Path.home() / ".cm_known_hosts"),
        "-o", "LogLevel=ERROR",
        f"cli@{host}"
    ]
    try:
        if sys.platform == "win32":
            subprocess.run(ssh_cmd)
        else:
            os.execvp("ssh", ssh_cmd)
    except KeyboardInterrupt:
        pass
    return 0


# ============================================================================
# ADMIN COMMANDS
# ============================================================================

async def cmd_create(args):
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
        ssh_public_key=args.ssh_key or None,
    )
    if result.is_error():
        print(f"Error: {result.info}")
        return 1

    data = result.get()
    print(f"✓ Container created")
    print(f"  ID:     {data['container_id']}")
    print(f"  Name:   {data['container_name']}")
    print(f"  Port:   {data['port']}")
    print(f"  URL:    {data['url']}")
    print(f"  Status: {data['status']}")
    if data.get("ssh_port"):
        print(f"  SSH:    {data['ssh_connection']}")
    return 0


async def cmd_list(args):
    result = await list_containers(
        app=get_app(),
        user_id=args.user_id or None,
        admin_key=ADMIN_KEY,
        all=args.all or False
    )
    if result.is_error():
        print(f"Error: {result.info}")
        return 1

    containers = result.get().get("containers", [])
    if not containers:
        print("No containers found")
        return 0

    print(f"\n{len(containers)} container(s):\n")
    for c in containers:
        dot = "●" if c["status"] == "running" else "○"
        ssh = f" [SSH:{c['port']}]" if c.get("ssh_port") else ""
        print(f"{dot} {c['container_id']}  {c['container_name']}  ({c['container_type']}){ssh}")
        print(f"    user={c['user_id']}  port={c['port']}  {c['url']}")
    return 0


async def cmd_delete(args):
    result = await delete_container(
        app=get_app(), container_id=args.container_id,
        admin_key=ADMIN_KEY, force=args.force or False
    )
    if result.is_error():
        print(f"Error: {result.info}")
        return 1
    print(f"✓ {result.get().get('message')}")
    return 0


async def cmd_start(args):
    result = await start_container(app=get_app(), container_id=args.container_id, admin_key=ADMIN_KEY)
    if result.is_error():
        print(f"Error: {result.info}")
        return 1
    print(f"✓ Started  status={result.get().get('status')}")
    return 0


async def cmd_stop(args):
    result = await stop_container(app=get_app(), container_id=args.container_id, admin_key=ADMIN_KEY)
    if result.is_error():
        print(f"Error: {result.info}")
        return 1
    print(f"✓ Stopped  status={result.get().get('status')}")
    return 0


async def cmd_restart(args):
    result = await restart_container(app=get_app(), container_id=args.container_id, admin_key=ADMIN_KEY)
    if result.is_error():
        print(f"Error: {result.info}")
        return 1
    print(f"✓ Restarted  status={result.get().get('status')}")
    return 0


async def cmd_logs(args):
    result = await container_logs(
        app=get_app(), container_id=args.container_id,
        admin_key=ADMIN_KEY, tail=args.tail or 100
    )
    if result.is_error():
        print(f"Error: {result.info}")
        return 1
    print(result.get().get("logs", ""))
    return 0


async def cmd_exec(args):
    result = await container_exec(
        app=get_app(), container_id=args.container_id,
        admin_key=ADMIN_KEY, command=args.command,
        timeout=args.timeout or 60
    )
    if result.is_error():
        print(f"Error: {result.info}")
        return 1
    data = result.get()
    print(f"exit={data.get('exit_code')}\n{data.get('output')}")
    return 0


async def cmd_generate_key(args):
    result = await generate_admin_key(app=get_app(), name=args.name or "admin")
    if result.is_error():
        print(f"Error: {result.info}")
        return 1
    return 0


async def cmd_ssh_info(args):
    """Show SSH info for a container (admin view)."""
    result = await get_container_ssh_info(
        app=get_app(), container_id=args.container_id, admin_key=ADMIN_KEY
    )
    if result.is_error():
        print(f"Error: {result.info}")
        return 1
    info = result.get()
    print(f"\n{'='*55}")
    print(f"SSH — {info['container_name']}")
    print(f"{'='*55}")
    print(f"Container:  {info['container_id']}")
    print(f"User:       {info['user_id']}")
    print(f"SSH port:   {info['ssh_port']}")
    print(f"Server:     {info['server_ip']}")
    print(f"\nConnect:    {info['connection_string']}")
    print(f"{'='*55}\n")

    import shutil
    if shutil.which("ssh"):
        try:
            print("Opening SSH session... (exit with 'exit' or Ctrl+D)")
            subprocess.run(["ssh", "-p", str(info["ssh_port"]), f"cli@{info['server_ip']}"])
        except KeyboardInterrupt:
            pass
    return 0


async def cmd_add_ssh_key(args):
    result = await add_ssh_key_to_container(
        app=get_app(), container_id=args.container_id,
        ssh_public_key=args.ssh_key, admin_key=ADMIN_KEY
    )
    if result.is_error():
        print(f"Error: {result.info}")
        return 1
    data = result.get()
    print(f"✓ {data['message']}")
    print(f"  Connect: {data['ssh_connection']}")
    return 0


async def cmd_list_ssh(args):
    result = await list_ssh_containers(
        app=get_app(), user_id=args.user_id or None, admin_key=ADMIN_KEY
    )
    if result.is_error():
        print(f"Error: {result.info}")
        return 1
    containers = result.get().get("containers", [])
    if not containers:
        print("No SSH-enabled containers found")
        return 0
    print(f"\n{len(containers)} SSH-enabled container(s):\n")
    for c in containers:
        dot = "●" if c.get("status") == "running" else "○"
        print(f"{dot} {c['container_id']}  {c['container_name']}  (user={c['user_id']})")
        print(f"    {c['connection_string']}")
    return 0


# ============================================================================
# MAIN
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(prog="cm", description="ContainerManager CLI")
    parser.add_argument("--admin-key", default=None)
    sub = parser.add_subparsers(dest="command")

    # --- USER COMMANDS ---
    sub.add_parser("setup", help="Generate SSH key pair for container access")

    rk = sub.add_parser("register-key", help="Register your public SSH key via auth token")
    rk.add_argument("public_key", nargs="?", default=None, help="SSH public key string (optional, reads from file if omitted)")

    sub.add_parser("my-ssh", help="Show SSH info for your containers")

    conn = sub.add_parser("connect", help="SSH into your container")
    conn.add_argument("container_id", nargs="?", default=None, help="Container ID (uses saved config if omitted)")

    # --- ADMIN COMMANDS ---
    cr = sub.add_parser("create", help="Create container for a user")
    cr.add_argument("user_id")
    cr.add_argument("type", nargs="?", default="cli_v4")
    cr.add_argument("--name")
    cr.add_argument("--image")
    cr.add_argument("--command")
    cr.add_argument("--memory")
    cr.add_argument("--cpu")
    cr.add_argument("--ssh-key", dest="ssh_key", help="Optional initial SSH public key for the user")

    ls = sub.add_parser("list", help="List containers")
    ls.add_argument("user_id", nargs="?")
    ls.add_argument("--all", action="store_true")

    dl = sub.add_parser("delete")
    dl.add_argument("container_id")
    dl.add_argument("--force", action="store_true")

    for cmd in ("start", "stop", "restart"):
        p = sub.add_parser(cmd)
        p.add_argument("container_id")

    lg = sub.add_parser("logs")
    lg.add_argument("container_id")
    lg.add_argument("--tail", type=int, default=100)

    ex = sub.add_parser("exec")
    ex.add_argument("container_id")
    ex.add_argument("command")
    ex.add_argument("--timeout", type=int, default=60)

    gk = sub.add_parser("generate-key")
    gk.add_argument("--name", default="admin")

    si = sub.add_parser("ssh", help="Show SSH info for a container (admin)")
    si.add_argument("container_id")

    ak = sub.add_parser("add-ssh-key", help="Inject SSH public key into a container (admin)")
    ak.add_argument("container_id")
    ak.add_argument("ssh_key")

    lss = sub.add_parser("list-ssh")
    lss.add_argument("user_id", nargs="?")

    ui_p = sub.add_parser("ui")
    ui_p.add_argument("host", nargs="?", default="localhost")
    ui_p.add_argument("port", nargs="?", default=8510, type=int)

    args = parser.parse_args()

    global ADMIN_KEY
    if args.admin_key:
        ADMIN_KEY = args.admin_key

    if not args.command:
        parser.print_help()
        return 0

    handlers = {
        "setup":        cmd_setup,
        "register-key": cmd_register_key,
        "my-ssh":       cmd_my_ssh,
        "connect":      cmd_connect,
        "create":       cmd_create,
        "list":         cmd_list,
        "delete":       cmd_delete,
        "start":        cmd_start,
        "stop":         cmd_stop,
        "restart":      cmd_restart,
        "logs":         cmd_logs,
        "exec":         cmd_exec,
        "generate-key": cmd_generate_key,
        "ssh":          cmd_ssh_info,
        "add-ssh-key":  cmd_add_ssh_key,
        "list-ssh":     cmd_list_ssh,
        "ui":           lambda a: start_ui(host=a.host, port=a.port),
    }

    fn = handlers.get(args.command)
    if fn:
        return await fn(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
