#!/usr/bin/env python3
"""
livesync_setup_share — set up a cross-machine VFS share on THIS server node.

Generates a share (id + AES-256 key + token), prints the token to distribute,
and runs the LiveSync server in the foreground so the share stays live.

Place at: toolboxv2/mods/CloudM/LiveSync/setup_share.py

Requires MinIO env on this node:
    MINIO_ENDPOINT, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD, [LIVESYNC_BUCKET]

Run on the node that hosts the shared folder:
    python -m toolboxv2.mods.CloudM.LiveSync.setup_share /srv/global \
        --ws-host sync.example.com --port 8765

Then on each client node:
    /vfs share connect /global /local/path --token <TOKEN>
"""
import argparse
import asyncio
import socket
import uuid

from .config import load_env_config
from .crypto import generate_encryption_key
from . import create_share_token
from .server import SyncServer


def main():
    ap = argparse.ArgumentParser(
        description="Set up a LiveSync VFS share on this server node.")
    ap.add_argument("folder", help="local folder on this server to share")
    ap.add_argument("--ws-host", default=socket.gethostname(),
                    help="host/IP clients use to reach this server (baked into the token)")
    ap.add_argument("--bind", default="0.0.0.0", help="bind address for the WS server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--share-id", default=None, help="reuse a fixed share id (optional)")
    args = ap.parse_args()

    env = load_env_config()
    share_id = args.share_id or uuid.uuid4().hex[:8]
    enc_key = generate_encryption_key()
    ws_endpoint = f"ws://{args.ws_host}:{args.port}"
    token = create_share_token(
        share_id=share_id, encryption_key=enc_key,
        minio_endpoint=env["endpoint"], ws_endpoint=ws_endpoint,
        bucket=env.get("bucket", "livesync"),
    )

    bar = "=" * 66
    print(bar)
    print(f"  VFS share ready — folder: {args.folder}")
    print(f"  share_id : {share_id}")
    print(f"  ws       : {ws_endpoint}")
    print(f"  minio    : {env['endpoint']}  bucket={env.get('bucket', 'livesync')}")
    print(bar)
    print("  TOKEN (distribute to nodes):")
    print(f"  {token}")
    print(bar)
    print("  On each node:")
    print(f"    /vfs share connect /global {args.folder} --token <TOKEN>")
    print("  Server runs below — Ctrl-C to stop.")
    print(bar)

    asyncio.run(SyncServer(args.folder, share_id, env).start(args.bind, args.port))


if __name__ == "__main__":
    main()
