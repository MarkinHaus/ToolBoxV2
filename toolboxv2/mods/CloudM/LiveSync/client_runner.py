"""Standalone subprocess entry for one SyncClient (e2e harness)."""
import argparse
import asyncio
import logging

from .config import ShareToken
from .client import SyncClient


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--token", required=True,
                    help="signed share token (v4) issued by the server node")
    ap.add_argument("--debounce", type=float, default=0.5)
    a = ap.parse_args()

    # Everything except the vault path comes from the token, and the raw token
    # goes back into the config: the server verifies it on AUTH.
    cfg = ShareToken.decode(a.token).to_sync_config(a.vault, raw_token=a.token)
    cfg.debounce_seconds = a.debounce
    await SyncClient(cfg).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s CLIENT %(levelname)s %(message)s")
    asyncio.run(main())
