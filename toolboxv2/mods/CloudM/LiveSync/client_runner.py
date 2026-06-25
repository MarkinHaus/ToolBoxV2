"""Standalone subprocess entry for one SyncClient (e2e harness)."""
import argparse, asyncio, logging
from .config import SyncConfig
from .client import SyncClient

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--share-id", required=True)
    ap.add_argument("--ws", required=True)
    ap.add_argument("--minio", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--bucket", default="tb-shared")
    ap.add_argument("--debounce", type=float, default=0.5)
    a = ap.parse_args()
    cfg = SyncConfig(
        share_id=a.share_id, vault_path=a.vault,
        minio_endpoint=a.minio, ws_endpoint=a.ws,
        encryption_key=a.key, bucket=a.bucket,
        prefix=a.share_id, debounce_seconds=a.debounce,
    )
    await SyncClient(cfg).run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s CLIENT %(levelname)s %(message)s")
    asyncio.run(main())
