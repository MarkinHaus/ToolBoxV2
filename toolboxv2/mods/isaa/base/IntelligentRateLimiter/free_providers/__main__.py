# toolboxv2/mods/isaa/base/IntelligentRateLimiter/__main__.py
"""
Standalone onboarding entry point.

Usage:
    python -m toolboxv2.mods.isaa.base.IntelligentRateLimiter
    python -m toolboxv2.mods.isaa.base.IntelligentRateLimiter --env /path/to/.env
    python -m toolboxv2.mods.isaa.base.IntelligentRateLimiter --out config.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.onboarding import onboarding_flow
from toolboxv2.mods.isaa.base.IntelligentRateLimiter.free_providers.fetch_models import fetch_available_models
from toolboxv2.mods.isaa.base.IntelligentRateLimiter.build_config import build_config


async def _main(env_path: Path, out_path: Path, rotation_mode: str) -> int:
    env_path = env_path.expanduser().resolve()
    out_path = out_path.expanduser().resolve()
    print(f"[*] env file: {env_path}")
    result = await onboarding_flow(env_path=env_path)
    if not result.active_providers:
        print("No active providers. Nothing to write.", file=sys.stderr)
        return 1

    print("\n[*] Fetching available models...")
    models_raw = await fetch_available_models(
        active_providers=result.active_providers,
        extra_env=result.extra_env,
    )
    for pid, models in models_raw.items():
        print(f"  {pid:<14} {len(models)} models")

    cfg = build_config(result, models_raw, key_rotation_mode=rotation_mode)
    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\n[✓] Config written to {out_path}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="IntelligentRateLimiter")
    ap.add_argument("--env", type=Path, default=Path(".env"))
    ap.add_argument("--out", type=Path, default=Path("rate_limiter_config.json"))
    ap.add_argument("--rotation", choices=("drain", "balance"), default="drain")
    args = ap.parse_args()
    sys.exit(asyncio.run(_main(args.env, args.out, args.rotation)))


if __name__ == "__main__":
    main()
