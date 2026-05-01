# toolboxv2/mods/isaa/base/IntelligentRateLimiter/__main__.py
import argparse
import asyncio
import json
import sys
from pathlib import Path

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.onboarding import (
    onboarding_flow, build_config, cost_table, format_cost_table,
)
from toolboxv2.mods.isaa.base.IntelligentRateLimiter.free_providers.fetch_models import fetch_available_models


async def _main(env_path, out_path, rotation_mode, include_paid,
                cost_max, cost_step, cache_hit):
    env_path = env_path.expanduser().resolve()
    out_path = out_path.expanduser().resolve()
    print(f"[*] env: {env_path}")

    result = await onboarding_flow(env_path=env_path, include_paid=include_paid)
    if not result.active_env_names:
        print("No active providers.", file=sys.stderr)
        return 1

    # fetch_available_models expects {pid: [values]} — it only uses one value per provider
    # for the API call. We pass env var NAMES resolved to VALUES for the fetch only.
    import os
    fetch_input = {
        pid: [os.environ[n] for n in names if os.environ.get(n)]
        for pid, names in result.active_env_names.items()
    }
    fetch_input = {k: v for k, v in fetch_input.items() if v}

    print("\n[*] Fetching models...")
    models_raw = await fetch_available_models(fetch_input, extra_env=result.extra_env)
    for pid, m in models_raw.items():
        print(f"  {pid:<18} {len(m)} models")

    cfg = build_config(result, models_raw, key_rotation_mode=rotation_mode)
    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\n[✓] Config → {out_path}")

    from toolboxv2.mods.isaa.base.IntelligentRateLimiter.onboarding import _render_final_stats
    _render_final_stats(result, models_raw)

    table = cost_table(cfg, max_tokens=cost_max, step=cost_step, cache_hit_fraction=cache_hit)
    if table:
        print("\n[*] PAYG cost table:")
        print(format_cost_table(table, step=cost_step))
    return 0


def main():
    ap = argparse.ArgumentParser(prog="IntelligentRateLimiter")
    ap.add_argument("--env", type=Path, default=Path(".env"))
    ap.add_argument("--out", type=Path, default=Path("rate_limiter_config.json"))
    ap.add_argument("--rotation", choices=("drain", "balance"), default="balance")
    ap.add_argument("--include-paid", choices=("none", "coding", "payg", "all"), default="none")
    ap.add_argument("--cost-max", type=int, default=5_000_000)
    ap.add_argument("--cost-step", type=int, default=500_000)
    ap.add_argument("--cache-hit", type=float, default=0.0)
    args = ap.parse_args()
    sys.exit(asyncio.run(_main(
        args.env, args.out, args.rotation, args.include_paid,
        args.cost_max, args.cost_step, args.cache_hit,
    )))


if __name__ == "__main__":
    main()
