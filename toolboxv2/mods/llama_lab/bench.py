# file: toolboxv2/mods/llama_lab/bench.py
"""Optional pre-benchmark with llama-bench (Point 3).

Runs the two sweeps from the tuning guide against a real GGUF and returns the
flags that peaked: threads (token generation) and ubatch-size (prefill). The
result feeds straight into the model's models.ini section.
"""

import json
import subprocess
from pathlib import Path

from .hw import HwInfo


def _bench(bin_dir: Path, args) -> list:
    exe = bin_dir / ("llama-bench.exe" if (bin_dir / "llama-bench.exe").exists() else "llama-bench")
    cmd = [str(exe), *args, "-o", "json"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"llama-bench produced no JSON:\n{out.stderr[-800:]}")


def _peak(rows, want: str):
    """Return (param_value, t/s) for the fastest row of test type `want` (pp|tg)."""
    best = None
    for r in rows:
        # newer llama-bench: n_prompt>0 == prefill(pp), n_gen>0 == tokengen(tg)
        is_pp = r.get("n_prompt", 0) and not r.get("n_gen", 0)
        is_tg = r.get("n_gen", 0) and not r.get("n_prompt", 0)
        if (want == "pp") != bool(is_pp) and (want == "tg") != bool(is_tg):
            continue
        ts = r.get("avg_ts") or r.get("t/s") or 0
        key = (r.get("n_threads"), r.get("n_ubatch"))
        if best is None or ts > best[2]:
            best = (key, r, ts)
    return best


def threads_sweep(bin_dir: Path, gguf: Path, hw: HwInfo, ngl: int = 99) -> dict:
    cores = hw.cpu_physical
    candidates = sorted({max(1, cores - 2), max(1, cores - 1), cores})
    rows = _bench(bin_dir, ["-m", str(gguf), "-ngl", str(ngl),
                            "-t", ",".join(map(str, candidates)), "-p", "512", "-n", "128"])
    best = _peak(rows, "tg")
    threads = best[0][0] if best else cores - 1
    return {"threads": threads, "tg_ts": round(best[2], 1) if best else 0.0,
            "tested": candidates}


def ubatch_sweep(bin_dir: Path, gguf: Path, threads: int, ngl: int = 99) -> dict:
    sizes = [256, 512, 1024, 2048]
    rows = _bench(bin_dir, ["-m", str(gguf), "-ngl", str(ngl), "-t", str(threads),
                            "-ub", ",".join(map(str, sizes)), "-p", "4096", "-n", "0"])
    best = _peak(rows, "pp")
    ub = best[0][1] if best else 512
    return {"ubatch": ub, "pp_ts": round(best[2], 1) if best else 0.0, "tested": sizes}


def autotune(bin_dir: Path, gguf: Path, hw: HwInfo, ngl: int = 99) -> dict:
    """Full sweep -> recommended {threads, ubatch} for this model+hardware."""
    t = threads_sweep(bin_dir, gguf, hw, ngl)
    u = ubatch_sweep(bin_dir, gguf, t["threads"], ngl)
    return {"threads": t["threads"], "ubatch": u["ubatch"],
            "tg_ts": t["tg_ts"], "pp_ts": u["pp_ts"]}
