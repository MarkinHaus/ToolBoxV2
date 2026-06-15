# file: toolboxv2/mods/llama_lab/hub.py
"""HF GGUF browser + downloader + installed-model manager (Point 2).

Part A (browse): search HF for GGUF repos, filterable/searchable, each card
  shows (1) model facts and (2) a run estimate for THIS hardware.
Part B (manage): list installed GGUFs from the llama.cpp cache, remove them.

Uses huggingface_hub when present; degrades to the public HF HTTP API.
The run estimate follows the offload hierarchy: weights that fit in VRAM run
fully on GPU; otherwise layers spill to RAM and become RAM-bandwidth bound.
"""

import json
import os
import platform
import re
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .hw import HwInfo

HF_API = "https://huggingface.co/api/models"


def cache_dir() -> Path:
    """llama.cpp's -hf download cache (also where we install GGUFs)."""
    env = os.environ.get("LLAMA_CACHE")
    if env:
        return Path(env)
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA", str(Path.home()))
        return Path(base) / "llama.cpp"
    return Path.home() / ".cache" / "llama.cpp"


# ---------------------------------------------------------------- browse ----

@dataclass
class RepoCard:
    repo_id: str
    downloads: int
    likes: int
    tags: list
    gguf_files: list   # [(filename, size_bytes)]
    has_mmproj: bool

    @property
    def modality(self) -> str:
        t = " ".join(self.tags).lower() + " " + self.repo_id.lower()
        if self.has_mmproj and ("audio" in t or "omni" in t or "ultravox" in t):
            return "omni"
        if self.has_mmproj or any(k in t for k in ("vl", "vision", "llava", "image")):
            return "vision"
        if any(k in t for k in ("embed", "bge", "gte", "e5")):
            return "embed"
        return "text"


def _http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "toolboxv2-llama_lab"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def _tree_sizes(repo_id: str) -> dict:
    """Real per-file byte sizes via the tree endpoint.

    model_info/siblings report 0 (or the tiny LFS pointer size) for GGUFs, so
    the actual size must be read from `lfs.size`. Pure urllib -> cross-platform,
    no huggingface_hub needed.
    """
    out = {}
    try:
        for e in _http_json(f"{HF_API}/{repo_id}/tree/main?recursive=1&expand=1"):
            if e.get("type") == "file":
                out[e["path"]] = (e.get("lfs") or {}).get("size") or e.get("size") or 0
    except Exception:
        pass
    return out


def search(query: str, limit: int = 20, sort: str = "downloads"):
    """Return repo ids matching the query, GGUF-only, most-downloaded first."""
    try:
        from huggingface_hub import HfApi
        models = HfApi().list_models(search=query or None, filter="gguf",
                                     sort=sort, direction=-1, limit=limit)
        return [m.id for m in models]
    except Exception:
        q = urllib.parse.urlencode({"search": query, "filter": "gguf",
                                    "sort": sort, "direction": -1, "limit": limit})
        try:
            return [m["id"] for m in _http_json(f"{HF_API}?{q}")]
        except Exception:
            return []


def card(repo_id: str) -> RepoCard:
    """Fetch a single model card with GGUF file list + real sizes."""
    sizes = _tree_sizes(repo_id)               # authoritative byte sizes
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, files_metadata=True)
        sib = info.siblings or []
        ggufs = [(s.rfilename, sizes.get(s.rfilename, s.size or 0)) for s in sib
                 if s.rfilename.lower().endswith(".gguf") and "mmproj" not in s.rfilename.lower()]
        has_mm = any("mmproj" in s.rfilename.lower() for s in sib)
        return RepoCard(repo_id, info.downloads or 0, info.likes or 0,
                        list(info.tags or []), ggufs, has_mm)
    except Exception:
        info = _http_json(f"{HF_API}/{repo_id}")
        sib = info.get("siblings", [])
        ggufs = [(s["rfilename"], sizes.get(s["rfilename"], s.get("size", 0))) for s in sib
                 if s["rfilename"].lower().endswith(".gguf")
                 and "mmproj" not in s["rfilename"].lower()]
        has_mm = any("mmproj" in s["rfilename"].lower() for s in sib)
        return RepoCard(repo_id, info.get("downloads", 0), info.get("likes", 0),
                        info.get("tags", []), ggufs, has_mm)


# ------------------------------------------------------------- estimate ----

def _params_from_name(name: str) -> float:
    """Best-effort active/total param count in billions from the file name."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", name)
    return float(m.group(1)) if m else 0.0


def estimate(file_size_bytes: int, filename: str, hw: HwInfo, ctx: int = 8192) -> dict:
    """Qualitative fit estimate for this hardware (no fabricated tok/s)."""
    weights = file_size_bytes / 1e9
    overhead = 1.3                                   # runtime + compute buffers
    kv = round(ctx / 1024 * 0.10, 2)                 # coarse KV-cache @ fp16
    need = weights + overhead + kv
    vram = hw.vram_gb
    is_moe = bool(re.search(r"(moe|a\d+b|reap)", filename.lower()))

    if vram <= 0:                                    # CPU-only host
        tier = "🟡 CPU only" if need <= hw.ram_gb else "🔴 too large for RAM"
        bound = "CPU/RAM bandwidth"
    elif need <= vram:
        tier, bound = "🟢 full GPU", "compute"
    elif weights <= vram:
        tier, bound = "🟡 GPU + small KV spill", "VRAM (reduce ctx / quant KV)"
    elif need <= vram + hw.ram_gb:
        tier, bound = "🟠 partial offload", "RAM/PCIe bandwidth"
    else:
        tier, bound = "🔴 won't fit", "out of memory"

    note = ""
    if is_moe and tier.startswith(("🟠", "🟡")):
        note = " — MoE: only active experts compute, far faster than dense at this size"
    return {"need_gb": round(need, 1), "kv_gb": kv, "tier": tier,
            "bound": bound, "note": note}


# -------------------------------------------------------------- manage ----

def download(repo_id: str, filename: str, with_mmproj: bool = True, spinner=None) -> Path:
    """Download one GGUF (+ mmproj sibling for vision/omni) into the cache."""
    cd = cache_dir()
    cd.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        if spinner:
            spinner.message = f"download {repo_id}/{filename}"
        out = hf_hub_download(repo_id, filename, local_dir=cd)
        if with_mmproj:
            for f in list_repo_files(repo_id):
                if "mmproj" in f.lower() and f.lower().endswith(".gguf"):
                    hf_hub_download(repo_id, f, local_dir=cd)
        return Path(out)
    except ImportError:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{urllib.parse.quote(filename)}"
        out = cd / Path(filename).name
        req = urllib.request.Request(url, headers={"User-Agent": "toolboxv2-llama_lab"})
        with urllib.request.urlopen(req, timeout=300) as r, open(out, "wb") as fh:
            shutil.copyfileobj(r, fh)
        return out


def installed():
    """List installed GGUFs (excluding mmproj) as [(path, size_bytes)]."""
    cd = cache_dir()
    if not cd.exists():
        return []
    out = []
    for p in cd.rglob("*.gguf"):
        if "mmproj" in p.name.lower():
            continue
        try:
            out.append((p, p.stat().st_size))
        except OSError:
            pass
    return sorted(out, key=lambda x: x[1], reverse=True)


def remove(path: Path) -> bool:
    try:
        Path(path).unlink()
        return True
    except OSError:
        return False
