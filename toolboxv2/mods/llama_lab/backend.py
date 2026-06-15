# file: toolboxv2/mods/llama_lab/backend.py
"""Install/update llama.cpp for the right backend (Point 1).

Two strategies:
  * prebuilt  -> pull the matching asset from the ggml-org GitHub releases API
                 (cpu / cuda12|13 / vulkan / hip / sycl), Windows CUDA also
                 grabs the cudart zip. macOS Metal uses brew or source.
  * source    -> git clone (official OR a fork+ref) and cmake with -DGGML_<BE>=ON.

Backend keywords are matched against the real asset names, so naming drift in
the release artifacts does not break selection.
"""

import json
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from .hw import HwInfo

REPO = "ggml-org/llama.cpp"
RELEASES = f"https://api.github.com/repos/{REPO}/releases/latest"

# backend -> substrings that must appear in a prebuilt asset name
_BE_KEYS = {
    "cpu": ["cpu"],
    "cuda": ["cuda"],
    "vulkan": ["vulkan"],
    "hip": ["hip", "rocm"],
    "sycl": ["sycl"],
}
_OS_KEYS = {"windows": ["win"], "linux": ["ubuntu", "linux"], "darwin": ["macos"]}

BINARIES = ("llama-server", "llama-bench", "llama-cli", "llama-mtmd-cli")


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "toolboxv2-llama_lab"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def _download(url: str, dest: Path, spinner=None):
    req = urllib.request.Request(url, headers={"User-Agent": "toolboxv2-llama_lab"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if spinner and total:
                spinner.message = f"download {dest.name} {done * 100 // total}%"


def _score_asset(name: str, os_keys, be_keys) -> int:
    n = name.lower()
    if not (n.endswith(".zip") or n.endswith(".tar.gz")):
        return -1
    if "cudart" in n:                       # handled separately
        return -1
    if not any(k in n for k in os_keys):
        return -1
    if not any(k in n for k in be_keys):
        return -1
    return sum(k in n for k in be_keys) + sum(k in n for k in os_keys)


def pick_asset(assets, os_name: str, backend: str, cuda_major: str = "") -> dict:
    os_keys = _OS_KEYS.get(os_name, [os_name])
    be_keys = list(_BE_KEYS.get(backend, [backend]))
    best, best_score = None, 0
    for a in assets:
        s = _score_asset(a["name"], os_keys, be_keys)
        if s <= 0:
            continue
        # Prefer the CUDA major that matches the driver (e.g. "cuda-12").
        if backend == "cuda" and cuda_major and f"cuda-{cuda_major}" in a["name"].lower():
            s += 5
        if s > best_score:
            best, best_score = a, s
    return best


def _extract(archive: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    else:
        with tarfile.open(archive) as t:
            t.extractall(out)


def _find_bin_dir(root: Path) -> Path:
    exe = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    for p in root.rglob(exe):
        return p.parent
    return root


def install_prebuilt(hw: HwInfo, backend: str, dest: Path, spinner=None) -> Path:
    """Download+extract the matching release; return the bin dir (with llama-server)."""
    rel = _get_json(RELEASES)
    assets = rel.get("assets", [])
    cuda_major = hw.cuda_version.split(".")[0] if hw.cuda_version else ""
    asset = pick_asset(assets, hw.os, backend, cuda_major)
    if not asset:
        raise RuntimeError(
            f"no prebuilt asset for os={hw.os} backend={backend} in {rel.get('tag_name')}; "
            f"use source build")
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / asset["name"]
    if spinner:
        spinner.message = f"download {asset['name']}"
    _download(asset["browser_download_url"], archive, spinner)
    _extract(archive, dest)

    # Windows CUDA builds need the matching cudart runtime DLLs alongside.
    if hw.os == "windows" and backend == "cuda":
        cud = next((a for a in assets
                    if "cudart" in a["name"].lower()
                    and (not cuda_major or f"cuda-{cuda_major}" in a["name"].lower())), None)
        if cud:
            cz = dest / cud["name"]
            _download(cud["browser_download_url"], cz, spinner)
            _extract(cz, _find_bin_dir(dest))
    return _find_bin_dir(dest)


def install_source(backend: str, dest: Path, repo: str = "", ref: str = "",
                   spinner=None) -> Path:
    """git clone (official or fork+ref) and cmake build with the backend flag."""
    if not shutil.which("git") or not shutil.which("cmake"):
        raise RuntimeError("source build needs git + cmake on PATH")
    url = repo or f"https://github.com/{REPO}.git"
    src = dest / "src"
    if not src.exists():
        if spinner:
            spinner.message = f"git clone {url}"
        subprocess.run(["git", "clone", "--depth", "1", url, str(src)], check=True)
    if ref:
        subprocess.run(["git", "-C", str(src), "fetch", "--depth", "1", "origin", ref], check=True)
        subprocess.run(["git", "-C", str(src), "checkout", ref], check=True)

    flag = {"cuda": "-DGGML_CUDA=ON", "hip": "-DGGML_HIP=ON",
            "vulkan": "-DGGML_VULKAN=ON", "sycl": "-DGGML_SYCL=ON",
            "metal": "-DGGML_METAL=ON", "cpu": ""}.get(backend, "")
    cfg = ["cmake", "-S", str(src), "-B", str(src / "build"), "-DCMAKE_BUILD_TYPE=Release"]
    if flag:
        cfg.append(flag)
    if spinner:
        spinner.message = "cmake configure"
    subprocess.run(cfg, check=True)
    if spinner:
        spinner.message = "cmake build (this takes a while)"
    subprocess.run(["cmake", "--build", str(src / "build"), "--config", "Release",
                    "-j", str(os.cpu_count() or 4)], check=True)
    return _find_bin_dir(src / "build")


def install_brew(spinner=None) -> Path:
    if not shutil.which("brew"):
        raise RuntimeError("brew not found")
    if spinner:
        spinner.message = "brew install llama.cpp"
    subprocess.run(["brew", "install", "llama.cpp"], check=True)
    return Path(shutil.which("llama-server")).parent


def verify(bin_dir: Path) -> str:
    exe = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    p = bin_dir / exe
    if not p.exists():
        raise RuntimeError(f"llama-server not found in {bin_dir}")
    out = subprocess.run([str(p), "--version"], capture_output=True, text=True, timeout=15)
    return (out.stdout + out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "ok"
