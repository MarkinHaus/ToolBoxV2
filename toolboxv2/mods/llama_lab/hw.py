# file: toolboxv2/mods/llama_lab/hw.py
"""Hardware probe for llama.cpp backend/flag selection.

Stdlib-only with optional psutil. Detects OS/arch, CPU cores, RAM and the
GPU vendor/VRAM, then maps that to the llama.cpp backend that should be
installed (cuda/hip/vulkan/metal/sycl/cpu). Used by backend install, the HF
run-estimator and the bench defaults.
"""

import ctypes
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class Gpu:
    vendor: str            # nvidia | amd | intel | apple
    name: str
    vram_gb: float


@dataclass
class HwInfo:
    os: str                # windows | linux | darwin
    arch: str              # x86_64 | arm64 | ...
    cpu_physical: int
    cpu_logical: int
    ram_gb: float
    gpus: list = field(default_factory=list)
    cuda_version: str = ""     # e.g. "12.4"
    suggested_backend: str = "cpu"

    @property
    def vram_gb(self) -> float:
        """Total VRAM across discrete GPUs; unified memory machines report RAM."""
        if self.gpus:
            return max(g.vram_gb for g in self.gpus)
        return self.ram_gb if self.suggested_backend == "metal" else 0.0


def _run(cmd) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return (out.stdout or "") + (out.stderr or "")
    except Exception:
        return ""


def _ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
        try:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
        except Exception:
            pass
    if platform.system() == "Windows":          # GlobalMemoryStatusEx
        class _MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("mp", ctypes.c_ulong),
                        ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                        ("ae", ctypes.c_ulonglong)]
        ms = _MS(); ms.l = ctypes.sizeof(_MS)
        try:
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return round(ms.tp / 1e9, 1)
        except Exception:
            pass
    return 0.0


def _cpu_physical(logical: int) -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    return max(1, logical // 2) if logical > 1 else 1


def _nvidia():
    if not shutil.which("nvidia-smi"):
        return [], ""
    txt = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in txt.splitlines():
        if "," in line:
            name, mem = line.rsplit(",", 1)
            try:
                gpus.append(Gpu("nvidia", name.strip(), round(float(mem) / 1024, 1)))
            except ValueError:
                pass
    ver = ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", _run(["nvidia-smi"]))
    if m:
        ver = m.group(1)
    return gpus, ver


def _amd():
    if not (shutil.which("rocm-smi") or shutil.which("rocminfo")):
        return []
    txt = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"])
    name = "AMD GPU"
    m = re.search(r"Card series:\s*(.+)", txt) or re.search(r"Card model:\s*(.+)", txt)
    if m:
        name = m.group(1).strip()
    vram = 0.0
    mm = re.search(r"vram.*?Total.*?:\s*(\d+)", txt, re.I | re.S)
    if mm:
        vram = round(int(mm.group(1)) / 1e9, 1)
    return [Gpu("amd", name, vram)]


def probe() -> HwInfo:
    system = platform.system().lower()                 # windows|linux|darwin
    arch = platform.machine().lower()
    logical = os.cpu_count() or 1
    info = HwInfo(os=system, arch=arch,
                  cpu_physical=_cpu_physical(logical),
                  cpu_logical=logical, ram_gb=_ram_gb())

    if system == "darwin" and ("arm" in arch or "aarch" in arch):
        info.gpus = [Gpu("apple", "Apple Silicon (unified)", info.ram_gb)]
        info.suggested_backend = "metal"
        return info

    gpus, cuda = _nvidia()
    if gpus:
        info.gpus, info.cuda_version = gpus, cuda
        info.suggested_backend = "cuda"
        return info

    gpus = _amd()
    if gpus:
        info.gpus = gpus
        # Vulkan is the lower-friction AMD path; HIP/ROCm is opt-in.
        info.suggested_backend = "vulkan"
        return info

    # Intel Arc / iGPU or unknown discrete -> Vulkan if any GPU device exists.
    if system == "linux" and os.path.exists("/dev/dri") and os.listdir("/dev/dri"):
        info.gpus = [Gpu("intel", "Integrated/Arc GPU", 0.0)]
        info.suggested_backend = "vulkan"
        return info

    info.suggested_backend = "cpu"
    return info


def summary(info: HwInfo) -> str:
    g = ", ".join(f"{x.name} ({x.vram_gb:g} GB)" for x in info.gpus) or "no GPU"
    cu = f" CUDA {info.cuda_version}" if info.cuda_version else ""
    return (f"{info.os}/{info.arch} | {info.cpu_physical}C/{info.cpu_logical}T | "
            f"{info.ram_gb:g} GB RAM | {g}{cu} -> backend: {info.suggested_backend}")
