# file: toolboxv2/mods/llama_lab/serve.py
"""Hosting layer (Point 4) — models.ini is the single source of truth.

A section in models.ini == one served model. Keys are llama-server long-flags
without the leading '--' (ctx-size, ngl, mmproj, embeddings, parallel, ...),
plus an extra `modality` key (text|embed|vision|audio|omni) so embeddings,
multimodal and full Omni models are first-class, not glued on.

Two serving modes overlay the section:
  * single : --parallel 1  -> one model instance tuned for max prefill+token
             throughput of a single stream.
  * mass   : --parallel N --cont-batching -> N slots, many concurrent requests.

Multi-model hot-swap (your guide's "router mode") is delegated to llama-swap
when present: we generate its config.yaml from the SAME models.ini. Without
llama-swap we spawn one llama-server for the chosen model.
"""

import configparser
import json
import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path

# flags that are bare switches (presence only) when truthy in the ini
_SWITCHES = {"flash-attn", "embeddings", "cont-batching", "mlock", "no-mmproj-offload",
             "no-webui", "no-slots", "no-mmproj"}
# ini key -> short flag override (else rendered as --<key>)
_SHORT = {"ngl": "-ngl", "model": "-m", "threads": "-t", "ctx-size": "-c",
          "parallel": "-np", "hf": "-hf", "flash-attn": "-fa"}
_NON_SERVER_KEYS = {"modality"}     # consumed by us, not passed to llama-server


def _exe(bin_dir: Path, name: str) -> str:
    win = bin_dir / f"{name}.exe"
    return str(win if win.exists() else bin_dir / name)


# --------------------------------------------------------------- models.ini --

def load_ini(path: Path) -> configparser.ConfigParser:
    cp = configparser.ConfigParser()
    cp.optionxform = str                 # keep flag case
    if Path(path).exists():
        cp.read(path)
    if "*" not in cp:
        cp["*"] = {}
    return cp


def save_ini(cp: configparser.ConfigParser, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        cp.write(f)


def merged(cp: configparser.ConfigParser, name: str) -> dict:
    """Section merged over [*] defaults."""
    out = dict(cp["*"]) if "*" in cp else {}
    out.update(dict(cp[name]))
    return out


def default_section(modality: str = "text") -> dict:
    """Sane defaults straight from the tuning guide ([*] block)."""
    base = {"ctx-size": "64000", "batch-size": "2048", "ubatch-size": "1024",
            "ngl": "99", "flash-attn": "on", "cache-type-k": "q8_0",
            "cache-type-v": "q8_0", "modality": modality}
    if modality == "embed":
        base.update({"embeddings": "1", "ngl": "99"})
        base.pop("cache-type-k", None)
        base.pop("cache-type-v", None)
    return base


# ----------------------------------------------------------- compile argv --

def to_argv(bin_dir: Path, cfg: dict, host: str, port: int, mode: str = "single",
            parallel: int = 4) -> list:
    """Translate a merged section into a real llama-server command line."""
    cfg = dict(cfg)
    modality = cfg.pop("modality", "text")

    # mode overlay
    if mode == "single":
        cfg["parallel"] = "1"
    else:                                # mass
        cfg["parallel"] = str(parallel)
        cfg["cont-batching"] = "1"

    if modality == "embed":
        cfg["embeddings"] = "1"
    # vision/audio/omni: an mmproj key (or -hf auto-detect) is all libmtmd needs.

    argv = [_exe(bin_dir, "llama-server"), "--host", host, "--port", str(port)]
    for key, val in cfg.items():
        if key in _NON_SERVER_KEYS:
            continue
        flag = _SHORT.get(key, f"--{key}")
        v = str(val).strip()
        if key in _SWITCHES:
            if v.lower() in ("on", "1", "true", "yes"):
                argv.append(flag if not flag.startswith("--no-") else flag)
            continue
        if v.lower() == "on" and key == "flash-attn":
            argv += [flag, "on"]
            continue
        argv += [flag, v]
    return argv


# ---------------------------------------------------------------- llama-swap --

def write_swap_config(bin_dir: Path, cp: configparser.ConfigParser, out: Path,
                      base_port: int = 8100, mode: str = "single", parallel: int = 4):
    """Generate a llama-swap config.yaml from models.ini (router/hot-swap)."""
    lines = ["models:"]
    port = base_port
    for name in cp.sections():
        if name == "*":
            continue
        argv = to_argv(bin_dir, merged(cp, name), "127.0.0.1", port, mode, parallel)
        cmd = " ".join(_q(a) for a in argv)
        lines += [f'  "{name}":', "    cmd: |", f"      {cmd}",
                  f"    proxy: http://127.0.0.1:{port}"]
        port += 1
    Path(out).write_text("\n".join(lines) + "\n")
    return out


def _q(s: str) -> str:
    return f'"{s}"' if " " in s else s


def has_swap() -> bool:
    return shutil.which("llama-swap") is not None


# ---------------------------------------------------------- process mgmt --

def _state_file(data_dir: Path) -> Path:
    return Path(data_dir) / "running.json"


def _read_state(data_dir: Path) -> dict:
    p = _state_file(data_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _write_state(data_dir: Path, st: dict):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    _state_file(data_dir).write_text(json.dumps(st, indent=2))


def start(bin_dir: Path, cp, name: str, data_dir: Path, host="0.0.0.0",
          port=8080, mode="single", parallel=4, log_dir: Path = None) -> dict:
    """Spawn a single llama-server for `name`. Returns the run record."""
    argv = to_argv(bin_dir, merged(cp, name), host, port, mode, parallel)
    log_dir = Path(log_dir or (Path(data_dir) / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logf = open(log_dir / f"{name}.log", "ab")
    creat = subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0
    proc = subprocess.Popen(argv, stdout=logf, stderr=logf, creationflags=creat) \
        if platform.system() == "Windows" else \
        subprocess.Popen(argv, stdout=logf, stderr=logf, start_new_session=True)
    rec = {"name": name, "pid": proc.pid, "port": port, "host": host,
           "mode": mode, "started": time.time(), "argv": argv}
    st = _read_state(data_dir)
    st[name] = rec
    _write_state(data_dir, st)
    return rec


def start_router(cp, data_dir: Path, swap_cfg: Path, host="0.0.0.0", port=8080) -> dict:
    """Launch llama-swap over the generated config (multi-model auto load/unload)."""
    argv = ["llama-swap", "--config", str(swap_cfg), "--listen", f"{host}:{port}"]
    log_dir = Path(data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logf = open(log_dir / "router.log", "ab")
    proc = subprocess.Popen(argv, stdout=logf, stderr=logf,
                            start_new_session=(platform.system() != "Windows"))
    rec = {"name": "__router__", "pid": proc.pid, "port": port, "host": host,
           "mode": "router", "started": time.time(), "argv": argv}
    st = _read_state(data_dir)
    st["__router__"] = rec
    _write_state(data_dir, st)
    return rec


def running(data_dir: Path) -> dict:
    """Prune dead PIDs and return live run records."""
    st = _read_state(data_dir)
    live = {}
    for name, rec in st.items():
        if _alive(rec["pid"]):
            live[name] = rec
    if live != st:
        _write_state(data_dir, live)
    return live


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return True


def stop(data_dir: Path, name: str) -> bool:
    st = _read_state(data_dir)
    rec = st.get(name)
    if not rec:
        return False
    try:
        if platform.system() == "Windows":
            os.kill(rec["pid"], signal.SIGTERM)
        else:
            os.killpg(os.getpgid(rec["pid"]), signal.SIGTERM)
    except Exception:
        pass
    st.pop(name, None)
    _write_state(data_dir, st)
    return True
