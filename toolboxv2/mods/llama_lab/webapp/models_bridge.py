# file: toolboxv2/mods/llama_lab/webapp/models_bridge.py
"""Model management bridged onto llama_lab (llama.cpp), replacing the old
Ollama-based model_manager. Browse/download/remove come from hub; load/serve/
unload/running from serve; capability flags are derived from the `modality`
already stored per models.ini section.

A "running model" == a served llama-server (OpenAI-compatible). chat/embeddings
are proxied straight to its /v1 base url, so no separate inference path.
"""

import json
import os
import platform

from pathlib import Path

from .. import backend as be          # noqa: F401  (install flows live in the CLI)
from .. import bench as bn            # noqa: F401
from .. import hub
from .. import hw as hwmod
from .. import serve as sv

# modality -> OpenAI-style capability flags
CAPS = {
    "text":   {"text": True,  "vision": False, "audio": False, "embedding": False},
    "vision": {"text": True,  "vision": True,  "audio": False, "embedding": False},
    "omni":   {"text": True,  "vision": True,  "audio": True,  "embedding": False},
    "audio":  {"text": False, "vision": False, "audio": True,  "embedding": False},
    "embed":  {"text": False, "vision": False, "audio": False, "embedding": True},
}

from toolboxv2 import get_app

def _ll_data_dir() -> Path:
    app = get_app()
    for attr in ("data_dir", "appdata", "app_data", "info_dir"):
        d = getattr(app, attr, None)
        if d:
            return Path(d) / "llama_lab"
    if platform.system() == "Windows":
        base = os.environ.get("APPDATA", str(Path.home()))
    else:
        base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(base) / "toolboxv2" / "llama_lab"


def _ll_cfg() -> dict:
    p = _ll_data_dir() / "config.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"bin_dir": "", "ini_path": str(_ll_data_dir() / "models.ini")}


class Models:
    """Thin facade used by the web app; all state lives in llama_lab."""

    def __init__(self):
        self.cfg = _ll_cfg()
        self.data_dir = _ll_data_dir()

    # ---- discovery / catalogue --------------------------------------------

    def hardware(self) -> dict:
        info = hwmod.probe()
        return {"summary": hwmod.summary(info), "vram_gb": info.vram_gb,
                "ram_gb": info.ram_gb, "backend": info.suggested_backend}

    def installed(self):
        return [{"name": p.name, "path": str(p), "size_gb": round(s / 1e9, 2)}
                for p, s in hub.installed()]

    def search_hf(self, query: str):
        info = hwmod.probe()
        out = []
        for repo in hub.search(query, limit=15):
            try:
                c = hub.card(repo)
            except Exception:
                continue
            files = []
            for fn, size in c.gguf_files:
                est = hub.estimate(size, fn, info)
                files.append({"file": fn, "size_gb": round(size / 1e9, 2),
                              "tier": est["tier"], "need_gb": est["need_gb"],
                              "bound": est["bound"], "note": est["note"]})
            out.append({"repo_id": c.repo_id, "downloads": c.downloads, "likes": c.likes,
                        "modality": c.modality, "tags": c.tags[:8], "files": files})
        return out

    def download(self, repo_id: str, filename: str):
        c = hub.card(repo_id)
        path = hub.download(repo_id, filename, with_mmproj=c.has_mmproj)
        section = Path(filename).stem.lower()
        cp = sv.load_ini(Path(self.cfg["ini_path"]))
        if not dict(cp["*"]):
            cp["*"].update(sv.default_section())
        sec = sv.default_section(c.modality)
        sec["model"] = str(path)
        sec["modality"] = c.modality
        mm = next((p for p in Path(path).parent.glob("*mmproj*.gguf")), None)
        if mm:
            sec["mmproj"] = str(mm)
        cp[section] = sec
        sv.save_ini(cp, Path(self.cfg["ini_path"]))
        return {"status": "downloaded", "path": str(path), "modality": c.modality,
                "section": section}

    def remove(self, path: str) -> bool:
        return hub.remove(Path(path))

    # ---- serving (== "loaded models") -------------------------------------

    def running(self):
        live = sv.running(self.data_dir)
        cp = sv.load_ini(Path(self.cfg["ini_path"]))
        out = []
        for name, rec in live.items():
            if name == "__router__":
                continue
            modality = sv.merged(cp, name).get("modality", "text") if name in cp else "text"
            out.append({"name": name, "modality": modality,
                        "capabilities": CAPS.get(modality, CAPS["text"]),
                        "base_url": f"http://{rec['host']}:{rec['port']}/v1",
                        "port": rec["port"], "mode": rec["mode"]})
        return out

    def load(self, name: str, mode: str = "single", port: int = 8080,
             host: str = "127.0.0.1", parallel: int = 4):
        if not self.cfg.get("bin_dir"):
            return {"error": "llama.cpp not installed; run `tb -c llama_lab cli`"}
        cp = sv.load_ini(Path(self.cfg["ini_path"]))
        if name not in cp.sections():
            return {"error": f"no section [{name}] in models.ini"}
        rec = sv.start(Path(self.cfg["bin_dir"]), cp, name, self.data_dir,
                       host, port, mode, parallel)
        return {"status": "loaded", "model": name, "port": rec["port"], "mode": mode}

    def unload(self, name: str) -> dict:
        return {"status": "unloaded" if sv.stop(self.data_dir, name) else "not_running",
                "model": name}

    # ---- routing for inference --------------------------------------------

    def active_models(self):
        return [{"name": m["name"], "type": m["modality"],
                 "capabilities": m["capabilities"]} for m in self.running()]

    def find_for_request(self, model_name=None, needs_vision=False, needs_audio=False):
        run = self.running()
        if model_name:
            for m in run:
                if m["name"] == model_name or model_name.lower() in m["name"].lower():
                    return m
        for m in run:
            c = m["capabilities"]
            if needs_audio and not c["audio"]:
                continue
            if needs_vision and not c["vision"]:
                continue
            if c["text"]:
                return m
        return run[0] if run and not (needs_vision or needs_audio) else None

    def find_embedding(self):
        return next((m for m in self.running() if m["capabilities"]["embedding"]), None)

