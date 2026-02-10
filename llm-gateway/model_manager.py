"""
Model Manager - Ollama Backend
Manages models via Ollama API (native or Docker).
Supports: text, vision, omni, embedding, tts, audio
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
except ImportError:
    HfApi = None


# Capability matrix per model type
MODEL_CAPABILITIES = {
    "text": {"text": True, "vision": False, "audio": False, "embedding": False, "tts": False},
    "vision": {"text": True, "vision": True, "audio": False, "embedding": False, "tts": False},
    "omni": {"text": True, "vision": True, "audio": True, "embedding": False, "tts": False},
    "embedding": {"text": True, "vision": False, "audio": False, "embedding": True, "tts": False},
    "vision-embedding": {"text": True, "vision": True, "audio": False, "embedding": True, "tts": False},
    "audio": {"text": False, "vision": False, "audio": True, "embedding": False, "tts": False},
    "tts": {"text": False, "vision": False, "audio": False, "embedding": False, "tts": True},
}


def detect_model_type(name: str) -> str:
    """Auto-detect model type from model name/tag."""
    n = name.lower()

    tts_patterns = [
        "tts", "speech", "vocoder", "outetts", "parler", "kokoro",
        "f5-tts", "f5tts", "coqui", "xtts", "bark", "vits",
        "piper", "silero", "speecht5", "tortoise",
    ]
    if any(p in n for p in tts_patterns):
        return "tts"
    if "omni" in n:
        return "omni"
    if "whisper" in n:
        return "audio"

    is_vision = any(x in n for x in ["-vl", "_vl", "vision", "llava", "minicpm-v", "bakllava"])
    is_embed = "embed" in n

    if is_vision and is_embed:
        return "vision-embedding"
    if is_embed:
        return "embedding"
    if is_vision:
        return "vision"
    return "text"


class ModelManager:
    """Manages models via Ollama HTTP API."""

    def __init__(self, config: Dict):
        self.config = config
        self.ollama_url = config.get("ollama_url", "http://127.0.0.1:11434")
        # Loaded model slots: model_name -> slot info
        self.loaded_models: Dict[str, Dict] = {}
        self.models_dir = Path(config.get("models_dir", "data/models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Ollama API helpers
    # ------------------------------------------------------------------

    def _client(self, timeout: float = 30.0) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.ollama_url, timeout=timeout)

    async def _ollama_health(self) -> bool:
        try:
            async with self._client(timeout=5.0) as c:
                r = await c.get("/")
                return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    async def pull_model(self, model_name: str) -> Dict:
        """Pull model from Ollama registry (ollama pull)."""
        async with self._client(timeout=600.0) as c:
            r = await c.post("/api/pull", json={"name": model_name, "stream": False})
            r.raise_for_status()
            return r.json()

    async def load_model(
        self,
        model_name: str,
        model_type: str = "auto",
        keep_alive: str = "-1",
    ) -> Dict:
        """Load / warm-up a model in Ollama so it stays in VRAM/RAM."""
        if model_type == "auto":
            model_type = detect_model_type(model_name)

        if model_type not in MODEL_CAPABILITIES:
            raise ValueError(f"Invalid model_type '{model_type}'. Must be: {list(MODEL_CAPABILITIES.keys())}")

        # Check Ollama is reachable
        if not await self._ollama_health():
            raise RuntimeError(f"Ollama not reachable at {self.ollama_url}")

        # Ensure model exists locally (pull if needed)
        local_models = await self.list_ollama_models()
        local_names = [m["name"] for m in local_models]
        # Normalise: ollama tags may include :latest
        norm = model_name if ":" in model_name else f"{model_name}:latest"
        if norm not in local_names and model_name not in local_names:
            await self.pull_model(model_name)

        # Warm-up: send a tiny generate request with keep_alive to load into memory
        if model_type == "embedding":
            async with self._client(timeout=300.0) as c:
                r = await c.post("/api/embed", json={
                    "model": model_name,
                    "input": "warmup",
                    "keep_alive": keep_alive,
                })
                r.raise_for_status()
        else:
            async with self._client(timeout=300.0) as c:
                r = await c.post("/api/generate", json={
                    "model": model_name,
                    "prompt": "hi",
                    "options": {"num_predict": 1},
                    "keep_alive": keep_alive,
                    "stream": False,
                })
                r.raise_for_status()

        info = {
            "name": model_name,
            "model_type": model_type,
            "capabilities": MODEL_CAPABILITIES[model_type],
            "keep_alive": keep_alive,
            "status": "running",
        }
        self.loaded_models[model_name] = info

        # Persist to config
        self._save_loaded_config()

        return {"status": "loaded", "model": model_name, "model_type": model_type}

    async def unload_model(self, model_name: str) -> Dict:
        """Unload model from Ollama memory."""
        try:
            async with self._client(timeout=30.0) as c:
                await c.post("/api/generate", json={
                    "model": model_name,
                    "prompt": "",
                    "keep_alive": "0",
                    "stream": False,
                })
        except Exception:
            pass

        self.loaded_models.pop(model_name, None)
        self._save_loaded_config()
        return {"status": "unloaded", "model": model_name}

    async def delete_model(self, model_name: str) -> Dict:
        """Delete model from Ollama."""
        # Unload first
        await self.unload_model(model_name)
        async with self._client(timeout=60.0) as c:
            r = await c.request("DELETE", "/api/delete", json={"name": model_name})
            r.raise_for_status()
        return {"status": "deleted", "model": model_name}

    async def shutdown_all(self):
        """Unload all loaded models."""
        for name in list(self.loaded_models.keys()):
            try:
                await self.unload_model(name)
            except Exception as e:
                print(f"Error unloading {name}: {e}")

    async def restore_models(self):
        """Restore models from config on startup."""
        models = self.config.get("loaded_models", {})
        for name, info in models.items():
            try:
                await self.load_model(
                    model_name=name,
                    model_type=info.get("model_type", "auto"),
                    keep_alive=info.get("keep_alive", "-1"),
                )
            except Exception as e:
                print(f"Failed to restore model {name}: {e}")

    # ------------------------------------------------------------------
    # Ollama model listing
    # ------------------------------------------------------------------

    async def list_ollama_models(self) -> List[Dict]:
        """List models available in Ollama."""
        async with self._client() as c:
            r = await c.get("/api/tags")
            r.raise_for_status()
            data = r.json()
        models = []
        for m in data.get("models", []):
            models.append({
                "name": m.get("name", ""),
                "size_gb": round(m.get("size", 0) / (1024 ** 3), 2),
                "modified_at": m.get("modified_at", ""),
                "family": m.get("details", {}).get("family", ""),
                "parameter_size": m.get("details", {}).get("parameter_size", ""),
                "quantization": m.get("details", {}).get("quantization_level", ""),
                "format": m.get("details", {}).get("format", ""),
            })
        return models

    async def list_running_models(self) -> List[Dict]:
        """List models currently loaded in Ollama memory."""
        try:
            async with self._client() as c:
                r = await c.get("/api/ps")
                r.raise_for_status()
                data = r.json()
            return data.get("models", [])
        except Exception:
            return []

    async def search_ollama_library(self, query: str) -> List[Dict]:
        """Search Ollama library via the website (scrape)."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                r = await c.get(f"https://ollama.com/search?q={query}")
                if r.status_code != 200:
                    return []
                # Parse basic info from the page - simplified
                # We return a minimal hint; the admin UI can link to ollama.com
                return [{"query": query, "hint": "Use 'ollama pull <model>' or enter model name to load"}]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Model import from GGUF / HuggingFace
    # ------------------------------------------------------------------

    async def import_gguf(self, gguf_path: str, model_name: str) -> Dict:
        """Import a local GGUF file into Ollama via Modelfile."""
        p = Path(gguf_path)
        if not p.exists():
            # Try relative to models_dir
            p = self.models_dir / gguf_path
        if not p.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        # Create a Modelfile
        modelfile_content = f'FROM "{p.resolve()}"\n'
        async with self._client(timeout=600.0) as c:
            r = await c.post("/api/create", json={
                "name": model_name,
                "modelfile": modelfile_content,
                "stream": False,
            })
            r.raise_for_status()
        return {"status": "imported", "model": model_name, "source": str(p)}

    async def download_hf_model(self, repo_id: str, filename: str) -> Dict:
        """Download GGUF from HuggingFace to local models dir."""
        if HfApi is None:
            return {"error": "huggingface_hub not installed"}

        hf_token = self.config.get("hf_token")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(self.models_dir),
                token=hf_token,
            )
            return {
                "status": "downloaded",
                "path": str(Path(local_path).relative_to(self.models_dir)),
                "repo": repo_id,
                "filename": filename,
            }
        except Exception as e:
            return {"error": str(e)}

    async def search_hf_models(self, query: str) -> Any:
        """Search HuggingFace for GGUF models."""
        if HfApi is None:
            return {"error": "huggingface_hub not installed"}

        api = HfApi()
        try:
            models = api.list_models(search=query, sort="downloads", direction=-1, limit=20)
            results = []
            for model in models:
                try:
                    files = list_repo_files(model.id)
                    model_files = [f for f in files if f.endswith(".gguf")]
                    if model_files:
                        results.append({
                            "repo_id": model.id,
                            "downloads": model.downloads,
                            "likes": model.likes,
                            "files": model_files[:15],
                        })
                except Exception:
                    pass
                if len(results) >= 10:
                    break
            return results
        except Exception as e:
            return {"error": str(e)}

    def list_local_gguf(self) -> List[Dict]:
        """List GGUF files in local models directory."""
        models = []
        for f in self.models_dir.glob("**/*.gguf"):
            size_mb = f.stat().st_size / (1024 ** 2)
            models.append({
                "name": f.name,
                "path": str(f.relative_to(self.models_dir)),
                "size_mb": round(size_mb, 1),
                "size_gb": round(size_mb / 1024, 2),
                "detected_type": detect_model_type(f.name),
            })
        return sorted(models, key=lambda x: x["name"])

    # ------------------------------------------------------------------
    # Smart model finding
    # ------------------------------------------------------------------

    def find_model(self, model_name: str) -> Optional[Dict]:
        """Find loaded model by name (exact or partial match)."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        # Partial match
        for name, info in self.loaded_models.items():
            if model_name.lower() in name.lower() or name.lower() in model_name.lower():
                return info
        return None

    def find_model_for_request(
        self,
        model_name: Optional[str] = None,
        needs_audio: bool = False,
        needs_vision: bool = False,
        needs_embedding: bool = False,
    ) -> Optional[Dict]:
        """Smart model finding based on request requirements."""
        # 1. Exact match
        if model_name:
            m = self.find_model(model_name)
            if m:
                caps = m.get("capabilities", {})
                if needs_audio and not caps.get("audio"):
                    pass
                elif needs_vision and not caps.get("vision"):
                    pass
                elif needs_embedding and not caps.get("embedding"):
                    pass
                else:
                    return m

        # 2. By capability
        for info in self.loaded_models.values():
            caps = info.get("capabilities", {})
            if needs_embedding and caps.get("embedding"):
                return info
            if needs_audio and caps.get("audio"):
                return info
            if needs_audio and not caps.get("audio"):
                continue
            if needs_vision and not caps.get("vision"):
                continue
            if caps.get("text"):
                return info
        return None

    def find_text_model(self) -> Optional[Dict]:
        for info in self.loaded_models.values():
            if info.get("capabilities", {}).get("text"):
                return info
        return None

    def find_vision_model(self) -> Optional[Dict]:
        for info in self.loaded_models.values():
            if info.get("capabilities", {}).get("vision"):
                return info
        return None

    def find_audio_model(self) -> Optional[Dict]:
        for info in self.loaded_models.values():
            if info.get("capabilities", {}).get("audio"):
                return info
        return None

    def find_embedding_model(self) -> Optional[Dict]:
        for info in self.loaded_models.values():
            if info.get("capabilities", {}).get("embedding"):
                return info
        return None

    def find_tts_model(self) -> Optional[Dict]:
        for info in self.loaded_models.values():
            if info.get("capabilities", {}).get("tts"):
                return info
        return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_active_models(self) -> List[Dict]:
        return [
            {
                "name": info["name"],
                "type": info["model_type"],
                "capabilities": info.get("capabilities", {}),
            }
            for info in self.loaded_models.values()
        ]

    def get_models_status(self) -> List[Dict]:
        result = []
        for name, info in self.loaded_models.items():
            result.append({
                "model_name": info["name"],
                "model_type": info["model_type"],
                "capabilities": info.get("capabilities", {}),
                "status": info.get("status", "unknown"),
                "keep_alive": info.get("keep_alive", "-1"),
            })
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_loaded_config(self):
        self.config["loaded_models"] = {
            name: {
                "model_type": info["model_type"],
                "keep_alive": info.get("keep_alive", "-1"),
            }
            for name, info in self.loaded_models.items()
        }
        config_path = Path(self.config.get("_config_path", "data/config.json"))
        config_path.write_text(json.dumps(self.config, indent=2))
