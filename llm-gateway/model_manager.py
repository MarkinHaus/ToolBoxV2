"""
Model Manager - Dynamic model loading/unloading with HF integration
Supports: text, vision, omni (audio+vision), embedding, audio (legacy whisper)
"""

import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import psutil

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
except ImportError:
    HfApi = None


class ModelManager:
    """Manages llama-server processes for various model types"""

    SLOT_PORTS = list(range(4801, 4808))  # 4801-4807

    # Model type capabilities
    MODEL_CAPABILITIES = {
        "text": {"text": True, "vision": False, "audio": False, "embedding": False},
        "vision": {"text": True, "vision": True, "audio": False, "embedding": False},
        "omni": {"text": True, "vision": True, "audio": True, "embedding": False},
        "embedding": {"text": True, "vision": False, "audio": False, "embedding": True},
        "vision-embedding": {"text": True, "vision": True, "audio": False, "embedding": True},
        "audio": {"text": False, "vision": False, "audio": True, "embedding": False},  # Legacy whisper
    }

    def __init__(self, base_dir: Path, models_dir: Path, config: Dict):
        self.base_dir = base_dir
        self.models_dir = models_dir
        self.config = config
        self.processes: Dict[int, subprocess.Popen] = {}
        self.slot_info: Dict[int, Dict] = {}

        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Binaries
        self.llama_server = base_dir / "llama-server"
        self.whisper_server = base_dir / "whisper-server"  # Legacy

    async def restore_slots(self):
        """Restore slots from config on startup"""
        slots = self.config.get("slots", {})
        for port_str, slot_config in slots.items():
            if slot_config and slot_config.get("model_path"):
                try:
                    await self.load_model(
                        slot=int(port_str),
                        model_path=slot_config["model_path"],
                        model_type=slot_config.get("model_type", "text"),
                        ctx_size=slot_config.get("ctx_size"),
                        threads=slot_config.get("threads")
                    )
                except Exception as e:
                    print(f"Failed to restore slot {port_str}: {e}")

    async def load_model(
        self,
        slot: int,
        model_path: str,
        model_type: str = "text",
        ctx_size: Optional[int] = None,
        threads: Optional[int] = None,
        mmproj_path: Optional[str] = None
    ) -> Dict:
        """Load a model into a slot"""

        if slot not in self.SLOT_PORTS:
            raise ValueError(f"Invalid slot {slot}. Must be 4801-4807")

        if model_type not in self.MODEL_CAPABILITIES:
            raise ValueError(f"Invalid model_type '{model_type}'. Must be: {list(self.MODEL_CAPABILITIES.keys())}")

        # Unload existing if any
        if slot in self.processes:
            await self.unload_model(slot)

        # Resolve model path
        if model_path.startswith("/") or model_path.startswith("./"):
            full_path = Path(model_path)
        else:
            full_path = self.models_dir / model_path

        if not full_path.exists():
            raise FileNotFoundError(f"Model not found: {full_path}")

        # Build command
        ctx = ctx_size or self.config.get("default_ctx_size", 8192)
        thr = threads or self.config.get("default_threads", 10)

        # Legacy whisper-server for audio-only
        if model_type == "audio":
            cmd = self._build_whisper_command(full_path, slot, thr)
        else:
            cmd = self._build_llama_command(full_path, slot, ctx, thr, model_type, mmproj_path)

        # Start process
        print(f"Starting model: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        # Wait for server to be ready
        # Large models on CPU need more time to load:
        # - text: 180s (3 min)
        # - embedding: 600s (10 min) - 8B embedding models are slow on CPU
        # - vision/omni/vision-embedding: 300s (5 min)
        if model_type in ("embedding",):
            timeout = 600  # 10 minutes for large embedding models
        elif model_type in ("omni", "vision", "vision-embedding"):
            timeout = 300  # 5 minutes
        else:
            timeout = 180  # 3 minutes
        ready = await self._wait_for_ready(slot, timeout=timeout, model_type=model_type)

        if not ready:
            # Get stderr for debugging
            try:
                _, stderr = process.communicate(timeout=2)
                error_msg = stderr.decode()[:1000] if stderr else "No error output"
            except:
                error_msg = "Could not get error output"

            process.kill()
            raise RuntimeError(f"Model failed to start on port {slot}. Error: {error_msg}")

        self.processes[slot] = process
        self.slot_info[slot] = {
            "port": slot,
            "model_path": str(full_path),
            "model_name": full_path.stem,
            "model_type": model_type,
            "ctx_size": ctx,
            "threads": thr,
            "pid": process.pid,
            "capabilities": self.MODEL_CAPABILITIES[model_type]
        }

        # Update config
        self._save_slot_config()

        return {
            "status": "loaded",
            "slot": slot,
            "model": full_path.stem,
            "model_type": model_type,
            "pid": process.pid
        }

    def _build_llama_command(
        self,
        model_path: Path,
        slot: int,
        ctx: int,
        threads: int,
        model_type: str,
        mmproj_path: Optional[str] = None
    ) -> List[str]:
        """Build llama-server command for text/vision/omni/embedding models"""

        if not self.llama_server.exists():
            raise FileNotFoundError("llama-server not found. Run setup.sh")

        # Get optimization settings from config
        perf_config = self.config.get("performance", {})
        use_flash_attn = perf_config.get("flash_attention", True)
        use_mlock = perf_config.get("mlock", True)
        kv_cache_quant = perf_config.get("kv_cache_quantization", "q8_0")  # q8_0, q4_0, f16
        batch_size = perf_config.get("batch_size", 512)

        cmd = [
            str(self.llama_server),
            "--model", str(model_path),
            "--port", str(slot),
            "--threads", str(threads),
            "--threads-batch", str(threads * 2),  # More threads for prompt eval
            "--host", "127.0.0.1",
        ]

        # ⚡ Flash Attention - significant speedup (20-30%)
        if use_flash_attn:
            cmd.append("--flash-attn")

        # 🔒 mlock - prevents swapping, keeps model in RAM
        if use_mlock:
            cmd.append("--mlock")

        # 💾 KV-Cache quantization - saves memory, allows larger context
        if kv_cache_quant and kv_cache_quant != "f16":
            cmd.extend(["--cache-type-k", kv_cache_quant])
            cmd.extend(["--cache-type-v", kv_cache_quant])

        # Embedding models (including vision-embedding)
        if model_type in ("embedding", "vision-embedding"):
            # For embedding models:
            # - ubatch-size must be >= batch-size (non-causal attention requirement)
            # - Use smaller ctx for embeddings (saves memory)
            # - Use "last" pooling for Qwen embedding models
            embed_batch = min(ctx, 2048)  # Reasonable default
            cmd.extend([
                "--ctx-size", str(embed_batch),
                "--batch-size", str(embed_batch),
                "--ubatch-size", str(embed_batch),  # Must be >= batch-size for embeddings
                "--embedding",
                "--pooling", "last",  # "last" works best for Qwen embeddings
            ])
        else:
            # Chat models get parallel processing, batching, and jinja
            cmd.extend([
                "--ctx-size", str(ctx),
                "--batch-size", str(batch_size),
                "--ubatch-size", str(batch_size),
                "--parallel", "2",
                "--cont-batching",
                "--jinja",
            ])

        # Vision, Omni, and Vision-Embedding models need mmproj
        if model_type in ("vision", "omni", "vision-embedding"):
            # Use manually specified mmproj if provided, otherwise auto-detect
            if mmproj_path:
                mmproj = Path(mmproj_path)
                if not mmproj.exists():
                    raise FileNotFoundError(f"Specified mmproj file not found: {mmproj_path}")
            else:
                mmproj = self._find_mmproj(model_path)

            if mmproj:
                cmd.extend(["--mmproj", str(mmproj)])
            else:
                raise FileNotFoundError(
                    f"{model_type.title()} model requires mmproj file. "
                    f"Download mmproj-*.gguf for this model and place it in the same directory."
                )

        return cmd

    def _build_whisper_command(self, model_path: Path, slot: int, threads: int) -> List[str]:
        """Build whisper-server command for legacy audio models"""

        if not self.whisper_server.exists():
            raise FileNotFoundError(
                "whisper-server not found. Consider using an 'omni' model instead, "
                "or run setup.sh to install whisper-server."
            )

        return [
            str(self.whisper_server),
            "--model", str(model_path),
            "--port", str(slot),
            "--threads", str(threads),
            "--host", "127.0.0.1"
        ]

    def _find_mmproj(self, model_path: Path) -> Optional[Path]:
        """Find mmproj file for vision/omni model"""
        model_dir = model_path.parent
        model_name = model_path.stem

        # Extract base model name (remove quantization suffix)
        # e.g., "Qwen2.5-Omni-7B-Q4_K_M" -> "Qwen2.5-Omni-7B"
        base_parts = []
        for part in model_name.split("-"):
            if part.startswith("Q") and "_" in part:  # Q4_K_M, Q8_0, etc.
                break
            base_parts.append(part)
        base_name = "-".join(base_parts) if base_parts else model_name

        # Search patterns (ordered by specificity)
        patterns = [
            f"mmproj-{base_name}*.gguf",
            f"mmproj-{model_name}*.gguf",
            f"mmproj*.gguf",
            f"{model_name}.mmproj.gguf",
            f"{base_name}*.mmproj*.gguf",
            "*mmproj*.gguf",
        ]

        # Search in model directory first
        for pattern in patterns:
            matches = list(model_dir.glob(pattern))
            if matches:
                # Prefer F16 or Q8 quality
                for quality in ["F16", "Q8", "F32"]:
                    for m in matches:
                        if quality in m.name:
                            return m
                return matches[0]

        # Also check in models_dir root
        for pattern in patterns:
            matches = list(self.models_dir.glob(pattern))
            if matches:
                for quality in ["F16", "Q8", "F32"]:
                    for m in matches:
                        if quality in m.name:
                            return m
                return matches[0]

        return None

    async def _wait_for_ready(self, port: int, timeout: int = 180, model_type: str = "text") -> bool:
        """Wait for server to respond to health check"""
        import httpx

        # Different endpoints for different servers
        if model_type == "audio":
            # whisper-server endpoints
            endpoints = [
                f"http://127.0.0.1:{port}/",
                f"http://127.0.0.1:{port}/health",
            ]
        else:
            # llama-server endpoints
            endpoints = [f"http://127.0.0.1:{port}/health"]

        start = asyncio.get_event_loop().time()
        last_status = None

        while asyncio.get_event_loop().time() - start < timeout:
            for url in endpoints:
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(url, timeout=5.0)
                        last_status = resp.status_code

                        # 200 = ready, 405 = method not allowed (server is up)
                        if resp.status_code in (200, 405):
                            return True

                        # 503 = Service Unavailable = still loading model
                        # This is normal, keep waiting
                        if resp.status_code == 503:
                            pass  # Continue waiting

                except httpx.TimeoutException:
                    pass  # Connection timeout, keep trying
                except httpx.ConnectError:
                    pass  # Server not yet listening, keep trying
                except Exception:
                    pass

            await asyncio.sleep(2.0)  # Longer sleep for big models

        print(f"Health check timeout. Last status: {last_status}")
        return False

    async def unload_model(self, slot: int) -> Dict:
        """Unload model from slot"""

        if slot not in self.processes:
            return {"status": "not_loaded", "slot": slot}

        process = self.processes[slot]

        try:
            # Graceful shutdown
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

        del self.processes[slot]
        if slot in self.slot_info:
            del self.slot_info[slot]

        self._save_slot_config()

        return {"status": "unloaded", "slot": slot}

    async def shutdown_all(self):
        """Shutdown all running model processes"""
        for slot in list(self.processes.keys()):
            try:
                await self.unload_model(slot)
            except Exception as e:
                print(f"Error shutting down slot {slot}: {e}")

    def _save_slot_config(self):
        """Persist slot configuration"""
        slots = {}
        for port in self.SLOT_PORTS:
            if port in self.slot_info:
                info = self.slot_info[port]
                slots[str(port)] = {
                    "model_path": info["model_path"],
                    "model_type": info["model_type"],
                    "ctx_size": info["ctx_size"],
                    "threads": info["threads"]
                }
            else:
                slots[str(port)] = None

        self.config["slots"] = slots
        config_path = self.base_dir / "data" / "config.json"
        config_path.write_text(json.dumps(self.config, indent=2))

    # === Model Finding Methods ===

    def find_model_slot(self, model_name: str) -> Optional[Dict]:
        """Find slot by model name (exact or partial match)"""
        for info in self.slot_info.values():
            if model_name.lower() in info["model_name"].lower() or \
               info["model_name"].lower() in model_name.lower():
                return info
        return None

    def find_slot_for_request(
        self,
        model_name: Optional[str] = None,
        needs_audio: bool = False,
        needs_vision: bool = False,
        needs_embedding: bool = False
    ) -> Optional[Dict]:
        """
        Smart slot finding based on request requirements.

        Priority:
        1. Exact model name match
        2. Model with required capabilities
        3. Any suitable model
        """

        # 1. Try exact model match first
        if model_name:
            slot = self.find_model_slot(model_name)
            if slot:
                # Verify capabilities
                caps = slot.get("capabilities", {})
                if needs_audio and not caps.get("audio"):
                    pass  # Continue searching
                elif needs_vision and not caps.get("vision"):
                    pass  # Continue searching
                elif needs_embedding and not caps.get("embedding"):
                    pass  # Continue searching
                else:
                    return slot

        # 2. Find by capabilities
        for info in self.slot_info.values():
            caps = info.get("capabilities", {})

            if needs_embedding:
                if caps.get("embedding"):
                    return info
                continue

            if needs_audio and not caps.get("audio"):
                continue
            if needs_vision and not caps.get("vision"):
                continue

            # Text capability is implied for non-embedding requests
            if caps.get("text"):
                return info

        return None

    def find_audio_slot(self) -> Optional[Dict]:
        """Find slot capable of audio processing (omni or audio)"""
        for info in self.slot_info.values():
            caps = info.get("capabilities", {})
            if caps.get("audio"):
                return info
        return None

    def find_vision_slot(self) -> Optional[Dict]:
        """Find slot capable of vision processing (omni or vision)"""
        for info in self.slot_info.values():
            caps = info.get("capabilities", {})
            if caps.get("vision"):
                return info
        return None

    def find_embedding_slot(self, needs_vision: bool = False) -> Optional[Dict]:
        """Find slot for embeddings, optionally with vision capability"""
        # If vision is needed, try to find vision-embedding first
        if needs_vision:
            for info in self.slot_info.values():
                caps = info.get("capabilities", {})
                if caps.get("embedding") and caps.get("vision"):
                    return info

        # Fallback to any embedding slot
        for info in self.slot_info.values():
            caps = info.get("capabilities", {})
            if caps.get("embedding"):
                return info
        return None

    def find_vision_embedding_slot(self) -> Optional[Dict]:
        """Find slot for vision embeddings (vision-embedding type)"""
        for info in self.slot_info.values():
            caps = info.get("capabilities", {})
            if caps.get("embedding") and caps.get("vision"):
                return info
        return None

    def find_text_slot(self) -> Optional[Dict]:
        """Find any slot capable of text chat"""
        for info in self.slot_info.values():
            caps = info.get("capabilities", {})
            if caps.get("text"):
                return info
        return None

    # === Status Methods ===

    def get_slots_status(self) -> List[Dict]:
        """Get status of all slots"""
        result = []

        for port in self.SLOT_PORTS:
            if port in self.slot_info:
                info = self.slot_info[port].copy()

                # Add process stats
                try:
                    proc = psutil.Process(info["pid"])
                    mem_info = proc.memory_info()
                    info["ram_mb"] = round(mem_info.rss / (1024**2), 1)
                    info["cpu_percent"] = proc.cpu_percent()
                    info["status"] = "running"
                except psutil.NoSuchProcess:
                    info["status"] = "crashed"
                    info["ram_mb"] = 0

                result.append(info)
            else:
                result.append({
                    "port": port,
                    "status": "empty",
                    "model_name": None,
                    "model_type": None,
                    "ram_mb": 0
                })

        return result

    def get_active_models(self) -> List[Dict]:
        """Get list of active models with capabilities"""
        return [
            {
                "name": info["model_name"],
                "type": info["model_type"],
                "port": info["port"],
                "capabilities": info.get("capabilities", {})
            }
            for info in self.slot_info.values()
        ]

    def get_process_stats(self) -> List[Dict]:
        """Get detailed process stats for all slots"""
        stats = []

        for port in self.SLOT_PORTS:
            if port in self.slot_info:
                info = self.slot_info[port]
                try:
                    proc = psutil.Process(info["pid"])
                    mem = proc.memory_info()
                    stats.append({
                        "port": port,
                        "model": info["model_name"],
                        "model_type": info["model_type"],
                        "pid": info["pid"],
                        "ram_mb": round(mem.rss / (1024**2), 1),
                        "ram_percent": round(mem.rss / psutil.virtual_memory().total * 100, 1),
                        "cpu_percent": proc.cpu_percent(),
                        "threads": len(proc.threads()),
                        "status": "running"
                    })
                except psutil.NoSuchProcess:
                    stats.append({
                        "port": port,
                        "model": info["model_name"],
                        "status": "crashed"
                    })
            else:
                stats.append({
                    "port": port,
                    "status": "empty"
                })

        return stats

    # === Local Model Management ===

    def list_local_models(self) -> List[Dict]:
        """List downloaded models with detected type"""
        models = []

        for f in self.models_dir.glob("**/*.gguf"):
            size_mb = f.stat().st_size / (1024**2)
            detected_type = self._detect_model_type(f.name)

            models.append({
                "name": f.name,
                "path": str(f.relative_to(self.models_dir)),
                "size_mb": round(size_mb, 1),
                "size_gb": round(size_mb / 1024, 2),
                "detected_type": detected_type
            })

        # Also check for whisper models (.bin)
        for f in self.models_dir.glob("**/*.bin"):
            size_mb = f.stat().st_size / (1024**2)
            models.append({
                "name": f.name,
                "path": str(f.relative_to(self.models_dir)),
                "size_mb": round(size_mb, 1),
                "size_gb": round(size_mb / 1024, 2),
                "detected_type": "audio"
            })

        return sorted(models, key=lambda x: x["name"])

    def _detect_model_type(self, filename: str) -> str:
        """Auto-detect model type from filename"""
        name = filename.lower()

        # Check for mmproj (not a loadable model)
        if "mmproj" in name:
            return "mmproj"

        # Omni models (audio + vision + text)
        if "omni" in name:
            return "omni"

        # Audio-only (whisper)
        if "whisper" in name or (name.startswith("ggml-") and any(
            x in name for x in ["large", "medium", "small", "base", "tiny"]
        )):
            return "audio"

        # Vision-Embedding models (VL + Embedding)
        is_vision = "-vl" in name or "_vl" in name or "vision" in name or "minicpm-v" in name
        is_embedding = "embed" in name

        if is_vision and is_embedding:
            return "vision-embedding"

        # Pure embedding models
        if is_embedding:
            return "embedding"

        # Vision models (VL = Vision-Language)
        if is_vision:
            return "vision"

        # Default to text
        return "text"

    def delete_model(self, model_path: str) -> Dict:
        """Delete a local model file"""
        # Resolve path
        if Path(model_path).is_absolute():
            full_path = Path(model_path)
        else:
            full_path = self.models_dir / model_path

        # Security check - must be inside models_dir
        try:
            full_path.resolve().relative_to(self.models_dir.resolve())
        except ValueError:
            raise ValueError("Invalid path - must be inside models directory")

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {model_path}")

        # Check if model is currently loaded
        for slot, info in self.slot_info.items():
            if info and str(full_path) in info.get("model_path", ""):
                raise RuntimeError(f"Model is currently loaded in slot {slot}. Unload it first.")

        full_path.unlink()
        return {"status": "deleted", "path": model_path}

    # === HuggingFace Integration ===

    async def search_hf_models(self, query: str) -> List[Dict]:
        """Search HuggingFace for GGUF and model files"""
        if HfApi is None:
            return {"error": "huggingface_hub not installed"}

        api = HfApi()

        try:
            models = api.list_models(
                search=query,
                sort="downloads",
                direction=-1,
                limit=20
            )

            results = []
            for model in models:
                try:
                    files = list_repo_files(model.id)

                    # Find model files: .gguf, .bin (whisper), mmproj files
                    model_files = []
                    for f in files:
                        if f.endswith(".gguf"):
                            model_files.append(f)
                        elif f.endswith(".bin") and ("whisper" in model.id.lower() or "ggml" in f.lower()):
                            model_files.append(f)

                    if model_files:
                        results.append({
                            "repo_id": model.id,
                            "downloads": model.downloads,
                            "likes": model.likes,
                            "files": model_files[:15]  # Show more files for mmproj
                        })
                except:
                    pass

                if len(results) >= 10:
                    break

            return results

        except Exception as e:
            return {"error": str(e)}

    async def download_hf_model(self, repo_id: str, filename: str) -> Dict:
        """Download model from HuggingFace"""
        if HfApi is None:
            return {"error": "huggingface_hub not installed"}

        hf_token = self.config.get("hf_token")

        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=self.models_dir,
                token=hf_token
            )

            return {
                "status": "downloaded",
                "path": str(Path(local_path).relative_to(self.models_dir)),
                "repo": repo_id,
                "filename": filename
            }

        except Exception as e:
            return {"error": str(e)}
