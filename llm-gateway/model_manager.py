"""
Model Manager - Dynamic model loading/unloading with HF integration
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
    """Manages llama-server and whisper-server processes"""
    
    SLOT_PORTS = list(range(4801, 4808))  # 4801-4807
    
    def __init__(self, base_dir: Path, models_dir: Path, config: Dict):
        self.base_dir = base_dir
        self.models_dir = models_dir
        self.config = config
        self.processes: Dict[int, subprocess.Popen] = {}
        self.slot_info: Dict[int, Dict] = {}
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Binaries
        self.llama_server = base_dir / "llama-server"
        self.whisper_server = base_dir / "whisper-server"
    
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
        threads: Optional[int] = None
    ) -> Dict:
        """Load a model into a slot"""
        
        if slot not in self.SLOT_PORTS:
            raise ValueError(f"Invalid slot {slot}. Must be 4801-4807")
        
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
        
        if model_type == "audio":
            # Whisper server
            if not self.whisper_server.exists():
                raise FileNotFoundError("whisper-server not found. Run setup.sh")
            
            cmd = [
                str(self.whisper_server),
                "--model", str(full_path),
                "--port", str(slot),
                "--threads", str(thr),
                "--host", "127.0.0.1"
            ]
        else:
            # llama-server for text and vision
            if not self.llama_server.exists():
                raise FileNotFoundError("llama-server not found. Run setup.sh")
            
            cmd = [
                str(self.llama_server),
                "--model", str(full_path),
                "--port", str(slot),
                "--ctx-size", str(ctx),
                "--threads", str(thr),
                "--host", "127.0.0.1",
                "--parallel", "2",  # Allow 2 concurrent requests
                "--cont-batching",
                "--jinja"  # Enable tool calling
            ]
            
            # Vision model needs mmproj
            if model_type == "vision":
                mmproj = full_path.with_suffix(".mmproj.gguf")
                if mmproj.exists():
                    cmd.extend(["--mmproj", str(mmproj)])
        
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        # Wait for server to be ready
        ready = await self._wait_for_ready(slot, timeout=60)
        
        if not ready:
            process.kill()
            raise RuntimeError(f"Model failed to start on port {slot}")
        
        self.processes[slot] = process
        self.slot_info[slot] = {
            "port": slot,
            "model_path": str(full_path),
            "model_name": full_path.stem,
            "model_type": model_type,
            "ctx_size": ctx,
            "threads": thr,
            "pid": process.pid
        }
        
        # Update config
        self._save_slot_config()
        
        return {
            "status": "loaded",
            "slot": slot,
            "model": full_path.stem,
            "pid": process.pid
        }
    
    async def _wait_for_ready(self, port: int, timeout: int = 60) -> bool:
        """Wait for server to respond to health check"""
        import httpx
        
        url = f"http://127.0.0.1:{port}/health"
        start = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code == 200:
                        return True
            except:
                pass
            await asyncio.sleep(0.5)
        
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
                process.wait()
        except ProcessLookupError:
            pass
        
        del self.processes[slot]
        if slot in self.slot_info:
            del self.slot_info[slot]
        
        self._save_slot_config()
        
        return {"status": "unloaded", "slot": slot}
    
    async def shutdown_all(self):
        """Shutdown all model processes"""
        for slot in list(self.processes.keys()):
            await self.unload_model(slot)
    
    def _save_slot_config(self):
        """Save current slot configuration"""
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
        """Get list of active models"""
        return [
            {
                "name": info["model_name"],
                "type": info["model_type"],
                "port": info["port"]
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
    
    def find_model_slot(self, model_name: str) -> Optional[Dict]:
        """Find which slot has a model loaded"""
        for info in self.slot_info.values():
            if model_name in info["model_name"] or info["model_name"] in model_name:
                return info
        return None
    
    def find_audio_slot(self) -> Optional[Dict]:
        """Find slot with audio model"""
        for info in self.slot_info.values():
            if info["model_type"] == "audio":
                return info
        return None
    
    def list_local_models(self) -> List[Dict]:
        """List downloaded models"""
        models = []
        
        for f in self.models_dir.glob("**/*.gguf"):
            if ".mmproj" in f.name:
                continue
            
            size_mb = f.stat().st_size / (1024**2)
            models.append({
                "name": f.stem,
                "path": str(f.relative_to(self.models_dir)),
                "size_mb": round(size_mb, 1),
                "size_gb": round(size_mb / 1024, 2)
            })
        
        # Also check for whisper models
        for f in self.models_dir.glob("**/*.bin"):
            size_mb = f.stat().st_size / (1024**2)
            models.append({
                "name": f.stem,
                "path": str(f.relative_to(self.models_dir)),
                "size_mb": round(size_mb, 1),
                "size_gb": round(size_mb / 1024, 2),
                "type": "audio"
            })
        
        return sorted(models, key=lambda x: x["name"])
    
    async def search_hf_models(self, query: str) -> List[Dict]:
        """Search HuggingFace for GGUF models"""
        if HfApi is None:
            return {"error": "huggingface_hub not installed"}
        
        api = HfApi()
        
        try:
            # Search for models with GGUF files
            models = api.list_models(
                search=query,
                sort="downloads",
                direction=-1,
                limit=20
            )
            
            results = []
            for model in models:
                # Check if has GGUF files
                try:
                    files = list_repo_files(model.id)
                    gguf_files = [f for f in files if f.endswith(".gguf")]
                    
                    if gguf_files:
                        results.append({
                            "repo_id": model.id,
                            "downloads": model.downloads,
                            "likes": model.likes,
                            "files": gguf_files[:10]  # Limit files shown
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
            # Download to models directory
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
