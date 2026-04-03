# Implementierungsplan: Instagram Video RAG Mod

## Dateistruktur

```
toolboxv2/mods/VideoRAG/
├── __init__.py              # Mod-Export + Registration
├── services/
│   ├── __init__.py
│   ├── embedding.py         # Qwen3-VL-Embedding Wrapper
│   ├── ocr.py               # DeepSeek-OCR Wrapper
│   ├── whisper_svc.py       # Whisper Transkription
│   └── video_utils.py       # FFmpeg Helpers
├── pipeline.py              # Ingestion Pipeline
├── search.py                # Search Engine
├── views.py                 # MinUI Views (Frontend)
└── tests/
    ├── __init__.py
    ├── test_pipeline.py
    ├── test_search.py
    └── test_services.py
```
# user muss bulk lik uplad mach konne manuelles videso und bilder uploding so wie nach text und bilder und videos suchen mit text bilder und videos
---

## Datei 1: `__init__.py` — Mod Registration

```python
"""
VideoRAG - Instagram Video Search with Multimodal RAG
=====================================================
Combines Qwen3-VL-Embedding + DeepSeek-OCR + Whisper + HybridMemoryStore
"""

import json
import time
from pathlib import Path
from typing import Any, Dict

from toolboxv2 import App, Result, RequestData, get_app, MainTool

MOD_NAME = "VideoRAG"
VERSION = "0.1.0"

export = get_app(f"mods.{MOD_NAME}").tb


class Tools(MainTool):
    def __init__(self, app: App):
        self.name = MOD_NAME
        self.version = VERSION
        self.pipeline = None
        self.search_engine = None
        self.memory = None

        self.tools = {
            "all": [
                ["stats", "Show memory statistics"],
                ["reindex", "Rebuild FAISS index"],
            ],
            "name": self.name,
            "stats": self.show_stats,
            "reindex": self.reindex,
        }

        super().__init__(
            load=self.on_start,
            v=self.version,
            tool=self.tools,
            name=self.name,
            on_exit=self.on_exit,
        )

    def on_start(self):
        self.app.logger.info(f"{self.name} v{self.version} initializing...")

        from toolboxv2.mods.isaa.base.hybrid_memory import HybridMemoryStore
        from .pipeline import VideoIngestionPipeline
        from .search import VideoSearchEngine
        from .services.embedding import EmbeddingService
        from .services.ocr import OCRService
        from .services.whisper_svc import WhisperService

        db_dir = self.app.data_dir / "video_rag"
        db_dir.mkdir(parents=True, exist_ok=True)

        # HybridMemoryStore — embedding_dim muss zu Qwen3-VL-Embedding passen
        self.memory = HybridMemoryStore(
            db_dir=str(db_dir),
            embedding_dim=768,        # Qwen3-VL-Embedding-2B default
            space="instagram_videos",
        )

        # Services (lazy-loaded, erst bei erstem Aufruf GPU-belastend)
        self.embed_svc = EmbeddingService(app=self.app)
        self.ocr_svc = OCRService()
        self.whisper_svc = WhisperService()

        # Pipeline + Search
        self.pipeline = VideoIngestionPipeline(
            memory=self.memory,
            embed_svc=self.embed_svc,
            ocr_svc=self.ocr_svc,
            whisper_svc=self.whisper_svc,
            app=self.app,
        )
        self.search_engine = VideoSearchEngine(
            memory=self.memory,
            embed_svc=self.embed_svc,
        )

        # CloudM UI registrieren
        try:
            self.app.run_any(
                ("CloudM", "add_ui"),
                name=self.name,
                title="Video RAG Search",
                path=f"/api/{self.name}/ui",
                description="Instagram Video Search with Multimodal RAG",
                auth=False,
            )
        except Exception as e:
            self.app.logger.warning(f"Could not register UI: {e}")

        self.app.logger.info(f"{self.name} ready. Memory stats: {self.memory.stats()}")

    def on_exit(self):
        if self.memory:
            self.memory.save_faiss_to_dir()
            self.memory.close()
        self.app.logger.info(f"{self.name} shut down.")

    def show_stats(self):
        return self.memory.stats() if self.memory else "Not initialized"

    def reindex(self):
        if self.memory:
            count = self.memory.rebuild_faiss_index()
            return f"Rebuilt index with {count} entries"
        return "Not initialized"


# ══════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════


@export(mod_name=MOD_NAME, name="ui", api=True, api_methods=["GET"])
async def get_ui(app: App) -> Result:
    """Serves the MinUI frontend"""
    # MinUI Views werden in views.py registriert
    return Result.ok(info="Use MinUI views: search, upload, library")


@export(mod_name=MOD_NAME, name="search", api=True, api_methods=["GET"],
        request_as_kwarg=True)
async def api_search(app: App, request: RequestData) -> Result:
    """
    GET /api/VideoRAG/search?q=rust+http+server&k=5&category=tech
    """
    tools = app.get_mod(MOD_NAME)
    if not tools or not tools.search_engine:
        return Result.default_internal_error(info="VideoRAG not initialized")

    q = request.query_params.get("q", "")
    k = int(request.query_params.get("k", "5"))
    category = request.query_params.get("category")

    if not q:
        return Result.default_user_error(info="Query parameter 'q' required")

    results = await tools.search_engine.search(q, k=k, category=category)
    return Result.json(data={"query": q, "results": results, "count": len(results)})


@export(mod_name=MOD_NAME, name="ingest", api=True, api_methods=["POST"],
        request_as_kwarg=True)
async def api_ingest(app: App, request: RequestData) -> Result:
    """
    POST /api/VideoRAG/ingest
    Body: multipart/form-data with 'video' file + optional JSON meta
    """
    tools = app.get_mod(MOD_NAME)
    if not tools or not tools.pipeline:
        return Result.default_internal_error(info="VideoRAG not initialized")

    body = request.json() if hasattr(request, "json") else {}
    video_path = body.get("video_path")
    meta = body.get("meta", {})

    if not video_path:
        return Result.default_user_error(info="'video_path' required")

    try:
        entry_id = await tools.pipeline.ingest(video_path, meta)
        return Result.json(data={"entry_id": entry_id, "status": "ingested"})
    except Exception as e:
        return Result.default_internal_error(info=f"Ingestion failed: {e}")


@export(mod_name=MOD_NAME, name="batch_ingest", api=True, api_methods=["POST"],
        request_as_kwarg=True)
async def api_batch_ingest(app: App, request: RequestData) -> Result:
    """
    POST /api/VideoRAG/batch_ingest
    Body: {"directory": "/path/to/videos", "category": "tech"}
    """
    tools = app.get_mod(MOD_NAME)
    body = request.json() if hasattr(request, "json") else {}
    directory = body.get("directory")
    category = body.get("category")

    if not directory:
        return Result.default_user_error(info="'directory' required")

    results = await tools.pipeline.batch_ingest(directory, default_category=category)
    return Result.json(data=results)


@export(mod_name=MOD_NAME, name="stats", api=True, api_methods=["GET"])
async def api_stats(app: App) -> Result:
    """GET /api/VideoRAG/stats"""
    tools = app.get_mod(MOD_NAME)
    if not tools or not tools.memory:
        return Result.default_internal_error(info="Not initialized")

    stats = tools.memory.stats()
    return Result.json(data=stats)


@export(mod_name=MOD_NAME, name="categories", api=True, api_methods=["GET"])
async def api_categories(app: App) -> Result:
    """GET /api/VideoRAG/categories — Alle verwendeten Kategorien"""
    tools = app.get_mod(MOD_NAME)
    if not tools or not tools.memory:
        return Result.default_internal_error(info="Not initialized")

    rows = tools.memory._exec(
        "SELECT DISTINCT meta_category FROM entries "
        "WHERE space = ? AND is_active = 1 AND meta_category IS NOT NULL",
        (tools.memory.space,),
    ).fetchall()
    categories = [r[0] for r in rows]
    return Result.json(data={"categories": categories})


# ══════════════════════════════════════════
# MINU VIEW REGISTRATION
# ══════════════════════════════════════════


@export(mod_name=MOD_NAME, name="initialize", initial=True)
def initialize(app: App, **kwargs) -> Result:
    """Initialize module and register MinUI views"""
    from toolboxv2.mods.Minu import register_view
    from .views import SearchView, UploadView, LibraryView

    register_view("video_search", SearchView)
    register_view("video_upload", UploadView)
    register_view("video_library", LibraryView)

    return Result.ok(info=f"{MOD_NAME} views registered")
```

---

## Datei 2: `services/embedding.py` — Qwen3-VL-Embedding

```python
"""
Qwen3-VL-Embedding Service
===========================
Multimodal embeddings für Text, Bilder und Videos.
Lazy-Loading: GPU wird erst beim ersten encode() belastet.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union


class EmbeddingService:
    """Wrapper um Qwen3-VL-Embedding mit lazy init"""

    def __init__(self, app=None, model_size: str = "2B"):
        self._model = None
        self._app = app
        self._model_size = model_size
        # Model path — kann via app.config überschrieben werden
        self._model_path = f"./models/Qwen3-VL-Embedding-{model_size}"
        self.dim = 768 if model_size == "2B" else 4096

    def _ensure_loaded(self):
        """Lazy-load: Model erst bei Bedarf laden"""
        if self._model is not None:
            return

        try:
            import torch
            from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

            if self._app:
                self._app.logger.info(
                    f"Loading Qwen3-VL-Embedding-{self._model_size}..."
                )

            self._model = Qwen3VLEmbedder(
                model_name_or_path=self._model_path,
                max_length=8192,
                max_frames=32,           # 32 Frames pro Video reichen
                fps=0.5,                 # 1 Frame alle 2 Sekunden
                torch_dtype=torch.bfloat16,
            )

            if self._app:
                self._app.logger.info("Qwen3-VL-Embedding loaded.")
        except ImportError:
            raise RuntimeError(
                "Qwen3-VL-Embedding nicht installiert. "
                "Run: git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git"
            )

    def encode_text(self, text: str, task: str = "retrieval.query") -> np.ndarray:
        """Text → Embedding Vector"""
        self._ensure_loaded()
        instruction = (
            "Find Instagram videos matching this description"
            if task == "retrieval.query"
            else "Represent this video content for retrieval"
        )
        emb = self._model.encode(
            [{"text": text}],
            task=task,
            instruction=instruction,
        )
        return np.array(emb[0], dtype=np.float32)

    def encode_video(self, video_path: str) -> np.ndarray:
        """Video → Embedding Vector (multimodal, versteht zeitlichen Kontext)"""
        self._ensure_loaded()
        emb = self._model.encode(
            [{"video": video_path}],
            task="retrieval.passage",
            instruction="Represent this Instagram video for retrieval",
        )
        return np.array(emb[0], dtype=np.float32)

    def encode_image(self, image_path: str) -> np.ndarray:
        """Bild → Embedding (für Screenshot-basierte Suche)"""
        self._ensure_loaded()
        emb = self._model.encode(
            [{"image": image_path}],
            task="retrieval.query",
        )
        return np.array(emb[0], dtype=np.float32)

    def encode_multimodal(
        self, text: str = None, image: str = None, video: str = None
    ) -> np.ndarray:
        """Gemischter Input → Unified Embedding"""
        self._ensure_loaded()
        doc = {}
        if text:
            doc["text"] = text
        if image:
            doc["image"] = image
        if video:
            doc["video"] = video

        emb = self._model.encode([doc], task="retrieval.passage")
        return np.array(emb[0], dtype=np.float32)
```

---

## Datei 3: `services/ocr.py` — DeepSeek-OCR

```python
"""
DeepSeek-OCR Service
====================
On-Screen Text Extraction aus Video-Frames.
Nutzt Ollama als Backend (einfachste Integration).

Fallback: Transformers-basiert wenn Ollama nicht verfügbar.
"""

import subprocess
from pathlib import Path
from typing import List, Optional


class OCRService:
    """DeepSeek-OCR via Ollama oder Transformers"""

    def __init__(self, backend: str = "ollama", model: str = "deepseek-ocr"):
        self.backend = backend
        self.model = model
        self._transformer_model = None

    def extract_text(self, image_path: str) -> str:
        """Extrahiere Text aus einem einzelnen Frame"""
        if self.backend == "ollama":
            return self._ollama_ocr(image_path)
        return self._transformers_ocr(image_path)

    def batch_extract(self, image_paths: List[str]) -> List[str]:
        """OCR über mehrere Frames — filtert leere Ergebnisse"""
        results = []
        seen = set()
        for path in image_paths:
            text = self.extract_text(path).strip()
            # Deduplizierung: Gleicher Text in verschiedenen Frames → nur einmal
            if text and text not in seen:
                results.append(text)
                seen.add(text)
        return results

    def _ollama_ocr(self, image_path: str) -> str:
        """OCR via Ollama API"""
        try:
            import ollama

            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": "<image>\nFree OCR.",
                        "images": [image_path],
                    }
                ],
            )
            return response.message.content
        except Exception as e:
            # Fallback: Ollama CLI
            return self._ollama_cli(image_path)

    def _ollama_cli(self, image_path: str) -> str:
        """Fallback: Ollama via CLI"""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, f"<image>\nFree OCR. {image_path}"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _transformers_ocr(self, image_path: str) -> str:
        """Fallback: Transformers-basiert (braucht mehr VRAM)"""
        if self._transformer_model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_name = "deepseek-ai/DeepSeek-OCR"
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._transformer_model = AutoModel.from_pretrained(
                model_name,
                _attn_implementation="flash_attention_2",
                trust_remote_code=True,
                use_safetensors=True,
            )
            self._transformer_model = (
                self._transformer_model.eval().cuda().to(torch.bfloat16)
            )

        result = self._transformer_model.infer(
            self._tokenizer,
            prompt="<image>\nFree OCR.",
            image_file=image_path,
        )
        return result if isinstance(result, str) else str(result)
```

---

## Datei 4: `services/whisper_svc.py` — Transkription

```python
"""
Whisper Transcription Service
"""

from pathlib import Path
from typing import Dict, Any, Optional


class WhisperService:
    """Whisper für Audio-Transkription mit Timestamps"""

    def __init__(self, model_size: str = "small", device: str = "auto"):
        self._model = None
        self._model_size = model_size
        self._device = device

    def _ensure_loaded(self):
        if self._model is not None:
            return

        import whisper

        device = self._device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = whisper.load_model(self._model_size, device=device)

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transkribiere Audio-Datei.

        Returns:
            {
                "text": "Vollständiges Transkript",
                "language": "de",
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": "Hallo Leute"},
                    {"start": 2.5, "end": 5.0, "text": "heute zeige ich euch"},
                ]
            }
        """
        self._ensure_loaded()

        result = self._model.transcribe(
            audio_path,
            language=None,     # Auto-detect
            task="transcribe",
            verbose=False,
        )

        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
        }
```

---

## Datei 5: `services/video_utils.py` — FFmpeg Helpers

```python
"""
Video Processing Utilities (FFmpeg)
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List


def extract_audio(video_path: str, output_path: str = None) -> str:
    """Extrahiere Audio als WAV"""
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vn",                  # Kein Video
            "-acodec", "pcm_s16le",
            "-ar", "16000",         # 16kHz für Whisper
            "-ac", "1",             # Mono
            "-y",                   # Überschreiben
            output_path,
        ],
        capture_output=True,
        check=True,
    )
    return output_path


def extract_keyframes(
    video_path: str,
    fps: float = 0.5,
    output_dir: str = None,
    max_frames: int = 10,
) -> List[str]:
    """
    Extrahiere Key-Frames aus Video.

    Args:
        fps: Frames pro Sekunde (0.5 = alle 2 Sekunden)
        max_frames: Maximum Frames

    Returns:
        Liste der Frame-Pfade
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="vrag_frames_")

    output_pattern = f"{output_dir}/frame_%04d.jpg"

    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps}",
            "-frames:v", str(max_frames),
            "-q:v", "2",            # Qualität (2 = hoch)
            "-y",
            output_pattern,
        ],
        capture_output=True,
        check=True,
    )

    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    return [str(f) for f in frames]


def get_video_duration(video_path: str) -> float:
    """Video-Dauer in Sekunden"""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def generate_thumbnail(video_path: str, output_path: str = None, time: float = 1.0) -> str:
    """Thumbnail aus Video generieren"""
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".jpg")

    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-ss", str(time),
            "-vframes", "1",
            "-q:v", "2",
            "-y",
            output_path,
        ],
        capture_output=True,
        check=True,
    )
    return output_path
```

---

## Datei 6: `pipeline.py` — Ingestion Pipeline

```python
"""
Video Ingestion Pipeline
========================
Video → Audio + Frames → Whisper + DeepSeek-OCR + Qwen3-VL-Embed → HybridMemoryStore
"""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from toolboxv2.mods.isaa.base.hybrid_memory import HybridMemoryStore

from .services.embedding import EmbeddingService
from .services.ocr import OCRService
from .services.whisper_svc import WhisperService
from .services.video_utils import (
    extract_audio,
    extract_keyframes,
    generate_thumbnail,
    get_video_duration,
)


class VideoIngestionPipeline:

    def __init__(
        self,
        memory: HybridMemoryStore,
        embed_svc: EmbeddingService,
        ocr_svc: OCRService,
        whisper_svc: WhisperService,
        app=None,
    ):
        self.memory = memory
        self.embed_svc = embed_svc
        self.ocr_svc = ocr_svc
        self.whisper_svc = whisper_svc
        self.app = app

    def _log(self, msg: str):
        if self.app:
            self.app.logger.info(f"[VideoRAG] {msg}")

    async def ingest(self, video_path: str, meta: Dict[str, Any] = None) -> str:
        """
        Vollständige Ingestion eines Videos.

        Args:
            video_path: Pfad zur Video-Datei
            meta: {
                "creator": "username",
                "caption": "Video caption text",
                "category": "tech",
                "hashtags": ["rust", "coding"],
                "instagram_url": "https://...",
            }

        Returns:
            entry_id im HybridMemoryStore
        """
        meta = meta or {}
        tmp_dir = tempfile.mkdtemp(prefix="vrag_")

        try:
            self._log(f"Ingesting: {video_path}")
            duration = get_video_duration(video_path)

            # ── 1. Audio extrahieren + transkribieren ──
            self._log("  [1/4] Audio → Whisper")
            audio_path = extract_audio(video_path, f"{tmp_dir}/audio.wav")
            transcript = self.whisper_svc.transcribe(audio_path)

            # ── 2. Key-Frames extrahieren + OCR ──
            self._log("  [2/4] Frames → DeepSeek-OCR")
            frames = extract_keyframes(
                video_path,
                fps=0.5,
                output_dir=f"{tmp_dir}/frames",
                max_frames=6,
            )
            ocr_texts = self.ocr_svc.batch_extract(frames)

            # ── 3. Multimodales Video-Embedding ──
            self._log("  [3/4] Video → Qwen3-VL-Embedding")
            video_embedding = self.embed_svc.encode_video(video_path)

            # ── 4. Thumbnail generieren ──
            thumbnail_path = generate_thumbnail(video_path, f"{tmp_dir}/thumb.jpg")

            # ── 5. Document bauen + in Memory speichern ──
            self._log("  [4/4] Storing in HybridMemoryStore")
            document = self._build_document(transcript, ocr_texts, meta)
            concepts = self._extract_concepts(document, meta)

            # MinIO Upload (Video + Thumbnail)
            minio_video_key = None
            minio_thumb_key = None
            try:
                minio_video_key = self._upload_to_minio(video_path, "videos")
                minio_thumb_key = self._upload_to_minio(thumbnail_path, "thumbnails")
            except Exception as e:
                self._log(f"  MinIO upload skipped: {e}")

            # HybridMemoryStore.add()
            entry_id = self.memory.add(
                content=document,
                embedding=video_embedding,
                content_type="video",
                meta={
                    "source": minio_video_key or video_path,
                    "category": meta.get("category"),
                    "language": transcript.get("language"),
                    "importance": 0.5,
                    # Alles Weitere in meta_custom (JSON)
                    "creator": meta.get("creator"),
                    "caption": meta.get("caption"),
                    "duration": duration,
                    "transcript_text": transcript["text"],
                    "transcript_segments": json.dumps(transcript["segments"]),
                    "ocr_texts": json.dumps(ocr_texts),
                    "thumbnail": minio_thumb_key,
                    "instagram_url": meta.get("instagram_url"),
                    "hashtags": json.dumps(meta.get("hashtags", [])),
                },
                concepts=concepts,
            )

            # ── 6. Entity-Relations ──
            creator = meta.get("creator")
            if creator:
                try:
                    self.memory.add_entity(
                        f"creator:{creator}", "person", creator
                    )
                    self.memory.add_relation(
                        entry_id, f"creator:{creator}", "CREATED_BY"
                    )
                except Exception:
                    pass  # Entity exists

            category = meta.get("category")
            if category:
                try:
                    self.memory.add_entity(
                        f"category:{category}", "category", category
                    )
                    self.memory.add_relation(
                        entry_id, f"category:{category}", "PART_OF"
                    )
                except Exception:
                    pass

            self._log(f"  Done → entry_id={entry_id}")
            return entry_id

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def batch_ingest(
        self, directory: str, default_category: str = None
    ) -> Dict[str, Any]:
        """Alle Videos in einem Verzeichnis verarbeiten"""
        video_dir = Path(directory)
        extensions = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
        videos = [f for f in video_dir.iterdir() if f.suffix.lower() in extensions]

        results = {"total": len(videos), "success": 0, "failed": 0, "entries": []}

        for video in videos:
            try:
                meta = {"category": default_category} if default_category else {}
                entry_id = await self.ingest(str(video), meta)
                results["success"] += 1
                results["entries"].append({"file": video.name, "entry_id": entry_id})
            except Exception as e:
                results["failed"] += 1
                results["entries"].append({"file": video.name, "error": str(e)})
                self._log(f"  FAILED: {video.name} → {e}")

        return results

    def _build_document(
        self, transcript: Dict, ocr_texts: List[str], meta: Dict
    ) -> str:
        """Durchsuchbarer Text aus allen Quellen"""
        parts = []

        if meta.get("caption"):
            parts.append(f"Caption: {meta['caption']}")
        if meta.get("creator"):
            parts.append(f"Creator: {meta['creator']}")
        if transcript.get("text"):
            parts.append(f"Transcript: {transcript['text']}")
        if ocr_texts:
            parts.append(f"On-Screen Text: {' | '.join(ocr_texts)}")
        if meta.get("hashtags"):
            parts.append(f"Tags: {', '.join(meta['hashtags'])}")

        return "\n".join(parts)

    def _extract_concepts(self, document: str, meta: Dict) -> List[str]:
        """Auto-extract Konzepte für den Concept-Index"""
        concepts = set()

        # Hashtags direkt als Concepts
        for tag in meta.get("hashtags", []):
            concepts.add(tag.lower().strip("#"))

        # Kategorie
        if meta.get("category"):
            concepts.add(meta["category"].lower())

        # Creator
        if meta.get("creator"):
            concepts.add(meta["creator"].lower())

        # Wörter > 4 Zeichen aus dem Dokument (einfaches Keyword-Extraction)
        stop_words = {
            "caption", "creator", "transcript", "screen", "text", "tags",
            "this", "that", "with", "from", "have", "been", "also", "which",
            "their", "about", "would", "these", "other", "than", "them",
            "wird", "eine", "dass", "auch", "nicht", "sich", "dann",
        }
        for word in document.lower().split():
            clean = word.strip(".,;:!?()[]\"'")
            if len(clean) > 4 and clean not in stop_words and clean.isalpha():
                concepts.add(clean)

        return list(concepts)[:50]  # Max 50 Concepts pro Entry

    def _upload_to_minio(self, file_path: str, prefix: str) -> str:
        """Upload zu MinIO (wenn konfiguriert)"""
        if not self.memory._minio:
            raise RuntimeError("MinIO not configured")

        key = f"{prefix}/{uuid4().hex[:16]}{Path(file_path).suffix}"
        self.memory._minio.fput_object(
            self.memory._minio_bucket, key, file_path
        )
        return key
```

---

## Datei 7: `search.py` — Search Engine

```python
"""
Video Search Engine
===================
Hybrid Search über HybridMemoryStore mit multimodalen Queries.
"""

import json
from typing import Any, Dict, List, Optional

from toolboxv2.mods.isaa.base.hybrid_memory import HybridMemoryStore

from .services.embedding import EmbeddingService


class VideoSearchEngine:

    def __init__(self, memory: HybridMemoryStore, embed_svc: EmbeddingService):
        self.memory = memory
        self.embed_svc = embed_svc

    async def search(
        self,
        query: str,
        k: int = 5,
        category: str = None,
        creator: str = None,
        search_modes: tuple = ("vector", "bm25", "relation"),
    ) -> List[Dict[str, Any]]:
        """
        Semantische Suche über alle Videos.

        Args:
            query: Freitext-Suche ("Wie baut man einen HTTP Server in Rust?")
            k: Anzahl Ergebnisse
            category: Optional Kategorie-Filter
            creator: Optional Creator-Filter

        Returns:
            Liste von Video-Ergebnissen mit Metadaten
        """
        # 1. Query → Embedding
        query_embedding = self.embed_svc.encode_text(query, task="retrieval.query")

        # 2. Meta-Filter bauen
        meta_filter = {}
        if category:
            meta_filter["category"] = category

        # 3. HybridMemoryStore.query() — Vector + BM25 + Relations + RRF
        results = self.memory.query(
            query_text=query,
            query_embedding=query_embedding,
            k=k,
            search_modes=search_modes,
            meta_filter=meta_filter if meta_filter else None,
            mode_weights={"vector": 0.50, "bm25": 0.30, "relation": 0.20},
        )

        # 4. Ergebnisse anreichern
        enriched = []
        for r in results:
            meta = r.get("meta", {})

            # meta_custom JSON parsen
            custom = {}
            if isinstance(meta.get("custom"), str):
                try:
                    custom = json.loads(meta["custom"])
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(meta, dict):
                custom = meta

            # Creator-Filter (post-filter, da nicht in meta_filter Spalten)
            if creator and custom.get("creator", "").lower() != creator.lower():
                continue

            # Transcript Segments parsen
            segments = []
            raw_segments = custom.get("transcript_segments", "[]")
            if isinstance(raw_segments, str):
                try:
                    segments = json.loads(raw_segments)
                except (json.JSONDecodeError, TypeError):
                    pass
            else:
                segments = raw_segments or []

            enriched.append({
                "id": r["id"],
                "score": round(r["score"], 4),
                "content_preview": r.get("content", "")[:300],
                "creator": custom.get("creator"),
                "category": meta.get("category"),
                "duration": custom.get("duration"),
                "transcript": custom.get("transcript_text", "")[:500],
                "ocr_texts": json.loads(custom.get("ocr_texts", "[]"))
                    if isinstance(custom.get("ocr_texts"), str)
                    else custom.get("ocr_texts", []),
                "video_source": meta.get("source"),
                "thumbnail": custom.get("thumbnail"),
                "instagram_url": custom.get("instagram_url"),
                "segments": segments[:10],  # Erste 10 Segmente
                "concepts": r.get("concepts", []),
            })

        return enriched

    async def search_by_image(self, image_path: str, k: int = 5) -> List[Dict]:
        """Suche via Bild (Screenshot → ähnliche Videos)"""
        query_embedding = self.embed_svc.encode_image(image_path)

        results = self.memory.query(
            query_text="",
            query_embedding=query_embedding,
            k=k,
            search_modes=("vector",),  # Nur Vector-Search für Bild-Queries
        )

        return [self._format_result(r) for r in results]

    def _format_result(self, r: Dict) -> Dict:
        """Minimal-Format für schnelle Ergebnisse"""
        meta = r.get("meta", {})
        return {
            "id": r["id"],
            "score": round(r["score"], 4),
            "category": meta.get("category"),
            "source": meta.get("source"),
        }
```

---

## Datei 8: `views.py` — MinUI Frontend

```python
"""
MinUI Views für Video RAG
==========================
Reactive Frontend-Views nach dem Minu-Pattern.
"""

from toolboxv2.mods.Minu.core import (
    Alert, Badge, Button, Card, Column, Divider, Grid,
    Heading, Icon, Image, Input, ListItem, MinuView, Modal,
    Progress, Row, Select, Spacer, Spinner, State, Switch,
    Table, Tabs, Text, List as MinuList, Dynamic,
)


# ══════════════════════════════════════════
# SEARCH VIEW
# ══════════════════════════════════════════


class SearchView(MinuView):
    """Hauptsuche — semantische Videosuche"""

    query = State("")
    results = State([])
    selected_category = State("")
    is_searching = State(False)
    result_count = State(0)

    def render(self):
        return Column(
            # Header
            Card(
                Heading("🎬 Video RAG Search", level=2),
                Text("Semantische Suche über deine Instagram-Videos"),
                Spacer(),
                # Search Bar
                Row(
                    Input(
                        placeholder="z.B. 'Wie baut man einen HTTP Server in Rust?'",
                        value=self.query.value,
                        bind="query",
                        on_submit="do_search",
                    ),
                    Button("🔍 Suchen", on_click="do_search", variant="primary"),
                    gap="2",
                ),
                # Category Filter
                Row(
                    Select(
                        options=[
                            {"value": "", "label": "Alle Kategorien"},
                            {"value": "tech", "label": "Tech"},
                            {"value": "tutorial", "label": "Tutorial"},
                            {"value": "lifestyle", "label": "Lifestyle"},
                            {"value": "science", "label": "Science"},
                        ],
                        value=self.selected_category.value,
                        bind="selected_category",
                        label="Kategorie",
                    ),
                    Text(
                        f"{self.result_count.value} Ergebnisse",
                        className="text-sm text-secondary",
                    ),
                    justify="between",
                ),
                className="card",
            ),

            # Loading
            Card(
                Spinner(),
                Text("Suche läuft..."),
                className="card text-center",
            ) if self.is_searching.value else None,

            # Results
            Column(
                *[
                    self._render_result(r, i)
                    for i, r in enumerate(self.results.value)
                ],
                gap="2",
            ) if self.results.value else None,

            # Empty State
            Card(
                Text("Keine Ergebnisse. Versuche eine andere Suchanfrage.",
                     className="text-secondary text-center"),
                className="card",
            ) if not self.results.value and not self.is_searching.value and self.query.value else None,
        )

    def _render_result(self, result: dict, index: int) -> "Component":
        """Einzelnes Suchergebnis als Card"""
        score_pct = int(result.get("score", 0) * 100)
        creator = result.get("creator", "Unbekannt")
        category = result.get("category", "")
        transcript = result.get("transcript", "")[:200]
        concepts = result.get("concepts", [])[:5]

        return Card(
            Row(
                # Thumbnail Placeholder
                Card(
                    Text("🎬", className="text-4xl"),
                    className="p-4",
                    style="width: 120px; height: 80px; background: #f0f0f0; "
                          "border-radius: 8px; display: flex; align-items: center; "
                          "justify-content: center;",
                ),
                # Content
                Column(
                    Row(
                        Heading(f"#{index + 1}", level=4),
                        Badge(f"{score_pct}%", variant="primary"),
                        Badge(category, variant="secondary") if category else None,
                        Text(f"@{creator}", className="text-sm text-secondary"),
                        gap="2",
                    ),
                    Text(transcript + "..." if transcript else "Kein Transkript",
                         className="text-sm"),
                    Row(
                        *[Badge(c, variant="default") for c in concepts],
                        gap="1",
                    ) if concepts else None,
                    gap="1",
                    className="flex-1",
                ),
                gap="4",
            ),
            className="card hover:bg-gray-50",
        )

    async def do_search(self, event):
        query = self.query.value.strip()
        if not query:
            return

        self.is_searching.value = True
        self.results.value = []

        try:
            from toolboxv2 import get_app
            app = get_app()
            tools = app.get_mod("VideoRAG")

            if tools and tools.search_engine:
                results = await tools.search_engine.search(
                    query=query,
                    k=10,
                    category=self.selected_category.value or None,
                )
                self.results.value = results
                self.result_count.value = len(results)
        except Exception as e:
            self.results.value = []
            self.result_count.value = 0
        finally:
            self.is_searching.value = False


# ══════════════════════════════════════════
# UPLOAD VIEW
# ══════════════════════════════════════════


class UploadView(MinuView):
    """Video Upload + Ingestion"""

    video_path = State("")
    creator = State("")
    caption = State("")
    category = State("tech")
    hashtags = State("")
    is_processing = State(False)
    progress_text = State("")
    last_result = State("")

    def render(self):
        return Card(
            Heading("📤 Video Import", level=2),
            Text("Video zur RAG-Bibliothek hinzufügen"),
            Spacer(),

            # Form
            Input(
                placeholder="/pfad/zum/video.mp4 oder MinIO Key",
                value=self.video_path.value,
                bind="video_path",
                label="Video-Pfad",
            ),
            Row(
                Input(
                    placeholder="@creator",
                    value=self.creator.value,
                    bind="creator",
                    label="Creator",
                ),
                Select(
                    options=[
                        {"value": "tech", "label": "Tech"},
                        {"value": "tutorial", "label": "Tutorial"},
                        {"value": "lifestyle", "label": "Lifestyle"},
                        {"value": "science", "label": "Science"},
                        {"value": "other", "label": "Sonstige"},
                    ],
                    value=self.category.value,
                    bind="category",
                    label="Kategorie",
                ),
                gap="4",
            ),
            Input(
                placeholder="Caption / Beschreibung",
                value=self.caption.value,
                bind="caption",
                label="Caption",
            ),
            Input(
                placeholder="#rust #coding #tutorial (kommagetrennt)",
                value=self.hashtags.value,
                bind="hashtags",
                label="Hashtags",
            ),
            Spacer(),

            # Actions
            Row(
                Button(
                    "🚀 Video verarbeiten",
                    on_click="start_ingest",
                    variant="primary",
                    disabled=self.is_processing.value,
                ),
                Button(
                    "📁 Batch-Import",
                    on_click="batch_import",
                    variant="secondary",
                    disabled=self.is_processing.value,
                ),
                gap="2",
            ),

            # Processing Feedback
            Card(
                Spinner(),
                Text(self.progress_text.value),
                className="card",
            ) if self.is_processing.value else None,

            # Result
            Alert(
                self.last_result.value,
                variant="success",
                dismissible=True,
            ) if self.last_result.value else None,

            title="Video Import",
            className="card max-w-lg",
        )

    async def start_ingest(self, event):
        path = self.video_path.value.strip()
        if not path:
            return

        self.is_processing.value = True
        self.progress_text.value = "Verarbeite Video..."

        try:
            from toolboxv2 import get_app
            tools = get_app().get_mod("VideoRAG")

            meta = {
                "creator": self.creator.value.strip().strip("@"),
                "caption": self.caption.value.strip(),
                "category": self.category.value,
                "hashtags": [
                    h.strip().strip("#")
                    for h in self.hashtags.value.split(",")
                    if h.strip()
                ],
            }

            entry_id = await tools.pipeline.ingest(path, meta)
            self.last_result.value = f"Erfolgreich! Entry-ID: {entry_id}"

            # Reset Form
            self.video_path.value = ""
            self.caption.value = ""
            self.hashtags.value = ""

        except Exception as e:
            self.last_result.value = f"Fehler: {e}"
        finally:
            self.is_processing.value = False
            self.progress_text.value = ""

    async def batch_import(self, event):
        path = self.video_path.value.strip()
        if not path:
            return

        self.is_processing.value = True
        self.progress_text.value = "Batch-Import läuft..."

        try:
            from toolboxv2 import get_app
            tools = get_app().get_mod("VideoRAG")
            results = await tools.pipeline.batch_ingest(
                path, default_category=self.category.value
            )
            self.last_result.value = (
                f"Batch fertig: {results['success']}/{results['total']} erfolgreich"
            )
        except Exception as e:
            self.last_result.value = f"Fehler: {e}"
        finally:
            self.is_processing.value = False


# ══════════════════════════════════════════
# LIBRARY VIEW
# ══════════════════════════════════════════


class LibraryView(MinuView):
    """Bibliotheks-Übersicht aller Videos"""

    videos = State([])
    filter_category = State("")
    stats = State({})

    async def on_mount(self):
        """Daten beim Öffnen laden"""
        await self._load_data()

    async def _load_data(self):
        try:
            from toolboxv2 import get_app
            tools = get_app().get_mod("VideoRAG")
            if not tools or not tools.memory:
                return

            self.stats.value = tools.memory.stats()

            # Letzte 50 Videos laden
            rows = tools.memory._exec(
                "SELECT id, content, meta_category, meta_source, created_at "
                "FROM entries WHERE space = ? AND is_active = 1 AND content_type = 'video' "
                "ORDER BY created_at DESC LIMIT 50",
                (tools.memory.space,),
            ).fetchall()

            self.videos.value = [
                {
                    "id": r["id"],
                    "content": r["content"][:150],
                    "category": r["meta_category"] or "—",
                    "source": r["meta_source"] or "—",
                }
                for r in rows
            ]
        except Exception:
            pass

    def render(self):
        stats = self.stats.value
        videos = self.videos.value

        return Column(
            # Stats
            Card(
                Heading("📚 Video-Bibliothek", level=2),
                Row(
                    Card(
                        Heading(str(stats.get("active", 0)), level=3),
                        Text("Videos"),
                        className="text-center p-4",
                    ),
                    Card(
                        Heading(str(stats.get("entities", 0)), level=3),
                        Text("Creators"),
                        className="text-center p-4",
                    ),
                    Card(
                        Heading(str(stats.get("concepts", 0)), level=3),
                        Text("Konzepte"),
                        className="text-center p-4",
                    ),
                    Card(
                        Heading(str(stats.get("faiss_size", 0)), level=3),
                        Text("Embeddings"),
                        className="text-center p-4",
                    ),
                    gap="4",
                ),
                Row(
                    Button("🔄 Refresh", on_click="refresh", variant="secondary"),
                    Button("🔧 Reindex FAISS", on_click="reindex", variant="ghost"),
                    gap="2",
                ),
                className="card",
            ),

            # Video-Liste
            Card(
                Table(
                    columns=[
                        {"key": "id", "label": "ID"},
                        {"key": "category", "label": "Kategorie"},
                        {"key": "content", "label": "Inhalt (Vorschau)"},
                    ],
                    data=videos,
                    on_row_click="select_video",
                ),
                title=f"{len(videos)} Videos",
                className="card",
            ) if videos else Card(
                Text("Noch keine Videos. Gehe zu Upload →", className="text-center"),
                className="card",
            ),
        )

    async def refresh(self, event):
        await self._load_data()

    async def reindex(self, event):
        from toolboxv2 import get_app
        tools = get_app().get_mod("VideoRAG")
        if tools and tools.memory:
            count = tools.memory.rebuild_faiss_index()
            self.stats.value = tools.memory.stats()

    async def select_video(self, event):
        pass  # Detail-View Erweiterung
```

---

## Datei 9: `tests/test_pipeline.py`

```python
"""Tests für die Ingestion Pipeline"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np


class TestBuildDocument(unittest.TestCase):
    """Test Document-Building Logic (kein GPU nötig)"""

    def setUp(self):
        from toolboxv2.mods.VideoRAG.pipeline import VideoIngestionPipeline

        # Mock alle Services
        self.pipeline = VideoIngestionPipeline(
            memory=MagicMock(),
            embed_svc=MagicMock(),
            ocr_svc=MagicMock(),
            whisper_svc=MagicMock(),
        )

    def test_build_document_full(self):
        transcript = {"text": "Hallo Leute, heute zeige ich euch Rust", "segments": []}
        ocr_texts = ["fn main() {}", "cargo build"]
        meta = {
            "creator": "rustdev",
            "caption": "Rust Tutorial #1",
            "hashtags": ["rust", "coding"],
        }

        doc = self.pipeline._build_document(transcript, ocr_texts, meta)

        self.assertIn("Rust Tutorial #1", doc)
        self.assertIn("rustdev", doc)
        self.assertIn("Hallo Leute", doc)
        self.assertIn("fn main()", doc)
        self.assertIn("rust, coding", doc)

    def test_build_document_minimal(self):
        doc = self.pipeline._build_document({"text": ""}, [], {})
        self.assertEqual(doc, "")

    def test_extract_concepts(self):
        doc = "Transcript: Heute zeige ich euch einen HTTP Server in Rust"
        meta = {"hashtags": ["#rust", "coding"], "category": "tech", "creator": "dev"}

        concepts = self.pipeline._extract_concepts(doc, meta)

        self.assertIn("rust", concepts)
        self.assertIn("coding", concepts)
        self.assertIn("tech", concepts)
        self.assertIn("dev", concepts)
        self.assertIn("server", concepts)

    def test_concepts_max_50(self):
        # Langes Dokument → max 50 Concepts
        doc = " ".join([f"concept{i}" for i in range(200)])
        concepts = self.pipeline._extract_concepts(doc, {})
        self.assertLessEqual(len(concepts), 50)


class TestSearchEngine(unittest.TestCase):
    """Test Search Logic"""

    def test_format_result(self):
        from toolboxv2.mods.VideoRAG.search import VideoSearchEngine

        engine = VideoSearchEngine(memory=MagicMock(), embed_svc=MagicMock())

        result = engine._format_result({
            "id": "abc123",
            "score": 0.85432,
            "meta": {"category": "tech", "source": "videos/test.mp4"},
        })

        self.assertEqual(result["id"], "abc123")
        self.assertEqual(result["score"], 0.8543)
        self.assertEqual(result["category"], "tech")


class TestOCRService(unittest.TestCase):
    """Test OCR Deduplication"""

    def test_batch_extract_dedup(self):
        from toolboxv2.mods.VideoRAG.services.ocr import OCRService

        svc = OCRService()
        # Mock: Gleicher Text in mehreren Frames
        svc.extract_text = MagicMock(side_effect=[
            "fn main() {}",
            "fn main() {}",      # Duplikat
            "cargo build",
            "",                    # Leer
            "cargo build",        # Duplikat
        ])

        results = svc.batch_extract(["f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg", "f5.jpg"])

        self.assertEqual(results, ["fn main() {}", "cargo build"])


if __name__ == "__main__":
    unittest.main()
```

---

## Setup & Deployment

### Voraussetzungen auf RYZEN

```bash
# 1. System-Dependencies
sudo apt install ffmpeg

# 2. Python Packages
pip install whisper faiss-cpu numpy minio --break-system-packages

# 3. Qwen3-VL-Embedding (einmalig ~4GB Download)
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git
cd Qwen3-VL-Embedding && bash scripts/setup_environment.sh
huggingface-cli download Qwen/Qwen3-VL-Embedding-2B --local-dir ./models/Qwen3-VL-Embedding-2B

# 4. DeepSeek-OCR via Ollama
ollama pull deepseek-ocr

# 5. ToolBoxV2 Mod aktivieren
tb mod install VideoRAG
```

### Erster Test

```bash
# CLI: Einzelnes Video ingestieren
tb run VideoRAG ingest --video /path/to/reel.mp4 --category tech --creator rustdev

# CLI: Stats anzeigen
tb run VideoRAG stats

# API: Suche
curl "http://localhost:5000/api/VideoRAG/search?q=rust+http+server&k=5"

# MinUI: Im Browser
# → http://localhost:5000 → VideoRAG → video_search View
```
