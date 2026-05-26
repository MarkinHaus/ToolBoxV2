"""ISAA OCR Engine — unified router with quality tiers.

Tiers:
  FAST      → Tesseract 5 (CPU, always-on fallback)
  BALANCED  → PaddleOCR-VL 1.5 (0.9B VLM via transformers, GPU)
  ACCURATE  → DeepSeek-OCR 2 (3B VLM, bf16/4bit/8bit, GPU)
  API       → Mistral OCR (/v1/ocr) OR generic Vision LLM via litellm_complete

Usage:
    from toolboxv2.mods.isaa.extras.ocr_engine import OCRRouter, IsaaOCRConfig
    router = OCRRouter(IsaaOCRConfig())
    result = await router.ocr("path.pdf", tier="accurate")
    print(result.text); print(result.to_markdown()); data = result.to_json()
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import tempfile
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

from aiohttp import ClientSession
from pydantic import BaseModel, Field

from toolboxv2 import get_logger, get_app
from toolboxv2.utils.manifest.schema import OCRTier, IsaaOCRAPIConfig, IsaaOCRConfig, IsaaOCRManifestConfig

logger = get_logger()

def get_session() -> ClientSession:
    return get_app().session.session

# ─── Data Models ───────────────────────────────────────────────────────────

class OCRPage:
    __slots__ = ("index", "text", "confidence", "metadata")

    def __init__(self, index: int, text: str, confidence: float | None = None, metadata: dict | None = None):
        self.index = index
        self.text = text
        self.confidence = confidence
        self.metadata = metadata or {}


class OCRResult:
    __slots__ = ("pages", "engine", "tier", "elapsed_s", "fallback_used")

    def __init__(self, pages: list[OCRPage], engine: str, tier: OCRTier,
                 elapsed_s: float, fallback_used: bool = False):
        self.pages = pages
        self.engine = engine
        self.tier = tier
        self.elapsed_s = elapsed_s
        self.fallback_used = fallback_used

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def to_json(self) -> dict:
        return {
            "engine": self.engine,
            "tier": self.tier.value,
            "elapsed_s": round(self.elapsed_s, 3),
            "fallback_used": self.fallback_used,
            "page_count": len(self.pages),
            "pages": [
                {
                    "index": p.index,
                    "text": p.text,
                    "confidence": p.confidence,
                    "metadata": p.metadata,
                }
                for p in self.pages
            ],
        }

    def to_markdown(self) -> str:
        if not self.pages:
            return ""
        if len(self.pages) == 1:
            return self.pages[0].text
        parts = []
        for p in self.pages:
            parts.append(f"<!-- page {p.index + 1} -->")
            parts.append(p.text)
        return "\n\n".join(parts)


# ─── Input Normalization ───────────────────────────────────────────────────

async def _load_images(source: str | Path | bytes, *, timeout_s: float, pdf_dpi: int) -> list[Any]:
    """Normalize input → list[PIL.Image.Image]."""
    from PIL import Image as PILImage

    if isinstance(source, bytes):
        # Sniff PDF magic
        if source[:5] == b"%PDF-":
            return _pdf_bytes_to_images(source, pdf_dpi)
        return [PILImage.open(io.BytesIO(source)).convert("RGB")]

    source = str(source)
    if source.startswith(("http://", "https://")):
        import httpx
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; isaa-ocr/1.0; +https://github.com/MarkinHaus/ToolBoxV2)"
            ),
            "Accept": "application/pdf,image/*,*/*;q=0.8",
        }
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_s, headers=headers) as client:
            resp = await client.get(source)
            resp.raise_for_status()
            data = resp.content
            ctype = (resp.headers.get("content-type") or "").lower()
        if "pdf" in ctype or source.lower().split("?")[0].endswith(".pdf") or data[:5] == b"%PDF-":
            return _pdf_bytes_to_images(data, pdf_dpi)
        return [PILImage.open(io.BytesIO(data)).convert("RGB")]

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"OCR input not found: {path}")
    if path.suffix.lower() == ".pdf":
        return _pdf_bytes_to_images(path.read_bytes(), pdf_dpi)
    return [PILImage.open(path).convert("RGB")]


def _pdf_bytes_to_images(pdf_bytes: bytes, dpi: int) -> list[Any]:
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise ImportError("pymupdf required for PDF input: pip install pymupdf") from e
    from PIL import Image as PILImage
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[Any] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    finally:
        doc.close()
    return images


def _image_to_b64(img: Any, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _image_to_data_url(img: Any, fmt: str = "PNG") -> str:
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{_image_to_b64(img, fmt)}"


# ─── Engine ABC ────────────────────────────────────────────────────────────

class OCREngine(ABC):
    name: str = ""
    tier: OCRTier = OCRTier.FAST

    def __init__(self) -> None:
        self._loaded: bool = False

    @abstractmethod
    def available(self) -> bool:
        """Return True iff dependencies are importable and ready."""

    async def load(self) -> None:
        self._loaded = True

    async def ensure_loaded(self) -> None:
        if not self._loaded:
            await self.load()

    @abstractmethod
    async def ocr(self, images: list[Any]) -> list[OCRPage]:
        """Run OCR on list of PIL.Image, return pages."""


# ─── Tesseract (FAST) ──────────────────────────────────────────────────────

class TesseractEngine(OCREngine):
    name = "tesseract"
    tier = OCRTier.FAST

    def __init__(self, langs: str = "eng+deu"):
        super().__init__()
        self.langs = langs

    def available(self) -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    async def ocr(self, images: list[Any]) -> list[OCRPage]:
        import pytesseract
        pages: list[OCRPage] = []
        for i, img in enumerate(images):
            text = await asyncio.to_thread(pytesseract.image_to_string, img, lang=self.langs)
            data = await asyncio.to_thread(
                pytesseract.image_to_data, img, lang=self.langs,
                output_type=pytesseract.Output.DICT,
            )
            confs = [int(c) for c, t in zip(data["conf"], data["text"]) if t.strip() and str(c).lstrip("-").isdigit() and int(c) > 0]
            avg = (sum(confs) / len(confs) / 100.0) if confs else None
            pages.append(OCRPage(index=i, text=text.strip(), confidence=avg, metadata={"lang": self.langs}))
        return pages


# ─── PaddleOCR-VL 1.5 (BALANCED) ───────────────────────────────────────────

class PaddleOCRVLEngine(OCREngine):
    """PaddleOCR-VL-1.5 via transformers v5+ (AutoModelForImageTextToText).

    Ref: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5
    """
    name = "paddleocr_vl"
    tier = OCRTier.BALANCED
    _PROMPTS = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "chart": "Chart Recognition:",
        "formula": "Formula Recognition:",
        "spotting": "Spotting:",
        "seal": "Seal Recognition:",
    }

    def __init__(self, model_id: str, task: str = "ocr"):
        super().__init__()
        self.model_id = model_id
        self.task = task if task in self._PROMPTS else "ocr"
        self._model = None
        self._processor = None
        self._device = None

    def available(self) -> bool:
        try:
            import transformers
            import torch  # noqa: F401
            major = int(transformers.__version__.split(".")[0])
            return major >= 5
        except Exception:
            return False

    async def load(self) -> None:
        await asyncio.to_thread(self._load_sync)
        self._loaded = True

    def _load_sync(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, attn_implementation="flash_attention_2", **kwargs
            ).to(self._device).eval()
        except Exception:
            self._model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs).to(self._device).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        logger.info("PaddleOCR-VL loaded: %s on %s", self.model_id, self._device)

    async def ocr(self, images: list[Any]) -> list[OCRPage]:
        await self.ensure_loaded()
        pages: list[OCRPage] = []
        for i, img in enumerate(images):
            text = await asyncio.to_thread(self._infer_single, img)
            pages.append(OCRPage(index=i, text=text, metadata={"task": self.task}))
        return pages

    def _infer_single(self, img: Any) -> str:
        prompt = self._PROMPTS[self.task]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self._device)
        outputs = self._model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        gen = outputs[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


# ─── DeepSeek-OCR 2 (ACCURATE) ─────────────────────────────────────────────

class DeepSeekOCR2Engine(OCREngine):
    """DeepSeek-OCR-2 via transformers, optional 4bit/8bit via bitsandbytes.

    Ref: https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
    The model exposes a custom `.infer()` method taking an image file path.
    """
    name = "deepseek_ocr2"
    tier = OCRTier.ACCURATE
    _PROMPT_DOC = "<image>\n<|grounding|>Convert the document to markdown."
    _PROMPT_FREE = "<image>\nFree OCR."

    def __init__(self, model_id: str, quantize: str = "bf16",
                 base_size: int = 1024, image_size: int = 768):
        super().__init__()
        self.model_id = model_id
        self.quantize = quantize  # bf16 | 4bit | 8bit
        self.base_size = base_size
        self.image_size = image_size
        self._model = None
        self._tokenizer = None

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch
            if not torch.cuda.is_available():
                logger.warning("DeepSeek-OCR-2 requires CUDA; not available")
                return False
            if self.quantize in ("4bit", "8bit"):
                import bitsandbytes  # noqa: F401
            return True
        except Exception:
            return False

    async def load(self) -> None:
        await asyncio.to_thread(self._load_sync)
        self._loaded = True

    def _load_sync(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "use_safetensors": True,
        }
        try:
            load_kwargs["_attn_implementation"] = "flash_attention_2"
            model = AutoModel.from_pretrained(self.model_id, **load_kwargs)
        except Exception:
            load_kwargs.pop("_attn_implementation", None)
            model = AutoModel.from_pretrained(self.model_id, **load_kwargs)

        if self.quantize == "4bit":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_quant_type="nf4")
            model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True, use_safetensors=True,
                quantization_config=bnb, device_map="auto",
            )
        elif self.quantize == "8bit":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True, use_safetensors=True,
                quantization_config=bnb, device_map="auto",
            )
        else:  # bf16
            model = model.eval().cuda().to(torch.bfloat16)
        self._model = model
        logger.info("DeepSeek-OCR-2 loaded: %s (%s)", self.model_id, self.quantize)

    async def ocr(self, images: list[Any]) -> list[OCRPage]:
        await self.ensure_loaded()
        pages: list[OCRPage] = []
        for i, img in enumerate(images):
            text = await asyncio.to_thread(self._infer_single, img)
            pages.append(OCRPage(index=i, text=text, metadata={"prompt": "grounding"}))
        return pages

    def _infer_single(self, img: Any) -> str:
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "page.png")
            img.save(img_path, "PNG")
            out_dir = os.path.join(td, "out")
            os.makedirs(out_dir, exist_ok=True)
            res = self._model.infer(
                self._tokenizer,
                prompt=self._PROMPT_DOC,
                image_file=img_path,
                output_path=out_dir,
                base_size=self.base_size,
                image_size=self.image_size,
                crop_mode=True,
                save_results=True,
            )
            text = ""
            if isinstance(res, str) and res.strip():
                text = res.strip()
            else:
                md_path = os.path.join(out_dir, "result.mmd")
                txt_path = os.path.join(out_dir, "result.txt")
                for cand in (md_path, txt_path):
                    if os.path.exists(cand):
                        text = Path(cand).read_text(encoding="utf-8", errors="replace").strip()
                        break
            return text


# ─── Mistral OCR API (/v1/ocr) ─────────────────────────────────────────────

class MistralOCRAPIEngine(OCREngine):
    """Mistral OCR via dedicated /v1/ocr endpoint (NOT chat completion)."""
    name = "mistral_ocr"
    tier = OCRTier.API

    def __init__(self, api_key: str, model: str, endpoint: str, timeout_s: float = 60.0):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def available(self) -> bool:
        try:
            import httpx  # noqa: F401
            return bool(self.api_key)
        except Exception:
            return False

    async def ocr(self, images: list[Any]) -> list[OCRPage]:
        import httpx
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        pages: list[OCRPage] = []
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            for i, img in enumerate(images):
                data_url = _image_to_data_url(img, "PNG")
                payload = {
                    "model": self.model,
                    "document": {"type": "image_url", "image_url": data_url},
                }
                resp = await client.post(self.endpoint, headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()
                api_pages = body.get("pages") or []
                if api_pages:
                    for p in api_pages:
                        pages.append(OCRPage(
                            index=len(pages),
                            text=(p.get("markdown") or p.get("text") or "").strip(),
                            metadata={"src_page_index": i, "api_index": p.get("index")},
                        ))
                else:
                    pages.append(OCRPage(index=i, text="", metadata={"warning": "empty_response"}))
        return pages


# ─── Generic Vision LLM (via ISAA litellm_complete adapter) ────────────────

class VisionLLMEngine(OCREngine):
    """OCR via any vision-capable LLM through ISAA's litellm_complete adapter.

    Works for Claude, GPT-4o, Gemini, Pixtral, etc. — whatever the adapter routes.
    """
    name = "vision_llm"
    tier = OCRTier.API

    def __init__(self, model: str, prompt: str):
        super().__init__()
        self.model = model
        self.prompt = prompt

    def available(self) -> bool:
        try:
            from toolboxv2.mods.isaa.extras.adapter import litellm_complete  # noqa: F401
            return bool(self.model)
        except Exception:
            return False

    async def ocr(self, images: list[Any]) -> list[OCRPage]:
        from toolboxv2.mods.isaa.extras.adapter import litellm_complete
        pages: list[OCRPage] = []
        for i, img in enumerate(images):
            data_url = _image_to_data_url(img, "PNG")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": self.prompt},
                ],
            }]
            response = await litellm_complete(model=self.model, messages=messages)
            text = _extract_response_text(response)
            pages.append(OCRPage(index=i, text=text.strip(), metadata={"model": self.model}))
        return pages


def _extract_response_text(response: Any) -> str:
    """Best-effort extraction of text from various LLM response shapes."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "choices" in response and response["choices"]:
            msg = response["choices"][0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        for key in ("text", "content", "output"):
            if key in response and isinstance(response[key], str):
                return response[key]
    # pydantic-like / OpenAI SDK objects
    try:
        return response.choices[0].message.content  # type: ignore[attr-defined]
    except Exception:
        return str(response)


# ─── Router ────────────────────────────────────────────────────────────────

class OCRRouter:
    def __init__(self, config: IsaaOCRConfig):
        if not config.enabled:
            raise RuntimeError("IsaaOCRConfig.enabled is False")
        self.config = config
        self._engines: dict[OCRTier, list[OCREngine]] = {t: [] for t in OCRTier}
        self._build_engines()

    def _build_engines(self) -> None:
        m, a = self.config.manifest, self.config.api
        if m.tesseract:
            self._engines[OCRTier.FAST].append(TesseractEngine(langs=m.tesseract_langs))
        if m.paddleocr_vl:
            self._engines[OCRTier.BALANCED].append(
                PaddleOCRVLEngine(model_id=m.paddleocr_vl_model, task=m.paddleocr_vl_task)
            )
        if m.deepseek_ocr2:
            self._engines[OCRTier.ACCURATE].append(DeepSeekOCR2Engine(
                model_id=m.deepseek_ocr2_model,
                quantize=m.deepseek_ocr2_quantize,
                base_size=m.deepseek_ocr2_base_size,
                image_size=m.deepseek_ocr2_image_size,
            ))
        if a.mistral_enabled:
            key = os.getenv(a.mistral_api_key_env, "")
            if key:
                self._engines[OCRTier.API].append(MistralOCRAPIEngine(
                    api_key=key, model=a.mistral_model,
                    endpoint=a.mistral_endpoint, timeout_s=self.config.http_timeout_s,
                ))
        if a.vision_llm_enabled and a.vision_llm_model:
            self._engines[OCRTier.API].append(VisionLLMEngine(
                model=a.vision_llm_model, prompt=a.vision_llm_prompt
            ))

    def list_available(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for tier, engines in self._engines.items():
            out[tier.value] = [e.name for e in engines if e.available()]
        return out

    def _resolve_chain(self, requested: OCRTier) -> list[OCREngine]:
        order: list[OCRTier] = [requested]
        for t in self.config.fallback_chain:
            if t != requested and t not in order:
                order.append(t)
        chain: list[OCREngine] = []
        for tier in order:
            for engine in self._engines.get(tier, []):
                if engine.available():
                    chain.append(engine)
        return chain

    async def ocr(self, source: str | Path | bytes, tier: OCRTier | str | None = None) -> OCRResult:
        """Run OCR with tier-based selection and fallback."""
        requested = OCRTier(tier) if isinstance(tier, str) else (tier or self.config.default_tier)
        images = await _load_images(
            source, timeout_s=self.config.http_timeout_s, pdf_dpi=self.config.pdf_dpi
        )
        if not images:
            raise ValueError("No images extracted from source")

        chain = self._resolve_chain(requested)
        if not chain:
            raise RuntimeError(
                f"No available engines for tier={requested.value}. "
                f"Status: {self.list_available()}"
            )

        last_err: Exception | None = None
        for idx, engine in enumerate(chain):
            t0 = time.perf_counter()
            try:
                pages = await engine.ocr(images)
                return OCRResult(
                    pages=pages, engine=engine.name, tier=engine.tier,
                    elapsed_s=time.perf_counter() - t0,
                    fallback_used=(idx > 0 or engine.tier != requested),
                )
            except Exception as e:
                logger.warning("OCR engine %s failed: %s", engine.name, e)
                last_err = e
                continue
        raise RuntimeError(f"All OCR engines in chain failed; last error: {last_err}")


# ─── Installer Helper ──────────────────────────────────────────────────────

class OCRInstaller:
    """Verifies and downloads OCR engine dependencies / model weights."""

    INSTALL_HINTS: dict[str, str] = {
        "tesseract": (
            "pip install pytesseract && "
            "apt-get install -y tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng"
        ),
        "paddleocr_vl": 'pip install "transformers>=5.0.0" torch pillow accelerate',
        "deepseek_ocr2": (
            "pip install torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3 "
            "einops addict easydict pillow && "
            "pip install flash-attn==2.7.3 --no-build-isolation"
        ),
        "deepseek_ocr2_quant": "pip install bitsandbytes",
        "mistral_ocr": "pip install httpx  # then set MISTRAL_API_KEY env var",
        "vision_llm": "ensure toolboxv2.mods.isaa.extras.adapter.litellm_complete is importable",
        "pdf": "pip install pymupdf",
    }

    @staticmethod
    def status(config: IsaaOCRConfig | None = None) -> dict:
        config = config or IsaaOCRConfig()
        out: dict[str, Any] = {}

        # Tesseract
        try:
            import pytesseract
            ver = str(pytesseract.get_tesseract_version())
            out["tesseract"] = {"ok": True, "version": ver, "langs": config.manifest.tesseract_langs}
        except Exception as e:
            out["tesseract"] = {"ok": False, "error": str(e), "hint": OCRInstaller.INSTALL_HINTS["tesseract"]}

        # PaddleOCR-VL
        out["paddleocr_vl"] = OCRInstaller._check_hf_engine(
            config.manifest.paddleocr_vl_model, "paddleocr_vl", min_transformers=5
        )

        # DeepSeek-OCR-2
        ds = OCRInstaller._check_hf_engine(config.manifest.deepseek_ocr2_model, "deepseek_ocr2")
        try:
            import torch
            ds["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:
            ds["cuda_available"] = False
        if config.manifest.deepseek_ocr2_quantize in ("4bit", "8bit"):
            try:
                import bitsandbytes  # noqa: F401
                ds["quant_ok"] = True
            except ImportError:
                ds["quant_ok"] = False
                ds["quant_hint"] = OCRInstaller.INSTALL_HINTS["deepseek_ocr2_quant"]
        out["deepseek_ocr2"] = ds

        # Mistral OCR API
        api_key = os.getenv(config.api.mistral_api_key_env, "")
        out["mistral_ocr"] = {
            "ok": bool(api_key) and OCRInstaller._has_httpx(),
            "api_key_present": bool(api_key),
            "env_var": config.api.mistral_api_key_env,
            "model": config.api.mistral_model,
        }

        # Vision LLM adapter
        try:
            from toolboxv2.mods.isaa.extras.adapter import litellm_complete  # noqa: F401
            out["vision_llm"] = {"ok": True, "model": config.api.vision_llm_model}
        except Exception as e:
            out["vision_llm"] = {"ok": False, "error": str(e), "hint": OCRInstaller.INSTALL_HINTS["vision_llm"]}

        # PDF support
        try:
            import fitz  # noqa: F401
            out["pdf_support"] = {"ok": True}
        except ImportError:
            out["pdf_support"] = {"ok": False, "hint": OCRInstaller.INSTALL_HINTS["pdf"]}

        return out

    @staticmethod
    def _has_httpx() -> bool:
        try:
            import httpx  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_hf_engine(model_id: str, key: str, min_transformers: int = 4) -> dict:
        try:
            import transformers
            major = int(transformers.__version__.split(".")[0])
            if major < min_transformers:
                return {
                    "ok": False,
                    "error": f"transformers v{transformers.__version__} < required v{min_transformers}",
                    "hint": OCRInstaller.INSTALL_HINTS[key],
                }
        except ImportError as e:
            return {"ok": False, "error": str(e), "hint": OCRInstaller.INSTALL_HINTS[key]}
        cached = OCRInstaller._hf_cached(model_id)
        return {
            "ok": True,
            "transformers_version": transformers.__version__,
            "model_id": model_id,
            "model_cached": cached,
            "download_hint": (
                f"python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('{model_id}')\""
            ) if not cached else None,
        }

    @staticmethod
    def _hf_cached(model_id: str) -> bool:
        try:
            from huggingface_hub import try_to_load_from_cache
            # Probe a file that should exist in any HF model
            for fname in ("config.json", "tokenizer_config.json"):
                p = try_to_load_from_cache(repo_id=model_id, filename=fname)
                if p is not None and p is not False:
                    return True
            return False
        except Exception:
            return False

    @staticmethod
    async def download_model(model_id: str, revision: str | None = None) -> str:
        """Download model weights via HF snapshot_download. Returns local path."""
        from huggingface_hub import snapshot_download
        kwargs: dict[str, Any] = {"repo_id": model_id}
        if revision:
            kwargs["revision"] = revision
        return await asyncio.to_thread(snapshot_download, **kwargs)


# ─── Module API ────────────────────────────────────────────────────────────

__all__ = [
    "IsaaOCRConfig",
    "IsaaOCRManifestConfig",
    "IsaaOCRAPIConfig",
    "OCRTier",
    "OCRPage",
    "OCRResult",
    "OCREngine",
    "TesseractEngine",
    "PaddleOCRVLEngine",
    "DeepSeekOCR2Engine",
    "MistralOCRAPIEngine",
    "VisionLLMEngine",
    "OCRRouter",
    "OCRInstaller",
]
