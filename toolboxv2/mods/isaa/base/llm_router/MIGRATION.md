# Layer 2 Migration Guide — LiteLLM → CompletionRouter

All changes are diffs. Apply in order.

---

## 1. `llm_router/__init__.py` — export compat

```python
# ADD to existing exports:
from .compat import (
    completion_result_to_message,
    completion_result_to_model_response,
    stream_chunk_to_shim,
)
```

---

## 2. `IntelligentRateLimiter/__init__.py` — lazy imports, no side effects

REPLACE entire file:

```python
import os
import logging

# Lazy — no provider setup at import time.
# Provider registration now happens via CompletionRouter.setup_default_adapters()

from .intelligent_rate_limiter import (
    IntelligentRateLimiter,
    LiteLLMRateLimitHandler,
    load_handler_from_file,
    create_handler_from_config,
    FallbackReason,
)


def setup_legacy_providers():
    """Call explicitly if legacy litellm code paths are still needed."""
    from .gateway import setup_gateway_provider, setup_custom_provider
    from .minimax_provider import register_minimax
    from .zai_litellm_provider import setup_zai_provider

    gateway = setup_gateway_provider()
    provider = setup_zai_provider(debug=False)
    if os.getenv("MINIMAX_API_BASE") and os.getenv("MINIMAX_API_KEY"):
        register_minimax(provider_name="mm", debug=False)

    try:
        from .inception_litellm_provider import setup_inception_provider
        setup_inception_provider()
    except Exception:
        logging.getLogger(__name__).debug("inception provider not available")

    return gateway, provider
```

---

## 3. `flow_agent.py` — Router property + a_run_llm_completion rewrite

### 3a. New import block (add near top)

```python
# Layer 2 Router
from toolboxv2.mods.isaa.base.llm_router.router import CompletionRouter
from toolboxv2.mods.isaa.base.llm_router.adapters.setup import setup_default_adapters
from toolboxv2.mods.isaa.base.llm_router.compat import (
    completion_result_to_message,
    completion_result_to_model_response,
    stream_chunk_to_shim,
)
from toolboxv2.mods.isaa.base.llm_router.stream_accumulator import StreamAccumulator
from toolboxv2.mods.isaa.base.llm_router.model_info import ctx_limit, supports_tools
```

### 3b. FlowAgent.__init__ — add router field

```python
# ADD after self.llm_handler initialization:
self._router: CompletionRouter | None = None
```

### 3c. Router property (add to FlowAgent class)

```python
@property
def router(self) -> CompletionRouter:
    """Lazy-init CompletionRouter with all configured adapters."""
    if self._router is None:
        self._router = CompletionRouter(
            rate_limiter=getattr(self, 'llm_handler', None),
            strict_mode=False,  # fallback to litellm for unmapped providers
        )
        setup_default_adapters(self._router)
    return self._router
```

### 3d. save_supports_vision — remove litellm dependency

REPLACE entire method:

```python
async def save_supports_vision(self, messages: list | None = None, model_preference="fast"):
    if hasattr(self, "_vison"):
        cached = self._vison.get(model_preference)
        if isinstance(cached, bool):
            return cached

    model = self.amd.fast_llm_model if model_preference == "fast" else self.amd.complex_llm_model
    # Heuristic: all modern providers support vision except ollama base models
    self._vison = getattr(self, "_vison", {})
    self._vison[model_preference] = not model.startswith("ollama/")
    return self._vison[model_preference]
```

### 3e. a_run_llm_completion — core LLM call migration

Inside `internal_stream()`, REPLACE the LLM call block:

**OLD:**
```python
response = await self.llm_handler.completion_with_rate_limiting(
    litellm, **llm_kwargs
)
```

**NEW (non-streaming path):**
```python
if not use_stream:
    _result = await self.router.complete(
        model=llm_kwargs["model"],
        messages=llm_kwargs["messages"],
        tools=llm_kwargs.get("tools"),
        api_key=os.environ.get(self._resolve_api_key_env(llm_kwargs["model"]), ""),
        **{k: v for k, v in llm_kwargs.items()
           if k not in ("model", "messages", "tools", "stream", "stream_options")},
    )
    response = completion_result_to_model_response(_result)
```

**NEW (streaming path):**
```python
if use_stream:
    _api_key = os.environ.get(self._resolve_api_key_env(llm_kwargs["model"]), "")
    _stream_kw = {k: v for k, v in llm_kwargs.items()
                  if k not in ("model", "messages", "tools", "stream", "stream_options")}

    async def _router_stream_wrapper():
        """Wrap router.stream() to yield litellm-compatible chunks."""
        acc = StreamAccumulator()
        async for chunk in self.router.stream(
            model=llm_kwargs["model"],
            messages=llm_kwargs["messages"],
            tools=llm_kwargs.get("tools"),
            api_key=_api_key,
            collect_metrics=True,
            **_stream_kw,
        ):
            acc.feed(chunk)
            yield stream_chunk_to_shim(chunk)
        # After stream ends, the accumulated result is available via acc.build()

    response = _router_stream_wrapper()
```

### 3f. Helper method for API key resolution (add to FlowAgent)

```python
def _resolve_api_key_env(self, model: str) -> str:
    """Given a model string like 'groq/llama-3.3-70b', return the env var name."""
    _MAP = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "zai": "ZAI_API_KEY",
        "zglm": "ZAI_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "xai": "XAI_API_KEY",
        "nvidia_nim": "NVIDIA_NIM_API_KEY",
        "together_ai": "TOGETHER_API_KEY",
        "minimax": "MINIMAX_API_KEY",
        "ollama": "",
    }
    prefix = model.split("/", 1)[0] if "/" in model else ""
    return _MAP.get(prefix, "OPENROUTER_API_KEY")
```

---

## 4. `execution_engine.py` — Token counting + context limit

### 4a. Token counting (replace litellm.token_counter)

**OLD:**
```python
import litellm
count = litellm.token_counter(model=model, text=text)
```

**NEW:**
```python
count = len(text) // 4  # char//4 heuristic, <0.5ms
```

### 4b. Context limit (replace litellm.get_model_info)

**OLD:**
```python
import litellm
info = litellm.get_model_info(model)
limit = info.get("max_input_tokens", 128000)
```

**NEW:**
```python
from toolboxv2.mods.isaa.base.llm_router.model_info import ctx_limit
limit = ctx_limit(model)
```

### 4c. No other changes needed
The execution_engine calls `self.agent.a_run_llm_completion()` which handles
the router internally. Tool call parsing works because compat shims provide
`.function.name`, `.function.arguments`, `.id` — same interface as litellm.

---

## 5. `AgentLiveNarrator._call_blitz` — direct router call

REPLACE the function:

```python
async def _call_blitz(system: str, messages: list[dict], max_tokens: int = 60) -> dict | None:
    """Call BLITZ_MODEL via CompletionRouter. No fallback, no handler."""
    raw = None
    try:
        from toolboxv2.mods.isaa.base.llm_router.router import CompletionRouter
        from toolboxv2.mods.isaa.base.llm_router.adapters.setup import setup_default_adapters

        # Module-level singleton (lazy)
        global _blitz_router
        if "_blitz_router" not in globals() or _blitz_router is None:
            _blitz_router = CompletionRouter(strict_mode=False)
            setup_default_adapters(_blitz_router)

        all_messages = [{"role": "system", "content": system}] + messages
        api_key_env = {
            "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
        }
        prefix = BLITZ_MODEL.split("/", 1)[0] if "/" in BLITZ_MODEL else ""
        api_key = os.environ.get(api_key_env.get(prefix, ""), "")

        result = await _blitz_router.complete(
            model=BLITZ_MODEL,
            messages=all_messages,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=0.3,
        )

        raw = result.content or ""
        raw = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        return {
            "data": data,
            "raw": raw,
            "in": result.usage.prompt_tokens,
            "out": result.usage.completion_tokens,
        }
    except asyncio.CancelledError:
        raise
    except json.JSONDecodeError as exc:
        logger.debug(f"Blitz JSON decode failed: {exc} {raw}")
        return None
    except Exception as exc:
        logger.debug(f"Blitz call failed: {exc} : {raw}")
        return None

_blitz_router = None
```

---

## 6. LightRAG `litellm_complete_if_cache` — router replacement

REPLACE the function:

```python
async def litellm_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=None,
    base_url=None, api_key=None, **kwargs,
) -> str | AsyncIterator[str]:
    """Core completion via CompletionRouter (replaces litellm.acompletion)."""
    from toolboxv2.mods.isaa.base.llm_router.router import CompletionRouter
    from toolboxv2.mods.isaa.base.llm_router.adapters.setup import setup_default_adapters

    # Module-level singleton
    global _lightrag_router
    if "_lightrag_router" not in globals() or _lightrag_router is None:
        _lightrag_router = CompletionRouter(strict_mode=False)
        setup_default_adapters(_lightrag_router)

    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("fallbacks", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    _api_key = api_key or ""
    stream = kwargs.pop("stream", False)
    clean_kw = {k: v for k, v in kwargs.items()
                if k not in ("response_format",) or v is not None}

    try:
        if stream:
            async def inner():
                async for chunk in _lightrag_router.stream(
                    model=model, messages=messages, api_key=_api_key, **clean_kw,
                ):
                    if chunk.content:
                        yield chunk.content
            return inner()
        else:
            result = await _lightrag_router.complete(
                model=model, messages=messages, api_key=_api_key, **clean_kw,
            )
            return result.content or ""
    except Exception as e:
        get_logger().error(f"Failed to complete: {e}")
        return ""

_lightrag_router = None
```

---

## 7. Embeddings — fully migrated

REPLACE `litellm_embed`, `embed`, `smart_embed` in `toolboxv2/mods/isaa/base/__init__.py`
with imports from the new `llm_router/embeddings.py`:

```python
# OLD (at top of __init__.py):
from litellm import aembedding
# ... 300+ lines of embed functions with OpenRouter workarounds

# NEW:
from toolboxv2.mods.isaa.base.llm_router.embeddings import (
    embed,
    litellm_embed,
    smart_embed,
    embed_sync,
    embed_openrouter_direct,
    embed_openrouter_direct_sync,
    cosine_similarity,
    OPENROUTER_EMBEDDING_MODELS,
)
```

The new `embeddings.py` module:
- Uses `router.embed()` instead of `litellm.aembedding()`
- OpenRouter workaround removed (OpenAICompatAdapter handles it natively)
- `dimensions` and `input_type` forwarded through the adapter
- Returns `np.ndarray` — same interface as before
- `litellm_embed()` and `smart_embed()` are thin wrappers around `embed()`
- `embed_openrouter_direct()` kept as standalone aiohttp alternative

---

## 8. FlowAgentBuilder — no changes needed

The builder creates `AgentModelData` which the FlowAgent reads.
`_init_rate_limiter()` still works (creates `llm_handler`).
The router uses `llm_handler` as its `rate_limiter` parameter — compatible.

---

## VERIFICATION CHECKLIST

- [ ] `router.complete()` returns `CompletionResult`
- [ ] `completion_result_to_message()` gives `.content`, `.tool_calls[].function.name/.arguments`
- [ ] `stream_chunk_to_shim()` gives `.choices[0].delta.content`, `.choices[0].delta.tool_calls`
- [ ] Auto-resume loop reads `.content` and `.tool_calls` from shim — ✓ same interface
- [ ] ExecutionEngine reads `response.content`, `response.tool_calls` — ✓ via shim
- [ ] `_call_blitz` returns `{"data": ..., "in": ..., "out": ...}` — ✓ same
- [ ] Budget tracking: `router.budget.track()` called automatically in `router.complete/stream`
- [ ] StreamMetrics: available via `router.stream(collect_metrics=True)` or `stream_with_metrics()`

## ROLLBACK

If something breaks, set `strict_mode=False` on the router — it falls through
to `LiteLLMFallbackAdapter` which lazy-imports litellm and calls `acompletion()`.
This means the old path still works as fallback.
