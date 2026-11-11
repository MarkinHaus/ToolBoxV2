"""ISAA must work without litellm.

litellm is an optional emergency fallback (providers without a native
adapter), not a dependency. With litellm unimportable:
  * every module on the ISAA start / mount / agent path imports,
  * the router completes through a native adapter,
  * only a real fallback use fails -- with a clear ProviderError.
"""
import asyncio
import importlib
import re
import sys
import unittest
from pathlib import Path


class _BlockLitellm:
    """Make `import litellm` (and submodules) raise ImportError, as if not installed."""

    def __enter__(self):
        self._saved = {k: v for k, v in sys.modules.items()
                       if k == "litellm" or k.startswith("litellm.")}
        for k in self._saved:
            del sys.modules[k]
        sys.modules["litellm"] = None  # import machinery raises ImportError
        return self

    def __exit__(self, *exc):
        sys.modules.pop("litellm", None)
        sys.modules.update(self._saved)


# Loaded when ISAA starts, when /isaa is mounted, or when an agent runs.
START_PATH_MODULES = (
    "toolboxv2.mods.isaa.base.llm_router",
    "toolboxv2.mods.isaa.base.llm_router.router",
    "toolboxv2.mods.isaa.base.llm_router.adapters.setup",
    "toolboxv2.mods.isaa.base.llm_router.adapters.litellm_fallback",
    "toolboxv2.mods.isaa.base.IntelligentRateLimiter",
    "toolboxv2.mods.isaa.extras.adapter",
    "toolboxv2.mods.isaa.extras.modes",
    "toolboxv2.mods.isaa.base.Agent.flow_agent",
    "toolboxv2.mods.isaa.CodingAgent.coder",
    "toolboxv2.mods.isaa.module",
    "toolboxv2.mods.isaa.ui.app",
)


class ImportsWithoutLitellmTest(unittest.TestCase):

    def test_litellm_is_really_blocked(self):
        with _BlockLitellm():
            with self.assertRaises(ImportError):
                import litellm  # noqa: F401

    def test_start_path_modules_import(self):
        with _BlockLitellm():
            for name in START_PATH_MODULES:
                with self.subTest(module=name):
                    sys.modules.pop(name, None)
                    importlib.import_module(name)

    def test_module_level_litellm_only_in_legacy_providers(self):
        """A top-level `import litellm` is only allowed in files that are loaded
        lazily through IntelligentRateLimiter.setup_legacy_providers()."""
        import toolboxv2
        isaa = Path(toolboxv2.tb_root_dir) / "mods" / "isaa"
        allowed = {
            "base/IntelligentRateLimiter/gateway.py",
            "base/IntelligentRateLimiter/inception_litellm_provider.py",
            "base/IntelligentRateLimiter/minimax_provider.py",
            "base/IntelligentRateLimiter/zai_litellm_provider.py",
            "base/IntelligentRateLimiter/test_minimax_provider.py",
        }
        offenders = []
        for py in isaa.rglob("*.py"):
            rel = py.relative_to(isaa).as_posix()
            text = py.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"^(import litellm|from litellm)", text, re.M) and rel not in allowed:
                offenders.append(rel)
        self.assertEqual([], offenders)

    def test_legacy_providers_are_not_imported_by_the_package(self):
        with _BlockLitellm():
            for k in [k for k in sys.modules if k.startswith(
                    "toolboxv2.mods.isaa.base.IntelligentRateLimiter")]:
                del sys.modules[k]
            importlib.import_module("toolboxv2.mods.isaa.base.IntelligentRateLimiter")
            for legacy in ("gateway", "minimax_provider", "zai_litellm_provider",
                           "inception_litellm_provider"):
                self.assertNotIn(
                    f"toolboxv2.mods.isaa.base.IntelligentRateLimiter.{legacy}", sys.modules)


class RouterWithoutLitellmTest(unittest.TestCase):

    def _router(self):
        from toolboxv2.mods.isaa.base.llm_router.adapter import ProviderAdapter
        from toolboxv2.mods.isaa.base.llm_router.router import CompletionRouter
        from toolboxv2.mods.isaa.base.llm_router.types import CompletionResult, UsageData

        class NativeAdapter(ProviderAdapter):
            def __init__(self):
                super().__init__(base_url="http://native.invalid")
                self.calls = []

            async def complete(self, session, api_key, model, messages, tools, **kw):
                self.calls.append(model)
                return CompletionResult(content="pong", tool_calls=None, finish_reason="stop",
                                        usage=UsageData(), model=model)

            async def stream(self, *a, **kw):  # pragma: no cover - not used
                raise NotImplementedError

            async def embed(self, *a, **kw):  # pragma: no cover - not used
                raise NotImplementedError

        router = CompletionRouter(session=type("S", (), {"session": object()})())
        native = NativeAdapter()
        router.register("native", native)
        return router, native

    def test_native_adapter_completes_without_litellm(self):
        with _BlockLitellm():
            router, native = self._router()
            result = asyncio.run(router.complete(
                "native/model-x", [{"role": "user", "content": "ping"}], cache=False))
            self.assertEqual("pong", result.content)
            self.assertEqual(["model-x"], native.calls)

    def test_fallback_without_litellm_gives_clear_error(self):
        from toolboxv2.mods.isaa.base.llm_router.types import ProviderError
        with _BlockLitellm():
            router, _ = self._router()
            adapter, model, prefix = router.resolve("some-unknown-model")
            self.assertEqual("", prefix)
            with self.assertRaises(ProviderError) as cm:
                asyncio.run(adapter.complete(None, "", model, [], None))
            self.assertIn("litellm is not installed", str(cm.exception))
            self.assertNotIsInstance(cm.exception, ImportError)


if __name__ == "__main__":
    unittest.main()
