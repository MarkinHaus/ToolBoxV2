# tests/test_openrouter_filter.py
import unittest

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.free_providers.fetch_models import (
    _filter_openrouter,
)


def _model(mid, *, modalities=("text",), prompt="0", completion="0",
           supported=("tools", "temperature")):
    return {
        "id": mid,
        "architecture": {"input_modalities": list(modalities)},
        "pricing": {"prompt": prompt, "completion": completion},
        "supported_parameters": list(supported),
    }


class TestOpenRouterFilter(unittest.TestCase):
    def test_keeps_free_text_tools(self):
        payload = {"data": [_model("meta/llama-3.3-70b:free")]}
        self.assertEqual(_filter_openrouter(payload), ["meta/llama-3.3-70b:free"])

    def test_drops_non_free_suffix(self):
        payload = {"data": [_model("meta/llama-3.3-70b")]}
        self.assertEqual(_filter_openrouter(payload), [])

    def test_drops_paid_pricing_even_with_free_suffix(self):
        # defensive: :free suffix but pricing != 0 -> drop
        payload = {"data": [_model("x/y:free", prompt="0.0001")]}
        self.assertEqual(_filter_openrouter(payload), [])

    def test_drops_no_tools(self):
        payload = {"data": [_model("x/y:free", supported=("temperature",))]}
        self.assertEqual(_filter_openrouter(payload), [])

    def test_drops_no_text_modality(self):
        payload = {"data": [_model("x/y:free", modalities=("image",))]}
        self.assertEqual(_filter_openrouter(payload), [])

    def test_accepts_text_plus_image(self):
        payload = {"data": [_model("x/y:free", modalities=("text", "image"))]}
        self.assertEqual(_filter_openrouter(payload), ["x/y:free"])

    def test_handles_missing_fields(self):
        payload = {"data": [{"id": "x/y:free"}]}
        self.assertEqual(_filter_openrouter(payload), [])

    def test_empty_payload(self):
        self.assertEqual(_filter_openrouter({}), [])
        self.assertEqual(_filter_openrouter({"data": []}), [])

    def test_mixed_list(self):
        payload = {"data": [
            _model("good/a:free"),
            _model("paid/b", prompt="0.001"),
            _model("good/c:free", modalities=("text", "image")),
            _model("notools/d:free", supported=("temperature",)),
        ]}
        self.assertEqual(sorted(_filter_openrouter(payload)),
                         ["good/a:free", "good/c:free"])


if __name__ == "__main__":
    unittest.main()
