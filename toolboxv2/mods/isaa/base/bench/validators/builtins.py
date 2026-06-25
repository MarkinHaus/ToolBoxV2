"""
Built-in validators for systematic (non-LLM) binary checks.
Each one: pass or fail, no ambiguity.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from toolboxv2.mods.isaa.base.bench.core import CheckResult, TaskContext
from toolboxv2.mods.isaa.base.bench.validators import Validator, register


def _needs_word_boundary(value: str) -> bool:
    """Short or numeric needles cause substring false positives ('10' in '12').
    Default such values to word-boundary matching. Longer multi-word phrases
    (e.g. 'march 15') keep plain substring — their false positives are semantic
    and belong to the judge, not the matcher.
    """
    v = value.strip()
    if not v:
        return False
    # pure number, or a token ending in a number ("confidence: 1"), or a short word
    has_digit = any(ch.isdigit() for ch in v)
    is_short = len(v) <= 4
    is_single_token = " " not in v
    # multi-word phrases that END in a bare number also need boundary ("confidence: 1")
    ends_in_number = v.split()[-1].isdigit() if v.split() else False
    return (has_digit and (is_single_token or ends_in_number)) or (is_short and is_single_token)


def _is_present(needle: str, haystack: str, word_boundary: bool) -> bool:
    """Substring or word-boundary presence test (both lowercased by caller)."""
    if not word_boundary:
        return needle in haystack
    # \b doesn't anchor well around punctuation like ':' — build an explicit
    # boundary: needle not flanked by alphanumerics.
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


@register("contains")
class ContainsValidator(Validator):
    """Check if response contains a value (case-insensitive)."""
    name = "contains"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        value = str(self.params["value"]).lower()
        found = value in ctx.response.lower()
        return CheckResult("contains", found, f"'{value}' {'found' if found else 'not found'}")


@register("not_contains")
class NotContainsValidator(Validator):
    """Check that response does NOT contain a value.

    ponytail: numeric and short alphanumeric needles (e.g. "10", "1") match with
    word boundaries by default, so "12" no longer trips a not_contains "10" check
    and "CONFIDENCE: 10" no longer trips not_contains "1". Set word_boundary: false
    to force raw substring. Longer phrases use plain substring (unchanged).
    """
    name = "not_contains"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        value = str(self.params["value"]).lower()
        resp = ctx.response.lower()
        wb = self.params.get("word_boundary", _needs_word_boundary(value))
        present = _is_present(value, resp, wb)
        absent = not present
        mode = " (wb)" if wb else ""
        return CheckResult("not_contains", absent, f"'{value}' {'absent' if absent else 'present'}{mode}")


@register("equals")
class EqualsValidator(Validator):
    """Check if response stripped equals expected value (case-insensitive)."""
    name = "equals"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        expected = str(self.params["value"]).strip().lower()
        actual = ctx.response.strip().lower()
        passed = actual == expected
        return CheckResult("equals", passed, f"expected '{expected}', got '{actual[:80]}'")


@register("regex")
class RegexValidator(Validator):
    """Check if response matches a regex pattern."""
    name = "regex"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        pattern = self.params["pattern"]
        flags = re.IGNORECASE if self.params.get("ignorecase", True) else 0
        match = bool(re.search(pattern, ctx.response, flags))
        return CheckResult("regex", match, f"pattern '{pattern}' {'matched' if match else 'no match'}")


@register("char_count_gte")
class CharCountGteValidator(Validator):
    """Response must be at least N characters."""
    name = "char_count_gte"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        n = int(self.params["value"])
        actual = len(ctx.response)
        passed = actual >= n
        return CheckResult("char_count_gte", passed, f"len={actual}, need>={n}")


@register("char_count_lte")
class CharCountLteValidator(Validator):
    """Response must be at most N characters."""
    name = "char_count_lte"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        n = int(self.params["value"])
        actual = len(ctx.response)
        passed = actual <= n
        return CheckResult("char_count_lte", passed, f"len={actual}, need<={n}")


@register("max_tokens")
class MaxTokensValidator(Validator):
    """Response token count must not exceed N (rough word-split estimate)."""
    name = "max_tokens"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        n = int(self.params["value"])
        # Rough: split by whitespace
        actual = len(ctx.response.split())
        passed = actual <= n
        return CheckResult("max_tokens", passed, f"tokens~{actual}, max={n}")


@register("json_valid")
class JsonValidValidator(Validator):
    """Response must be valid JSON."""
    name = "json_valid"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        try:
            json.loads(ctx.response)
            return CheckResult("json_valid", True, "valid JSON")
        except (json.JSONDecodeError, ValueError) as e:
            return CheckResult("json_valid", False, f"invalid JSON: {e}")


@register("json_has_key")
class JsonHasKeyValidator(Validator):
    """Response JSON must contain a specific key."""
    name = "json_has_key"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        key = self.params["key"]
        try:
            data = json.loads(ctx.response)
            passed = key in data if isinstance(data, dict) else False
            return CheckResult("json_has_key", passed, f"key '{key}' {'found' if passed else 'missing'}")
        except (json.JSONDecodeError, ValueError):
            return CheckResult("json_has_key", False, "not valid JSON")


@register("tool_calls_lte")
class ToolCallsLteValidator(Validator):
    """Number of tool calls must not exceed N."""
    name = "tool_calls_lte"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        n = int(self.params["value"])
        actual = len(ctx.tool_calls)
        passed = actual <= n
        return CheckResult("tool_calls_lte", passed, f"tool_calls={actual}, max={n}")


@register("tool_calls_gte")
class ToolCallsGteValidator(Validator):
    """Number of tool calls must be at least N."""
    name = "tool_calls_gte"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        n = int(self.params["value"])
        actual = len(ctx.tool_calls)
        passed = actual >= n
        return CheckResult("tool_calls_gte", passed, f"tool_calls={actual}, min={n}")


@register("file_exists")
class FileExistsValidator(Validator):
    """Check if a file was created (in sandbox_state or filesystem)."""
    name = "file_exists"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        path = self.params["path"]
        # Check sandbox_state first
        if ctx.sandbox_state and path in ctx.sandbox_state.get("files_created", []):
            return CheckResult("file_exists", True, f"'{path}' in sandbox")
        # Fallback: actual filesystem
        exists = Path(path).exists()
        return CheckResult("file_exists", exists, f"'{path}' {'exists' if exists else 'missing'}")


@register("latency_lte")
class LatencyLteValidator(Validator):
    """Execution time must not exceed N seconds."""
    name = "latency_lte"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        n = float(self.params["value"])
        actual = ctx.execution_time
        passed = actual <= n
        return CheckResult("latency_lte", passed, f"time={actual:.2f}s, max={n}s")


@register("any_of")
class AnyOfValidator(Validator):
    """Response must contain at least one of the given values."""
    name = "any_of"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        values = self.params["values"]
        resp_lower = ctx.response.lower()
        found = [v for v in values if str(v).lower() in resp_lower]
        passed = len(found) > 0
        return CheckResult("any_of", passed,
                          f"found {found}" if passed else f"none of {values} found")


@register("all_of")
class AllOfValidator(Validator):
    """Response must contain all of the given values."""
    name = "all_of"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        values = self.params["values"]
        resp_lower = ctx.response.lower()
        missing = [v for v in values if str(v).lower() not in resp_lower]
        passed = len(missing) == 0
        return CheckResult("all_of", passed,
                          "all present" if passed else f"missing: {missing}")


@register("none_of")
class NoneOfValidator(Validator):
    """Response must contain none of the given values.

    ponytail: per-value word boundary for numeric/short tokens (same rule as
    not_contains), so "0" in none_of won't fire on "10"/"30" and
    "confidence: 1" won't fire on "CONFIDENCE: 10".
    """
    name = "none_of"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        values = self.params["values"]
        resp_lower = ctx.response.lower()
        wb_default = self.params.get("word_boundary", None)
        found = []
        for v in values:
            v_low = str(v).lower()
            wb = wb_default if wb_default is not None else _needs_word_boundary(v_low)
            if _is_present(v_low, resp_lower, wb):
                found.append(v)
        passed = len(found) == 0
        return CheckResult("none_of", passed,
                          "none present" if passed else f"found: {found}")
