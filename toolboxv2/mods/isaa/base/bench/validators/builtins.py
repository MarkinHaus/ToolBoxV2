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
    """Check that response does NOT contain a value."""
    name = "not_contains"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        value = str(self.params["value"]).lower()
        absent = value not in ctx.response.lower()
        return CheckResult("not_contains", absent, f"'{value}' {'absent' if absent else 'present'}")


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
    """Response must contain none of the given values."""
    name = "none_of"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        values = self.params["values"]
        resp_lower = ctx.response.lower()
        found = [v for v in values if str(v).lower() in resp_lower]
        passed = len(found) == 0
        return CheckResult("none_of", passed,
                          "none present" if passed else f"found: {found}")
