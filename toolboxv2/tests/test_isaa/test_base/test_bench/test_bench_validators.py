"""Unit tests for bench.validators — all built-in validators."""

import unittest

from toolboxv2.mods.isaa.base.bench.core import Check, Task, TaskContext, CheckResult
from toolboxv2.mods.isaa.base.bench.validators import create_validator, get_validator, list_validators, _REGISTRY


def make_ctx(response="The answer is 4.", tool_calls=None, execution_time=0.0, **kw):
    """Factory for TaskContext with sensible defaults."""
    t = Task(id="test", complexity="tutorial", modality=["text"], prompt="q", checks=[])
    return TaskContext(
        task=t,
        prompt="q",
        response=response,
        tool_calls=tool_calls or [],
        execution_time=execution_time,
        **kw,
    )


class TestRegistry(unittest.TestCase):

    def test_list_validators_includes_builtins(self):
        names = list_validators()
        for expected in ["contains", "not_contains", "regex", "equals",
                         "json_valid", "any_of", "all_of", "none_of",
                         "char_count_gte", "char_count_lte", "max_tokens",
                         "tool_calls_lte", "tool_calls_gte",
                         "file_exists", "latency_lte", "json_has_key"]:
            self.assertIn(expected, names, msg=f"Missing built-in: {expected}")

    def test_get_validator_unknown_raises_keyerror(self):
        with self.assertRaises(KeyError):
            get_validator("nonexistent_validator_xyz")

    def test_create_validator_from_dict(self):
        v = create_validator({"type": "contains", "value": "hello"})
        self.assertEqual(v.name, "contains")
        self.assertEqual(v.params["value"], "hello")


class TestContainsValidator(unittest.IsolatedAsyncioTestCase):

    async def test_value_present_passes(self):
        v = create_validator({"type": "contains", "value": "4"})
        r = await v.validate(make_ctx("The answer is 4."))
        self.assertTrue(r.passed)

    async def test_value_absent_fails(self):
        v = create_validator({"type": "contains", "value": "5"})
        r = await v.validate(make_ctx("The answer is 4."))
        self.assertFalse(r.passed)

    async def test_case_insensitive(self):
        v = create_validator({"type": "contains", "value": "ANSWER"})
        r = await v.validate(make_ctx("The answer is 4."))
        self.assertTrue(r.passed)

    async def test_empty_response(self):
        v = create_validator({"type": "contains", "value": "x"})
        r = await v.validate(make_ctx(""))
        self.assertFalse(r.passed)


class TestNotContainsValidator(unittest.IsolatedAsyncioTestCase):

    async def test_value_absent_passes(self):
        v = create_validator({"type": "not_contains", "value": "wrong"})
        r = await v.validate(make_ctx("The answer is 4."))
        self.assertTrue(r.passed)

    async def test_value_present_fails(self):
        v = create_validator({"type": "not_contains", "value": "answer"})
        r = await v.validate(make_ctx("The answer is 4."))
        self.assertFalse(r.passed)


class TestEqualsValidator(unittest.IsolatedAsyncioTestCase):

    async def test_exact_match_passes(self):
        v = create_validator({"type": "equals", "value": "42"})
        r = await v.validate(make_ctx("42"))
        self.assertTrue(r.passed)

    async def test_whitespace_stripped(self):
        v = create_validator({"type": "equals", "value": "42"})
        r = await v.validate(make_ctx("  42  \n"))
        self.assertTrue(r.passed)

    async def test_case_insensitive(self):
        v = create_validator({"type": "equals", "value": "Yes"})
        r = await v.validate(make_ctx("yes"))
        self.assertTrue(r.passed)

    async def test_mismatch_fails(self):
        v = create_validator({"type": "equals", "value": "42"})
        r = await v.validate(make_ctx("43"))
        self.assertFalse(r.passed)


class TestRegexValidator(unittest.IsolatedAsyncioTestCase):

    async def test_pattern_matches(self):
        v = create_validator({"type": "regex", "pattern": r"\d+"})
        r = await v.validate(make_ctx("The answer is 4."))
        self.assertTrue(r.passed)

    async def test_pattern_no_match(self):
        v = create_validator({"type": "regex", "pattern": r"^\d+$"})
        r = await v.validate(make_ctx("The answer is 4."))
        self.assertFalse(r.passed)

    async def test_case_insensitive_default(self):
        v = create_validator({"type": "regex", "pattern": r"ANSWER"})
        r = await v.validate(make_ctx("the answer"))
        self.assertTrue(r.passed)


class TestCharCountValidators(unittest.IsolatedAsyncioTestCase):

    async def test_gte_passes_when_above(self):
        v = create_validator({"type": "char_count_gte", "value": 5})
        r = await v.validate(make_ctx("hello world"))
        self.assertTrue(r.passed)

    async def test_gte_fails_when_below(self):
        v = create_validator({"type": "char_count_gte", "value": 100})
        r = await v.validate(make_ctx("short"))
        self.assertFalse(r.passed)

    async def test_gte_exact_boundary_passes(self):
        v = create_validator({"type": "char_count_gte", "value": 5})
        r = await v.validate(make_ctx("12345"))
        self.assertTrue(r.passed)

    async def test_lte_passes_when_below(self):
        v = create_validator({"type": "char_count_lte", "value": 100})
        r = await v.validate(make_ctx("short"))
        self.assertTrue(r.passed)

    async def test_lte_fails_when_above(self):
        v = create_validator({"type": "char_count_lte", "value": 3})
        r = await v.validate(make_ctx("too long"))
        self.assertFalse(r.passed)

    async def test_empty_response_gte_zero(self):
        v = create_validator({"type": "char_count_gte", "value": 0})
        r = await v.validate(make_ctx(""))
        self.assertTrue(r.passed)


class TestMaxTokensValidator(unittest.IsolatedAsyncioTestCase):

    async def test_within_limit_passes(self):
        v = create_validator({"type": "max_tokens", "value": 10})
        r = await v.validate(make_ctx("one two three"))
        self.assertTrue(r.passed)

    async def test_exceeds_limit_fails(self):
        v = create_validator({"type": "max_tokens", "value": 2})
        r = await v.validate(make_ctx("one two three four five"))
        self.assertFalse(r.passed)


class TestJsonValidators(unittest.IsolatedAsyncioTestCase):

    async def test_valid_json_passes(self):
        v = create_validator({"type": "json_valid"})
        r = await v.validate(make_ctx('{"key": "value"}'))
        self.assertTrue(r.passed)

    async def test_invalid_json_fails(self):
        v = create_validator({"type": "json_valid"})
        r = await v.validate(make_ctx("not json at all"))
        self.assertFalse(r.passed)

    async def test_empty_string_fails(self):
        v = create_validator({"type": "json_valid"})
        r = await v.validate(make_ctx(""))
        self.assertFalse(r.passed)

    async def test_json_has_key_present(self):
        v = create_validator({"type": "json_has_key", "key": "name"})
        r = await v.validate(make_ctx('{"name": "Alice", "age": 30}'))
        self.assertTrue(r.passed)

    async def test_json_has_key_absent(self):
        v = create_validator({"type": "json_has_key", "key": "email"})
        r = await v.validate(make_ctx('{"name": "Alice"}'))
        self.assertFalse(r.passed)

    async def test_json_has_key_invalid_json(self):
        v = create_validator({"type": "json_has_key", "key": "x"})
        r = await v.validate(make_ctx("not json"))
        self.assertFalse(r.passed)

    async def test_json_has_key_array_response(self):
        v = create_validator({"type": "json_has_key", "key": "x"})
        r = await v.validate(make_ctx("[1, 2, 3]"))
        self.assertFalse(r.passed)


class TestToolCallsValidators(unittest.IsolatedAsyncioTestCase):

    async def test_lte_within_limit(self):
        v = create_validator({"type": "tool_calls_lte", "value": 3})
        r = await v.validate(make_ctx(tool_calls=[{"name": "a"}, {"name": "b"}]))
        self.assertTrue(r.passed)

    async def test_lte_exceeds_limit(self):
        v = create_validator({"type": "tool_calls_lte", "value": 1})
        r = await v.validate(make_ctx(tool_calls=[{"name": "a"}, {"name": "b"}]))
        self.assertFalse(r.passed)

    async def test_gte_meets_minimum(self):
        v = create_validator({"type": "tool_calls_gte", "value": 2})
        r = await v.validate(make_ctx(tool_calls=[{"name": "a"}, {"name": "b"}]))
        self.assertTrue(r.passed)

    async def test_gte_below_minimum(self):
        v = create_validator({"type": "tool_calls_gte", "value": 5})
        r = await v.validate(make_ctx(tool_calls=[{"name": "a"}]))
        self.assertFalse(r.passed)

    async def test_empty_tool_calls_lte_zero(self):
        v = create_validator({"type": "tool_calls_lte", "value": 0})
        r = await v.validate(make_ctx(tool_calls=[]))
        self.assertTrue(r.passed)


class TestLatencyValidator(unittest.IsolatedAsyncioTestCase):

    async def test_within_limit_passes(self):
        v = create_validator({"type": "latency_lte", "value": 5.0})
        r = await v.validate(make_ctx(execution_time=2.3))
        self.assertTrue(r.passed)

    async def test_exceeds_limit_fails(self):
        v = create_validator({"type": "latency_lte", "value": 1.0})
        r = await v.validate(make_ctx(execution_time=2.3))
        self.assertFalse(r.passed)


class TestCollectionValidators(unittest.IsolatedAsyncioTestCase):

    async def test_any_of_one_match(self):
        v = create_validator({"type": "any_of", "values": ["cat", "dog", "fish"]})
        r = await v.validate(make_ctx("I have a dog"))
        self.assertTrue(r.passed)

    async def test_any_of_no_match(self):
        v = create_validator({"type": "any_of", "values": ["cat", "dog", "fish"]})
        r = await v.validate(make_ctx("I have a bird"))
        self.assertFalse(r.passed)

    async def test_all_of_all_present(self):
        v = create_validator({"type": "all_of", "values": ["cat", "dog"]})
        r = await v.validate(make_ctx("I have a cat and a dog"))
        self.assertTrue(r.passed)

    async def test_all_of_one_missing(self):
        v = create_validator({"type": "all_of", "values": ["cat", "dog"]})
        r = await v.validate(make_ctx("I have a cat"))
        self.assertFalse(r.passed)

    async def test_none_of_none_present(self):
        v = create_validator({"type": "none_of", "values": ["error", "fail"]})
        r = await v.validate(make_ctx("Everything is fine"))
        self.assertTrue(r.passed)

    async def test_none_of_one_present(self):
        v = create_validator({"type": "none_of", "values": ["error", "fail"]})
        r = await v.validate(make_ctx("An error occurred"))
        self.assertFalse(r.passed)

    async def test_any_of_case_insensitive(self):
        v = create_validator({"type": "any_of", "values": ["Munich"]})
        r = await v.validate(make_ctx("I live in munich"))
        self.assertTrue(r.passed)


if __name__ == "__main__":
    unittest.main()
