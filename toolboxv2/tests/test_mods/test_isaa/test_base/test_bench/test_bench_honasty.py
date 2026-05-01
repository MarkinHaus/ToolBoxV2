"""Unit tests for bench.validators.honesty — tool call and honesty validators."""

import unittest

from toolboxv2.mods.isaa.base.bench.core import Task, TaskContext
from toolboxv2.mods.isaa.base.bench.validators import create_validator


def make_ctx(response="OK", tool_calls=None, prompt="do something", **kw):
    t = Task(id="test", complexity="tutorial", modality=["text"], prompt=prompt, checks=[])
    return TaskContext(
        task=t, prompt=prompt, response=response,
        tool_calls=tool_calls or [], **kw,
    )


SAMPLE_TOOLS = [
    {"name": "search", "args": {"query": "weather berlin"}, "result": "sunny, 18°C"},
    {"name": "vfs_read", "args": {"filename": "data.txt"}, "result": "project alpha status: active"},
]


class TestToolCalledValidator(unittest.IsolatedAsyncioTestCase):

    async def test_tool_was_called(self):
        v = create_validator({"type": "tool_called", "name": "search"})
        r = await v.validate(make_ctx(tool_calls=SAMPLE_TOOLS))
        self.assertTrue(r.passed)

    async def test_tool_not_called(self):
        v = create_validator({"type": "tool_called", "name": "delete"})
        r = await v.validate(make_ctx(tool_calls=SAMPLE_TOOLS))
        self.assertFalse(r.passed)

    async def test_empty_tool_calls(self):
        v = create_validator({"type": "tool_called", "name": "search"})
        r = await v.validate(make_ctx(tool_calls=[]))
        self.assertFalse(r.passed)


class TestToolNotCalledValidator(unittest.IsolatedAsyncioTestCase):

    async def test_tool_absent_passes(self):
        v = create_validator({"type": "tool_not_called", "name": "delete"})
        r = await v.validate(make_ctx(tool_calls=SAMPLE_TOOLS))
        self.assertTrue(r.passed)

    async def test_tool_present_fails(self):
        v = create_validator({"type": "tool_not_called", "name": "search"})
        r = await v.validate(make_ctx(tool_calls=SAMPLE_TOOLS))
        self.assertFalse(r.passed)


class TestToolCalledNValidator(unittest.IsolatedAsyncioTestCase):

    async def test_exact_count(self):
        tools = [
            {"name": "search", "args": {}, "result": "a"},
            {"name": "search", "args": {}, "result": "b"},
            {"name": "read", "args": {}, "result": "c"},
        ]
        v = create_validator({"type": "tool_called_n", "name": "search", "count": 2})
        r = await v.validate(make_ctx(tool_calls=tools))
        self.assertTrue(r.passed)

    async def test_wrong_count(self):
        tools = [{"name": "search", "args": {}, "result": "a"}]
        v = create_validator({"type": "tool_called_n", "name": "search", "count": 3})
        r = await v.validate(make_ctx(tool_calls=tools))
        self.assertFalse(r.passed)

    async def test_zero_count(self):
        v = create_validator({"type": "tool_called_n", "name": "search", "count": 0})
        r = await v.validate(make_ctx(tool_calls=[]))
        self.assertTrue(r.passed)


class TestToolCalledWithValidator(unittest.IsolatedAsyncioTestCase):

    async def test_matching_arg(self):
        v = create_validator({
            "type": "tool_called_with",
            "name": "search", "arg": "query", "value": "weather"
        })
        r = await v.validate(make_ctx(tool_calls=SAMPLE_TOOLS))
        self.assertTrue(r.passed)

    async def test_non_matching_arg(self):
        v = create_validator({
            "type": "tool_called_with",
            "name": "search", "arg": "query", "value": "stocks"
        })
        r = await v.validate(make_ctx(tool_calls=SAMPLE_TOOLS))
        self.assertFalse(r.passed)

    async def test_tool_not_called(self):
        v = create_validator({
            "type": "tool_called_with",
            "name": "delete", "arg": "file", "value": "x"
        })
        r = await v.validate(make_ctx(tool_calls=SAMPLE_TOOLS))
        self.assertFalse(r.passed)

    async def test_arg_key_missing(self):
        v = create_validator({
            "type": "tool_called_with",
            "name": "search", "arg": "nonexistent", "value": "x"
        })
        r = await v.validate(make_ctx(tool_calls=SAMPLE_TOOLS))
        self.assertFalse(r.passed)


class TestToolResultInResponseValidator(unittest.IsolatedAsyncioTestCase):

    async def test_result_appears(self):
        v = create_validator({"type": "tool_result_in_response", "name": "search"})
        r = await v.validate(make_ctx(
            response="The weather is sunny in Berlin.",
            tool_calls=SAMPLE_TOOLS,
        ))
        self.assertTrue(r.passed)

    async def test_result_absent(self):
        v = create_validator({"type": "tool_result_in_response", "name": "search"})
        r = await v.validate(make_ctx(
            response="I have no idea about anything.",
            tool_calls=SAMPLE_TOOLS,
        ))
        self.assertFalse(r.passed)

    async def test_tool_not_called(self):
        v = create_validator({"type": "tool_result_in_response", "name": "delete"})
        r = await v.validate(make_ctx(tool_calls=SAMPLE_TOOLS))
        self.assertFalse(r.passed)


class TestToolOrderValidator(unittest.IsolatedAsyncioTestCase):

    async def test_correct_order(self):
        tools = [
            {"name": "list", "args": {}, "result": ""},
            {"name": "open", "args": {}, "result": ""},
            {"name": "read", "args": {}, "result": ""},
        ]
        v = create_validator({"type": "tool_order", "names": ["list", "open", "read"]})
        r = await v.validate(make_ctx(tool_calls=tools))
        self.assertTrue(r.passed)

    async def test_wrong_order(self):
        tools = [
            {"name": "read", "args": {}, "result": ""},
            {"name": "list", "args": {}, "result": ""},
        ]
        v = create_validator({"type": "tool_order", "names": ["list", "read"]})
        r = await v.validate(make_ctx(tool_calls=tools))
        self.assertFalse(r.passed)

    async def test_subsequence_with_extras(self):
        tools = [
            {"name": "list", "args": {}, "result": ""},
            {"name": "log", "args": {}, "result": ""},
            {"name": "read", "args": {}, "result": ""},
        ]
        v = create_validator({"type": "tool_order", "names": ["list", "read"]})
        r = await v.validate(make_ctx(tool_calls=tools))
        self.assertTrue(r.passed, "should match subsequence with extras in between")

    async def test_missing_tool_in_order(self):
        tools = [{"name": "list", "args": {}, "result": ""}]
        v = create_validator({"type": "tool_order", "names": ["list", "read"]})
        r = await v.validate(make_ctx(tool_calls=tools))
        self.assertFalse(r.passed)


class TestNoHallucinationValidator(unittest.IsolatedAsyncioTestCase):

    async def test_honest_response(self):
        tools = [{"name": "search", "args": {}, "result": "temperature is 18 degrees"}]
        v = create_validator({"type": "no_hallucination"})
        r = await v.validate(make_ctx(
            response="The temperature is 18 degrees.",
            tool_calls=tools,
        ))
        self.assertTrue(r.passed)

    async def test_hallucinated_numbers(self):
        tools = [{"name": "search", "args": {}, "result": "weather is sunny"}]
        v = create_validator({"type": "no_hallucination"})
        r = await v.validate(make_ctx(
            response="Weather is sunny. Temperature is 25°C and humidity 60%.",
            tool_calls=tools,
        ))
        self.assertFalse(r.passed, "25 and 60 are not in tool results")

    async def test_no_facts_is_honest(self):
        v = create_validator({"type": "no_hallucination"})
        r = await v.validate(make_ctx(
            response="I can help you with that. Let me know what you need.",
            tool_calls=[],
        ))
        self.assertTrue(r.passed)

    async def test_facts_from_prompt_are_ok(self):
        v = create_validator({"type": "no_hallucination"})
        r = await v.validate(make_ctx(
            prompt="The file has 42 entries.",
            response="You mentioned the file has 42 entries.",
            tool_calls=[],
        ))
        self.assertTrue(r.passed, "facts from prompt should not be hallucination")


class TestAdmitsUncertaintyValidator(unittest.IsolatedAsyncioTestCase):

    async def test_admits_english(self):
        v = create_validator({"type": "admits_uncertainty"})
        r = await v.validate(make_ctx(response="I don't know the answer to that."))
        self.assertTrue(r.passed)

    async def test_admits_german(self):
        v = create_validator({"type": "admits_uncertainty"})
        r = await v.validate(make_ctx(response="Ich habe keine Information dazu."))
        self.assertTrue(r.passed)

    async def test_confident_response_fails(self):
        v = create_validator({"type": "admits_uncertainty"})
        r = await v.validate(make_ctx(response="The answer is definitely X."))
        self.assertFalse(r.passed)


class TestNoUncertaintyValidator(unittest.IsolatedAsyncioTestCase):

    async def test_confident_passes(self):
        v = create_validator({"type": "no_uncertainty"})
        r = await v.validate(make_ctx(response="The capital of France is Paris."))
        self.assertTrue(r.passed)

    async def test_uncertain_fails(self):
        v = create_validator({"type": "no_uncertainty"})
        r = await v.validate(make_ctx(response="I'm not sure but maybe Paris."))
        self.assertFalse(r.passed)


class TestResponseUsesToolDataValidator(unittest.IsolatedAsyncioTestCase):

    async def test_uses_data(self):
        tools = [{"name": "db", "args": {}, "result": "project alpha status active"}]
        v = create_validator({"type": "response_uses_tool_data"})
        r = await v.validate(make_ctx(
            response="Project alpha is currently active.",
            tool_calls=tools,
        ))
        self.assertTrue(r.passed)

    async def test_ignores_data(self):
        tools = [{"name": "db", "args": {}, "result": "project alpha status active"}]
        v = create_validator({"type": "response_uses_tool_data"})
        r = await v.validate(make_ctx(
            response="I completed the task successfully.",
            tool_calls=tools,
        ))
        self.assertFalse(r.passed)

    async def test_no_tool_calls(self):
        v = create_validator({"type": "response_uses_tool_data"})
        r = await v.validate(make_ctx(tool_calls=[]))
        self.assertFalse(r.passed)


if __name__ == "__main__":
    unittest.main()
