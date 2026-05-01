import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import json
from pydantic import BaseModel

# Implemented Modules (angepasst an deinen Workspace)
from toolboxv2.mods.isaa.base.Agent.flow_agent import (
    FlowAgent,
    parse_media_from_query,
    _is_media_error,
    _extract_failed_media_type,
    _remove_media_by_type
)
from toolboxv2.mods.isaa.base.Agent.types import AgentModelData


# --- Test Data & Dummies ---

class DummyResponseSchema(BaseModel):
    name: str
    age: int


def create_mock_litellm_response(content="", tool_calls=None, finish_reason="stop", usage=None):
    """Erstellt einen sauberen Fake für litellm Responses (Netzwerk-Boundary)."""
    res = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    choice.message.role = "assistant"
    choice.finish_reason = finish_reason

    res.choices = [choice]

    res.usage = MagicMock()
    if usage:
        res.usage.prompt_tokens = usage.get("prompt_tokens", 10)
        res.usage.completion_tokens = usage.get("completion_tokens", 10)
        res.usage.total_tokens = res.usage.prompt_tokens + res.usage.completion_tokens
    else:
        res.usage.prompt_tokens = 0
        res.usage.completion_tokens = 0
        res.usage.total_tokens = 0

    return res


# --- 1. PURE LOGIC TESTS (No I/O, No Mocks needed) ---

class TestMediaParser(unittest.TestCase):

    def test_parse_media_from_query_valid_image_returns_cleaned_query_and_vision_format(self):
        # Arrange
        query = "Analyze[media:https://example.com/image.jpg] this."

        # Act
        cleaned, media_list = parse_media_from_query(query)

        # Assert
        self.assertEqual(cleaned, "Analyze this.")
        self.assertEqual(len(media_list), 1)
        self.assertEqual(media_list[0]["type"], "image_url")
        self.assertEqual(media_list[0]["image_url"]["url"], "https://example.com/image.jpg")
        self.assertEqual(media_list[0]["image_url"]["format"], "image/jpeg")

    def test_parse_media_from_query_unknown_extension_defaults_to_image_url(self):
        # Arrange
        query = "Look at [media:file.unknown]"

        # Act
        _, media_list = parse_media_from_query(query)

        # Assert
        self.assertEqual(len(media_list), 1)
        self.assertEqual(media_list[0]["type"], "image_url")
        self.assertEqual(media_list[0]["image_url"]["url"], "file.unknown")
        self.assertNotIn("format", media_list[0]["image_url"])

    def test_is_media_error_matches_known_patterns_returns_true(self):
        # Arrange
        err = Exception("Unsupported image type: image/svg")

        # Act
        result = _is_media_error(err)

        # Assert
        self.assertTrue(result)

    def test_extract_failed_media_type_finds_pdf_returns_pdf(self):
        # Arrange
        err = Exception("PDF not supported by this model")

        # Act
        media_type = _extract_failed_media_type(err)

        # Assert
        self.assertEqual(media_type, "pdf")

    def test_remove_media_by_type_removes_specified_target_and_keeps_others(self):
        # Arrange
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze"},
                {"type": "image_url", "image_url": {"url": "doc.pdf"}},
                {"type": "image_url", "image_url": {"url": "pic.jpg"}}
            ]
        }]

        # Act
        cleaned, removed = _remove_media_by_type(messages, types_to_remove=["pdf"])

        # Assert
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["media_type"], "pdf")
        self.assertEqual(len(cleaned[0]["content"]), 2)  # Text + pic.jpg remaining


# --- 2. AGENT BEHAVIOR & LLM TESTS (Async, heavily isolated) ---

class TestFlowAgentLLM(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.amd = AgentModelData(
            name="TestAgent",
            fast_llm_model="test/fast",
            complex_llm_model="test/complex",
            system_message="SysMsg",
            vfs_max_window_lines=100
        )

        # Patching System Boundaries directly during init to prevent side-effects
        self.patcher_audit = patch("toolboxv2.mods.isaa.base.Agent.flow_agent.get_app")
        self.mock_get_app = self.patcher_audit.start()

        self.agent = FlowAgent(self.amd, auto_load_checkpoint=False)
        self.agent.llm_handler.completion_with_rate_limiting = AsyncMock()
        self.agent.save_supports_vision = AsyncMock(return_value=False)  # Skip vision check overhead

    def tearDown(self):
        self.patcher_audit.stop()

    async def test_a_run_llm_completion_standard_text_returns_string(self):
        # Arrange
        mock_response = create_mock_litellm_response(content="Hello World")
        self.agent.llm_handler.completion_with_rate_limiting.return_value = mock_response

        # Act
        result = await self.agent.a_run_llm_completion(
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            with_context=False
        )

        # Assert
        self.assertEqual(result, "Hello World")
        self.agent.llm_handler.completion_with_rate_limiting.assert_called_once()

    async def test_a_run_llm_completion_max_tokens_triggers_auto_resume(self):
        # Arrange
        # First call hits limit, second finishes normally
        res_part1 = create_mock_litellm_response(content="Part 1 ", finish_reason="length")
        res_part2 = create_mock_litellm_response(content="Part 2", finish_reason="stop")

        self.agent.llm_handler.completion_with_rate_limiting.side_effect = [res_part1, res_part2]

        # Act
        result = await self.agent.a_run_llm_completion(
            messages=[{"role": "user", "content": "Write story"}],
            stream=False,
            with_context=False
        )

        # Assert
        self.assertEqual(result, "Part 1 Part 2")
        self.assertEqual(self.agent.llm_handler.completion_with_rate_limiting.call_count, 2)

    async def test_a_run_llm_completion_media_error_triggers_retry_without_media(self):
        # Arrange
        # Mock raises specific media error first, succeeds on second try
        media_error = Exception("unsupported image type")
        success_res = create_mock_litellm_response(content="Done without image")

        self.agent.llm_handler.completion_with_rate_limiting.side_effect = [media_error, success_res]

        # Act
        result = await self.agent.a_run_llm_completion(
            messages=[{"role": "user", "content": "Analyze [media:x.pdf]"}],
            stream=False,
            with_context=False
        )

        # Assert
        self.assertEqual(result, "Done without image")
        self.assertEqual(self.agent.llm_handler.completion_with_rate_limiting.call_count, 2)

    async def test_a_format_class_invalid_json_fallback_extracts_yaml(self):
        # Arrange
        # Model returns messy YAML instead of strict JSON formatting
        yaml_content = "```yaml\nname: John\nage: 30\n```"

        # First call (JSON attempt) fails, second call (YAML fallback) succeeds
        self.agent.a_run_llm_completion = AsyncMock(side_effect=[
            json.JSONDecodeError("Expecting value", "", 0),
            yaml_content
        ])

        # Assert
        with self.assertRaises(json.decoder.JSONDecodeError):
            # Act
            result = await self.agent.a_format_class(
                pydantic_model=DummyResponseSchema,
                prompt="Extract user",
                auto_context=False
            )


# --- 3. TOOL MANAGER & PERMISSION TESTS ---

class TestFlowAgentTools(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.amd = AgentModelData(name="TestAgent", fast_llm_model="test", complex_llm_model="test")
        self.patcher_audit = patch("toolboxv2.mods.isaa.base.Agent.flow_agent.get_app")
        self.patcher_audit.start()
        self.agent = FlowAgent(self.amd, auto_load_checkpoint=False)

    def tearDown(self):
        self.patcher_audit.stop()

    def test_add_tool_registers_function_and_applies_contracts(self):
        # Arrange
        def dummy_tool(x: int): return x * 2

        # Act
        self.agent.add_tool(
            tool_func=dummy_tool,
            name="multiply",
            expected_type=int
        )

        # Assert
        tool_entry = self.agent.tool_manager.get("multiply")
        self.assertIsNotNone(tool_entry)
        self.assertEqual(tool_entry.result_contract["expected_type"], int)

    async def test_arun_function_tool_not_allowed_raises_permission_error(self):
        # Arrange
        def restricted_tool(): return "secret"

        self.agent.add_tool(restricted_tool, name="secret_tool")

        # Mock Session to simulate blocked permissions
        mock_session = MagicMock()
        mock_session.is_tool_allowed.return_value = False

        self.agent.session_manager.get = MagicMock(return_value=mock_session)
        self.agent.active_session = "test_session"

        # Act & Assert
        with self.assertRaisesRegex(PermissionError, "restricted in session"):
            await self.agent.arun_function("secret_tool")
