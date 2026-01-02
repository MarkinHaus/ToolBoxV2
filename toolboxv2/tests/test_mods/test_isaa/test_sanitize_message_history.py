#!/usr/bin/env python3
"""
Tests for sanitize_message_history function in FlowAgent.

This function prevents LiteLLM errors like:
"Missing corresponding tool call for tool response message"

Run with:
    pytest toolboxv2/tests/test_mods/test_isaa/test_sanitize_message_history.py -v
"""

import unittest
from unittest.mock import MagicMock, patch
import sys


class TestSanitizeMessageHistory(unittest.TestCase):
    """Tests for the sanitize_message_history method"""

    def setUp(self):
        """Create a mock agent with the sanitize_message_history method"""
        # Import the actual method from agent.py
        from toolboxv2.mods.isaa.base.Agent.agent import FlowAgent, AgentModelData

        # Create a minimal agent instance
        amd = AgentModelData(name="TestAgent")
        self.agent = FlowAgent(amd)

    def test_empty_messages(self):
        """Test with empty message list"""
        result = self.agent.sanitize_message_history([])
        self.assertEqual(result, [])

    def test_none_messages(self):
        """Test with None - should return None"""
        result = self.agent.sanitize_message_history(None)
        self.assertIsNone(result)

    def test_simple_conversation_no_tools(self):
        """Test simple conversation without tool calls passes through unchanged"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"},
        ]
        result = self.agent.sanitize_message_history(messages)
        self.assertEqual(result, messages)

    def test_valid_tool_call_pair(self):
        """Test valid tool call and response pair is preserved"""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_123", "function": {"name": "get_weather", "arguments": "{}"}}
                ]
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 72°F"},
            {"role": "assistant", "content": "The weather is sunny and 72°F!"},
        ]
        result = self.agent.sanitize_message_history(messages)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[2]["role"], "tool")

    def test_orphaned_tool_response_removed(self):
        """Test orphaned tool response (no matching tool_call) is removed"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            # Orphaned tool response - no preceding tool_call
            {"role": "tool", "tool_call_id": "call_orphan", "content": "Some result"},
            {"role": "user", "content": "Thanks"},
        ]
        result = self.agent.sanitize_message_history(messages)
        # Should have 3 messages (orphaned tool removed)
        self.assertEqual(len(result), 3)
        self.assertNotIn("tool", [m["role"] for m in result])

    def test_unmatched_tool_calls_cleaned(self):
        """Test assistant message with unmatched tool_calls is cleaned"""
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": "Let me try that.",
                "tool_calls": [
                    {"id": "call_never_responded", "function": {"name": "some_tool"}}
                ]
            },
            # No tool response follows - conversation continues
            {"role": "user", "content": "Never mind"},
            {"role": "assistant", "content": "Okay!"},
        ]
        result = self.agent.sanitize_message_history(messages)
        # The assistant message with unmatched tool_calls should have tool_calls removed
        # but content preserved
        assistant_with_tools = result[1]
        self.assertNotIn("tool_calls", assistant_with_tools)
        self.assertEqual(assistant_with_tools["content"], "Let me try that.")

    def test_multiple_tool_calls_partial_response(self):
        """Test multiple tool calls where only some get responses"""
        messages = [
            {"role": "user", "content": "Get weather and news"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_weather", "function": {"name": "get_weather"}},
                    {"id": "call_news", "function": {"name": "get_news"}},
                ]
            },
            # Only weather responded
            {"role": "tool", "tool_call_id": "call_weather", "content": "Sunny"},
            {"role": "assistant", "content": "Weather is sunny, but news failed."},
        ]
        result = self.agent.sanitize_message_history(messages)
        # Weather tool response should be kept
        tool_messages = [m for m in result if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_weather")

    def test_tool_call_id_in_different_formats(self):
        """Test tool_call_id can be in 'id' or 'tool_call_id' field"""
        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"tool_call_id": "alt_format_123", "function": {"name": "test"}}
                ]
            },
            {"role": "tool", "tool_call_id": "alt_format_123", "content": "Result"},
        ]
        result = self.agent.sanitize_message_history(messages)
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()

