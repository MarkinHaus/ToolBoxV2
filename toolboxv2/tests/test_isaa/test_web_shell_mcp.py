import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from toolboxv2.mods.isaa.base.patch.web_shell_tool import make_web_shell

@pytest.fixture
def mock_mcp_manager():
    manager = MagicMock()
    session = AsyncMock()

    # Mock tool call results
    async def mock_call_tool(name, args):
        res = MagicMock()
        res.is_error = False
        if name == "evaluate":
            if "document.title" in args.get("expression", ""):
                content = MagicMock()
                content.text = json.dumps("Mock Page Title")
                res.content = [content]
            elif "document.body.innerText" in args.get("expression", ""):
                content = MagicMock()
                content.text = json.dumps("Mock body content")
                res.content = [content]
            else:
                content = MagicMock()
                content.text = json.dumps(True)
                res.content = [content]
        elif name == "capture_screenshot":
            content = MagicMock()
            content.text = "dGVzdCBzY3JlZW5zaG90" # "test screenshot" in base64
            res.content = [content]
        else:
            res.content = []
        return res

    session.call_tool.side_effect = mock_call_tool
    manager.get_session = AsyncMock(return_value=session)
    return manager

import json

@pytest.mark.asyncio
async def test_make_web_shell_mcp_toggle(mock_mcp_manager):
    session = MagicMock()
    session.vfs = MagicMock()
    session.vfs.read.return_value = {"success": False}
    session.skills = MagicMock()
    session.skills._mcp_session_manager = mock_mcp_manager
    session._web_config = {
        "engine": "mcp",
        "mcp_server": "test-server"
    }

    web_shell = make_web_shell(session)
    # Since it's a closure, we might need to check side effects on mock_mcp_manager

    # Try calling a command
    result = await web_shell(None, "goto https://example.com")
    assert "example.com" in result["stdout"]

    # Verify manager was called
    mock_mcp_manager.get_session.assert_called()
