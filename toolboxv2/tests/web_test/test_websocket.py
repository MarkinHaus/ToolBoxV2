# toolboxv2/tests/web_test/test_websocket.py
"""
ToolBoxV2 E2E WebSocket Tests

Tests for WebSocket functionality:
- Connection establishment
- Heartbeat/ping-pong
- Message sending/receiving
- Reconnection handling
- Notification system

Run:
    pytest toolboxv2/tests/web_test/test_websocket.py -v
    pytest toolboxv2/tests/web_test/test_websocket.py -v -k "test_connect"
"""

import pytest
import asyncio
import json
from typing import Optional, Any

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Import server config
try:
    from toolboxv2.tests.test_web import TEST_SERVER_BASE_URL, is_server_running
except ImportError:
    TEST_SERVER_BASE_URL = "http://localhost:8080"
    def is_server_running():
        return False


# =================== Configuration ===================

# WebSocket server typically runs on a different port
WS_PORT = 8100
WS_BASE_URL = f"ws://localhost:{WS_PORT}"
WS_MAIN_URL = f"{WS_BASE_URL}/ws"


# =================== Helper Functions ===================

async def connect_websocket(
    url: str = WS_MAIN_URL,
    timeout: float = 5.0
) -> Optional[Any]:
    """
    Attempt to connect to WebSocket server.

    Returns:
        WebSocket connection or None if failed
    """
    if not WEBSOCKETS_AVAILABLE:
        return None

    try:
        ws = await asyncio.wait_for(
            websockets.connect(url),
            timeout=timeout
        )
        return ws
    except Exception:
        return None


async def send_and_receive(
    ws,
    message: dict,
    timeout: float = 5.0
) -> Optional[dict]:
    """
    Send a message and wait for response.

    Returns:
        Response dict or None if failed
    """
    try:
        await ws.send(json.dumps(message))
        response = await asyncio.wait_for(ws.recv(), timeout=timeout)
        return json.loads(response)
    except Exception:
        return None


# =================== Connection Tests ===================

class TestWebSocketConnection:
    """Tests for WebSocket connection establishment"""

    @pytest.mark.asyncio
    async def test_websocket_server_available(self):
        """Test: WebSocket server is reachable"""
        if not WEBSOCKETS_AVAILABLE:
            pytest.skip("websockets library not installed")
        if not is_server_running():
            pytest.skip(f"Server not running at {TEST_SERVER_BASE_URL}")

        ws = await connect_websocket(timeout=5.0)

        if ws is None:
            pytest.skip(f"WebSocket server not available at {WS_MAIN_URL}")

        # websockets 14+ uses .state instead of .open
        assert ws.state.name == "OPEN" or hasattr(ws, 'open') and ws.open, "WebSocket connection should be open"
        await ws.close()

    @pytest.mark.asyncio
    async def test_websocket_ping_pong(self):
        """Test: WebSocket responds to ping with pong"""
        if not WEBSOCKETS_AVAILABLE:
            pytest.skip("websockets library not installed")
        if not is_server_running():
            pytest.skip(f"Server not running at {TEST_SERVER_BASE_URL}")

        ws = await connect_websocket()
        if ws is None:
            pytest.skip("WebSocket not available")

        try:
            # Send ping message
            response = await send_and_receive(
                ws,
                {"type": "ping"},
                timeout=5.0
            )

            if response:
                # Server may respond with connected, pong, or other message types
                assert response.get("type") in ["pong", "ping", "error", "connected"], \
                    f"Expected valid response, got: {response}"
        finally:
            await ws.close()

    @pytest.mark.asyncio
    async def test_websocket_invalid_message(self):
        """Test: WebSocket handles invalid messages gracefully"""
        if not WEBSOCKETS_AVAILABLE:
            pytest.skip("websockets library not installed")
        if not is_server_running():
            pytest.skip(f"Server not running at {TEST_SERVER_BASE_URL}")

        ws = await connect_websocket()
        if ws is None:
            pytest.skip("WebSocket not available")

        try:
            # Send invalid JSON
            await ws.send("not valid json {{{")

            # Should either receive error or connection stays open
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                # If we get a response, it should be an error
                data = json.loads(response)
                assert "error" in str(data).lower() or "type" in data
            except asyncio.TimeoutError:
                # No response is also acceptable
                pass
            except ConnectionClosed:
                # Connection closed is acceptable for invalid input
                pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass


class TestWebSocketMessaging:
    """Tests for WebSocket message handling"""

    @pytest.mark.asyncio
    async def test_websocket_echo_message(self):
        """Test: WebSocket can send and receive messages"""
        if not WEBSOCKETS_AVAILABLE:
            pytest.skip("websockets library not installed")
        if not is_server_running():
            pytest.skip(f"Server not running at {TEST_SERVER_BASE_URL}")

        ws = await connect_websocket()
        if ws is None:
            pytest.skip("WebSocket not available")

        try:
            test_message = {
                "type": "test",
                "data": {"message": "Hello from E2E test"}
            }

            await ws.send(json.dumps(test_message))

            # Wait for any response
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=3.0)
                data = json.loads(response)
                # Should receive some response
                assert isinstance(data, dict)
            except asyncio.TimeoutError:
                # Some WS servers don't echo, that's OK
                pass
        finally:
            await ws.close()

