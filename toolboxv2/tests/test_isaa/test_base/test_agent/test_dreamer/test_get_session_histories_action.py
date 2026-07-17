"""
Tests the dream_act action 'get_session_histories' end-to-end through the
DreamerToolHandler dispatcher (handle_act), with a mocked SessionManager.

Verifies:
  - action is registered (not 'unknown')
  - provider wired -> full histories returned (all roles, no truncation)
  - missing provider -> clean error, no crash
"""

import json

from toolboxv2.mods.isaa.base.dreamer.tool_handler import DreamerToolHandler


class DummyChat:
    def __init__(self, msgs):
        self.history = msgs


class DummySession:
    def __init__(self, chat):
        self._chat_session = chat


class DummySM:
    def __init__(self, sessions):
        self.sessions = sessions


def _msg(role, content):
    return {"role": role, "content": content}


def _handler(provider):
    return DreamerToolHandler(
        skills={}, rules={}, patterns=[], personas={},
        session_manager_provider=provider,
    )


def test_action_registered():
    assert "get_session_histories" in DreamerToolHandler._VALID_ACTIONS


def test_returns_full_histories():
    long = "y" * 3000
    sm = DummySM({
        "s1": DummySession(DummyChat([_msg("user", long), _msg("assistant", "ok")])),
        "s2": DummySession(DummyChat([_msg("tool", "t")])),
    })
    h = _handler(lambda: sm)
    out = json.loads(h.handle_act("get_session_histories", {}))
    assert out["success"] is True
    assert out["session_count"] == 2
    assert out["total_messages"] == 3
    # full content preserved, all roles present
    assert out["histories"]["s1"][0]["content"] == long
    assert out["histories"]["s1"][1]["role"] == "assistant"
    assert out["histories"]["s2"][0]["role"] == "tool"


def test_no_provider_clean_error():
    h = _handler(None)
    out = json.loads(h.handle_act("get_session_histories", {}))
    assert out["success"] is False
    assert "provider" in out["error"]


def test_provider_returns_none():
    h = _handler(lambda: None)
    out = json.loads(h.handle_act("get_session_histories", {}))
    assert out["success"] is False


def test_unknown_action_still_guarded():
    h = _handler(lambda: DummySM({}))
    out = json.loads(h.handle_act("bogus_action", {}))
    assert out["success"] is False
    assert "unknown action" in out["error"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
