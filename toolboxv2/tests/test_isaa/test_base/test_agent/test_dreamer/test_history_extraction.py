"""
Standalone test: proves full-session-history extraction from the parent
SessionManager (in-memory) works BEFORE wiring it into dream_act.

Mocks the exact runtime shape:
  session_manager.sessions[sid]._chat_session.history -> [{"role","content",...}]
No agent boot, no disk, no async.

The Dreamer gets the FULL context: all roles, no truncation (history is
already pruned/compressed by the agent).
"""

from toolboxv2.mods.isaa.base.dreamer.history_utils import extract_session_histories


class DummyChat:
    def __init__(self, msgs):
        self.history = msgs


class DummySession:
    def __init__(self, chat=None):
        self._chat_session = chat


class DummySM:
    def __init__(self, sessions):
        self.sessions = sessions


def _msg(role, content):
    return {"role": role, "content": content}


def test_empty_manager():
    assert extract_session_histories(DummySM({})) == {}


def test_no_sessions_attr():
    class Bare:
        pass
    assert extract_session_histories(Bare()) == {}


def test_all_roles_no_truncation():
    long = "x" * 5000
    msgs = [
        _msg("assistant", "hi"),
        _msg("user", long),
        _msg("tool", "tool out"),
        _msg("system", "sys"),
    ]
    out = extract_session_histories(DummySM({"s1": DummySession(DummyChat(msgs))}))
    # every role kept, full content preserved
    assert out["s1"] == msgs
    assert out["s1"][1]["content"] == long


def test_cap_keeps_newest():
    msgs = [_msg("user", f"q{i}") for i in range(5)]
    out = extract_session_histories(DummySM({"s1": DummySession(DummyChat(msgs))}),
                                    max_per_session=2)
    assert out["s1"] == [_msg("user", "q3"), _msg("user", "q4")]


def test_multiple_sessions_skip_broken():
    sm = DummySM({
        "sA": DummySession(DummyChat([_msg("user", "a1"), _msg("assistant", "a2")])),
        "sB": DummySession(DummyChat([_msg("tool", "b1")])),  # tool-only still kept
        "sC": DummySession(None),                # no chat session
        "sD": DummySession(DummyChat([])),       # empty history
    })
    out = extract_session_histories(sm, max_per_session=10)
    assert set(out.keys()) == {"sA", "sB"}
    assert out["sA"] == [_msg("user", "a1"), _msg("assistant", "a2")]
    assert out["sB"] == [_msg("tool", "b1")]


def test_ignores_empty_content_and_nondict():
    msgs = [
        _msg("user", ""),          # empty -> skipped
        {"role": "user"},          # no content key -> skipped
        "not a dict",              # junk -> skipped
        _msg("assistant", "real"),
    ]
    out = extract_session_histories(DummySM({"s1": DummySession(DummyChat(msgs))}))
    assert out["s1"] == [_msg("assistant", "real")]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
