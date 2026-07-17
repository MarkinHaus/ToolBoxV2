"""
Dreamer V3 - Session History Extraction

Pulls the FULL conversation from ALL live sessions of the parent agent's
SessionManager. This is the data the Dreamer was missing: the rich
explanations/corrections in normal chat flow (not just resume-corrections
captured in the TaskMap).

The session history is already pruned + compressed by the agent, so the
Dreamer gets the complete, ready context - all roles, no truncation.

Pure logic - no agent boot, fully unit-testable. The parent SessionManager
keeps every session's ChatSession.history in RAM, so no disk glob is needed.

Author: FlowAgent V3
"""

from typing import Any


def extract_session_histories(
    session_manager: Any,
    max_per_session: int = 100,
) -> dict[str, list[dict]]:
    """
    Extract the newest messages (all roles) from every session.

    History is already compressed/cleaned by the agent -> full context,
    no role filtering, no content truncation.

    Args:
        session_manager: parent SessionManager (has .sessions dict).
        max_per_session: keep at most this many NEWEST messages per session.

    Returns:
        {session_id: [{"role","content",...}, ...]} - empty sessions omitted.
        Never raises: a broken session is skipped, not fatal.
    """
    result: dict[str, list[dict]] = {}
    sessions = getattr(session_manager, "sessions", None) or {}

    for sid, sess in sessions.items():
        try:
            chat = getattr(sess, "_chat_session", None)
            history = getattr(chat, "history", None) if chat else None
            if not history:
                continue

            msgs = [m for m in history if isinstance(m, dict) and m.get("content")]
            if max_per_session > 0:
                msgs = msgs[-max_per_session:]

            if msgs:
                result[str(sid)] = msgs
        except Exception:
            # defensive: one bad session must not break the dream cycle
            continue

    return result
