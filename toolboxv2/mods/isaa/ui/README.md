# ISAA UI

FastTB-based web frontend for ToolBoxV2 / ISAA / FlowAgent V3.

## Scope (this release: P0–P3)

Implemented:
- Welcome screen (logo + greeting + input bubble)
- Streaming chat via `agent.a_stream` → WS frames with seq
- Step cards with Level 0 / 1 / 2 expansion
- Tool pills with pretty inline args (no raw JSON)
- Sidebar: New chat, Chats list, VFS tree+read+write+download+ZIP, Agent config, Skills
- Drag-drop upload (desktop)
- Step rollback (anchor click)
- Reconnect with frame replay
- localStorage persist: active chat, sidebar panel, expanded steps, draft text

Not in this release (P4+):
- Widget mode / GridStack
- Mobile bottom-nav
- Sub-agent live footer
- Custom template registration
- ISA logo full spin animation during execution

## Launch

```bash
# From repo root (ToolBoxV2 must be importable):
python -m toolboxv2.mods.isaa.ui.app --host 127.0.0.1 --port 8765

# Or via tb CLI (see launchUI command in module.py — register the entry point).
tb -m isaa launchUI --port 8765
```

Open: <http://127.0.0.1:8765>

The TBJS Glass main.css is injected automatically by `FastTBHandler._maybe_inject_style` (no manual link needed).

## Architecture

```
Browser (localStorage mirror)
        ▲
        │  HTTP REST + WS /ws/chat
        ▼
ui/app.py (FastTB)
        ├─ routes/chats.py     — chat CRUD + rollback
        ├─ routes/vfs.py       — VFS tree/read/write/upload/zip
        ├─ routes/agents.py    — list + GET/PUT config
        ├─ routes/skills.py    — per-agent skills CRUD
        ├─ chat_store.py       — JSONL persistence per chat
        └─ stream_bridge.py    — a_stream → WS frames + persist
                ↓
        isaa.get_agent(name) → FlowAgent → a_stream(...)
```

### Storage

```
<app.data_dir>/isaa_ui/chats/
    <chat_id>.jsonl       # append-only, one frame per line, seq monotonic
    <chat_id>.meta.json   # title, agent, session_id, run_id, ui state
```

### WS protocol

Fixed path `/ws/chat`. Client sends `hello` with `chat_id` and `last_seq`,
server replays all frames > last_seq, then streams live ones.

Client ops: `hello`, `send`, `pause`, `cancel`, `ping`, `rollback`.
Server frame types: see `chat.js` and `stream_bridge.py`.

### run_id tracking

`StreamBridge` reads `engine._session_last_run.get(session_id)` after the
first chunk to capture the run_id for later pause/cancel. Explicit `run_id`
in `paused`/`cancelled` chunks override.

## Tests

```bash
python -m unittest toolboxv2.mods.isaa.ui.tests.test_chat_store
python -m unittest toolboxv2.mods.isaa.ui.tests.test_stream_bridge

# or all:
python -m unittest discover -s toolboxv2/mods/isaa/ui/tests -p 'test_*.py' -v
```

Tests use only `unittest` (no pytest, per project rules). Tests for routes
require a live FastTB app and are covered by manual smoke + the screenshot
checklist below.

## Smoke checklist (manual)

1. Start the server, open `http://127.0.0.1:8765`.
2. Welcome screen shows logo + greeting.
3. Type any prompt → Enter. Expect: user bubble appears, then a step card
   starts streaming, tool pills appear inline as tools fire.
4. Reload the browser. Chat should reappear identically (replay works).
5. Open Sidebar → Chats. New chat is listed. Click to switch.
6. Open Sidebar → VFS. Tree shows. Click a file → modal opens with content.
   Edit + Save closes modal and persists.
7. Open Sidebar → Agent. Change `system_message`, save. Status line shows
   `applied_hot: [system_message]`.
8. Open Sidebar → Skills. Add a new skill via the prompt flow. It appears
   in the list. Click → edit form opens.
9. Drag a file onto the bubble. Chip appears. Send. Assistant sees the path.
10. Click the small dot anchor of any past step → confirm → rollback. Chat
    truncates back to before that step.

## Known limits / next iterations

- **WS handler context resolution** uses a fallback `_MinCtx` if the real
  `WebSocketContext` is not passed by the ToolBoxV2 WS infrastructure.
  If broadcasts don't reach the client, dig into
  `WebSocketMessageHandler.handle_ws_message` and confirm what shape
  `payload` / `session` / `ctx` arrive in for `@app.websocket`-class methods.
- **Step grouping** in `chat.js` uses `iteration_start` as the boundary.
  If the engine starts yielding without an `iteration_start` frame for the
  first iteration, the first step may merge into the second. Adjust
  `groupFrames` if needed.
- **Skill `from_dict`** assumes `Skill` has a `from_dict` classmethod; if
  not, the route falls back to `Skill(**body)` which requires the body to
  match the dataclass shape exactly.
