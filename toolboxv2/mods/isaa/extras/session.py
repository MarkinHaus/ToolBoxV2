import os
from datetime import datetime

from toolboxv2 import get_app


class ChatSession:

    def __init__(self, mem, max_length=None, space_name = "chat_session"):
        self.mem = mem
        self.space_name = space_name
        if max_length is None:
            max_length = 100
        self.max_length = max_length
        self.history = []
        if not self.mem.load_memory(self.space_name, f'{get_app().appdata}/{space_name}.mem'):
            self.mem.create_memory(self.space_name)
            os.makedirs(f'{get_app().appdata}', exist_ok=True)
            os.makedirs(f'{get_app().appdata}/ChatSession', exist_ok=True)

    def clear_history(self):
        self.history = []

    async def add_message(self, message, direct=True, **kwargs):
        self.history.append(message)
        role = ""
        if message['role'].startswith('s'):
            role = "system"
        elif message['role'].startswith('u'):
            role = "user"
        elif message['role'].startswith('a'):
            role = "assistant"
        elif message['role'].startswith('t'):
            role = "tool"
        else:
            raise ValueError(f"Invalid role value {message['role']}")

        if '__sub__' in self.space_name:
            return
        await self.mem.add_data(self.space_name, message['content'],
                          {'role': role,
                            'timestamp': datetime.now().isoformat(), **kwargs}, direct=direct)
        if self.max_length and len(self.history) > self.max_length:
            self.history.pop(0)

    async def get_reference(self, text,row=False, **kwargs):
        kwargs["to_str"] = not row

        if '__sub__' in self.space_name:
            return await self.mem.query(text, None, **kwargs) if not row else await self.mem.query(text, None, **kwargs)
        return await self.mem.query(text, self.space_name, **kwargs) if not row else await self.mem.query(text, self.space_name, **kwargs)

    def get_past_x(self, x, last_u=False):
        if last_u:
            return self.get_start_with_last_user()
        return self.history[-x:]

    def get_start_with_last_user(self, x=None):
        # x = number of USER messages to include, counting back from the end.
        # x=None → 1 (history since the last user message; preserves last_u path).
        target_users = 1 if x is None else x
        if target_users <= 0:
            return []
        history = []
        users_seen = 0
        for h in self.history[::-1]:
            history.append(h)
            if h.get('role') == 'user':
                users_seen += 1
                if users_seen >= target_users:
                    break
        history.reverse()
        # ponytail: ChatSession is the clean conversation layer (no tool_calls),
        # so adjacent same-role text msgs are safe to merge → keeps strict
        # user/assistant alternation for providers that require it (Anthropic).
        merged = []
        for h in history:
            if (merged and merged[-1].get('role') == h.get('role')
                and 'tool_calls' not in merged[-1] and 'tool_calls' not in h):
                merged[-1] = {**merged[-1],
                              'content': f"{merged[-1].get('content', '')}\n{h.get('content', '')}".strip()}
            else:
                merged.append(h)
        return merged

    def on_exit(self):
        if '__sub__' in self.space_name:
            self.clear_history()
            return
        self.mem.save_memory(self.space_name, f'{get_app().appdata}/{self.space_name}.mem')

    def get_volume(self):
        return self.mem.get_memory_size(self.space_name)


