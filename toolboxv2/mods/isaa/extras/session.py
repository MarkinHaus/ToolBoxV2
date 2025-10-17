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
        if os.path.exists(f'{get_app().appdata}/{space_name}.mem'):
            self.mem.load_memory(self.space_name, f'{get_app().appdata}/{space_name}.mem')
        else:
            self.mem.create_memory(self.space_name)
            os.makedirs(f'{get_app().appdata}', exist_ok=True)
            os.makedirs(f'{get_app().appdata}/ChatSession', exist_ok=True)

    async def add_message(self, message, direct=True):
        self.history.append(message)
        role = ""
        if message['role'].startswith('s'):
            role = "system"
        elif message['role'].startswith('u'):
            role = "user"
        elif message['role'].startswith('a'):
            role = "assistant"
        else:
            raise ValueError(f"Invalid role value {message['role']}")
        await self.mem.add_data(self.space_name, message['content'],
                          [{'role': role,
                            'timestamp': datetime.now().isoformat()}], direct=direct)
        if self.max_length and len(self.history) > self.max_length:
            self.history.pop(0)

    async def get_reference(self, text, **kwargs):
        return "\n".join([str(x) for x in await self.mem.query(text, self.space_name, **kwargs)])

    def get_past_x(self, x, last_u=False):
        if last_u:
            return self.get_start_with_last_user()
        return self.history[-x:]

    def get_start_with_last_user(self, x=None):
        if x is None:
            x = len(self.history)
        history = []
        for h in self.get_past_x(x, last_u=False)[::-1]:
            history.append(h)
            if h.get('role') == 'user':
                break
        return history[::-1]
    def on_exit(self):
        self.mem.save_memory(self.space_name, f'{get_app().appdata}/{self.space_name}.mem')


