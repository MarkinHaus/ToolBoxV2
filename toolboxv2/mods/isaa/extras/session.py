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
        try:
            if not self.mem.load_memory(self.space_name, f'{get_app().appdata}/{space_name}.mem'):
                self.mem.create_memory(self.space_name)
        except ValueError:
            pass

    async def add_message(self, message):
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
                            'timestamp': datetime.now().isoformat()}])
        if self.max_length and len(self.history) > self.max_length:
            self.history.pop(0)

    async def get_reference(self, text, **kwargs):
        return "\n".join([str(x) for x in await self.mem.query(text, self.space_name, **kwargs)])

    def get_past_x(self, x):
        return self.history[-x:]

    def on_exit(self):
        self.mem.save_memory(self.space_name, f'{get_app().appdata}/{self.space_name}.mem')


