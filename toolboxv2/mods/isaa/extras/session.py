from datetime import datetime
from toolboxv2 import Singleton

class ChatSession:

    def __init__(self, mem, max_length=None):
        self.mem = mem
        self.space_name = "chat_session"
        self.max_length = max_length
        self.history = []
        self.mem.create_memory(self.space_name)

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

    async def get_reference(self, text):
        return "\n".join([str(x) for x in await self.mem.query(text, self.space_name)])

    def get_past_x(self, x):
        return self.history[-x:]


