import asyncio
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from enum import Enum
import subprocess
import select


class SubprocessClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.process = None
        self.stdin = None
        self.stdout = None
        self.healthy = False

    async def start(self,program="python", *args):
        self.process = await asyncio.create_subprocess_exec(
            program,  *args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.stdin = self.process.stdin
        self.stdout = self.process.stdout
        await self.initialize_connection()

    async def initialize_connection(self):
        await self.send_command({"type": "handshake", "client_id": self.client_id})
        response = await self.read_response()
        self.healthy = response.get("status") == "ready"

    async def send_command(self, command: Dict):
        data = json.dumps(command) + "\n"
        self.stdin.write(data.encode())
        await self.stdin.drain()

    async def read_response(self):
        line = await self.stdout.readline()
        return json.loads(line.decode().strip())
