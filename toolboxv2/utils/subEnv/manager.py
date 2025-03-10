import asyncio
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from enum import Enum
import subprocess
import select

from toolboxv2 import ApiResult
from toolboxv2.utils.subEnv.client import SubprocessClient


class ClientManager:
    def __init__(self):
        self.clients: List[SubprocessClient] = []
        self.round_robin_index = 0

    async def add_client(self,program="python", *args):
        client = SubprocessClient(f"client{len(self.clients)}")
        await client.start(program=program, *args)
        self.clients.append(client)
        return client

    async def execute_function(self, module: str, func: str, args: list, kwargs: dict):
        client = self._select_client()
        command = {
            "type": "execute",
            "module": module,
            "function": func,
            "args": args,
            "kwargs": kwargs
        }
        await client.send_command(command)
        response = await client.read_response()
        return ApiResult(**response).as_result()

    def _select_client(self):
        # Simple round-robin load balancing
        client = self.clients[self.round_robin_index % len(self.clients)]
        self.round_robin_index += 1
        return client

    async def health_check(self):
        for client in self.clients:
            try:
                await client.send_command({"type": "ping"})
                pong = await client.read_response()
                client.healthy = pong.get("type") == "pong"
            except Exception as e:
                client.healthy = False

    def remove_unhealthy(self):
        self.clients = [c for c in self.clients if c.healthy]
