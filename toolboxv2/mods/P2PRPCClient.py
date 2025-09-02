import asyncio
import json
import os
import uuid

from toolboxv2 import App, Result, get_app
from toolboxv2.utils.security.cryp import Code

# Define the module name and export function
Name = 'P2PRPCClient'
export = get_app(f"{Name}.Export").tb
version = "0.1.0"

class P2PRPCClient:
    def __init__(self, app: App, host: str, port: int, tb_r_key: str = None):
        self.app = app
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None
        self.futures = {}
        self.code = Code()

        if tb_r_key is None:
            tb_r_key = os.getenv("TB_R_KEY")
            if tb_r_key is None:
                raise ValueError("TB_R_KEY environment variable is not set.")

        if len(tb_r_key) < 24:
            raise ValueError("TB_R_KEY must be at least 24 characters long for security.")
        self.auth_key_part = tb_r_key[:24]
        self.identification_part = tb_r_key[24:]
        self.session_key = None

    async def connect(self):
        """Connects to the local tcm instance and performs key exchange."""
        try:
            self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
            print(f"RPC Client: Connected to tcm at {self.host}:{self.port}")

            # Receive encrypted session key from server
            len_data = await self.reader.readexactly(4)
            encrypted_session_key_len = int.from_bytes(len_data, 'big')
            encrypted_session_key = (await self.reader.readexactly(encrypted_session_key_len)).decode('utf-8')

            # Decrypt session key using auth_key_part
            self.session_key = self.code.decrypt_symmetric(encrypted_session_key, self.auth_key_part)

            # Send challenge back to server, encrypted with session key
            challenge = "CHALLENGE_ACK"
            encrypted_challenge = self.code.encrypt_symmetric(challenge, self.session_key)
            self.writer.write(len(encrypted_challenge).to_bytes(4, 'big'))
            self.writer.write(encrypted_challenge.encode('utf-8'))
            await self.writer.drain()

            # Start a background task to listen for responses
            asyncio.create_task(self.listen_for_responses())

        except ConnectionRefusedError:
            print(f"RPC Client: Connection to {self.host}:{self.port} refused. Is the tcm peer running?")
            raise
        except Exception as e:
            print(f"RPC Client: Error during connection/key exchange: {e}")
            raise

    async def listen_for_responses(self):
        """Listens for incoming responses, decrypts them, and resolves the corresponding future."""
        try:
            while True:
                len_data = await self.reader.readexactly(4)
                msg_len = int.from_bytes(len_data, 'big')
                encrypted_msg_data = (await self.reader.readexactly(msg_len)).decode('utf-8')

                decrypted_msg_data = self.code.decrypt_symmetric(encrypted_msg_data, self.session_key)
                response = json.loads(decrypted_msg_data)

                call_id = response.get('call_id')
                if call_id in self.futures:
                    future = self.futures.pop(call_id)
                    future.set_result(response)
        except asyncio.IncompleteReadError:
            print("RPC Client: Connection closed.")
        except Exception as e:
            print(f"RPC Client: Error listening for responses: {e}")
        finally:
            # Clean up any pending futures
            for future in self.futures.values():
                future.set_exception(ConnectionError("Connection lost"))
            self.futures.clear()

    async def call(self, module: str, function: str, *args, **kwargs):
        """Makes a remote procedure call."""
        if not self.writer:
            await self.connect()

        call_id = str(uuid.uuid4())
        request = {
            "type": "request",
            "call_id": call_id,
            "module": module,
            "function": function,
            "args": args,
            "kwargs": kwargs,
            "identification_part": self.identification_part
        }

        future = asyncio.get_running_loop().create_future()
        self.futures[call_id] = future

        try:
            request_str = json.dumps(request)
            encrypted_request = self.code.encrypt_symmetric(request_str, self.session_key)

            self.writer.write(len(encrypted_request).to_bytes(4, 'big'))
            self.writer.write(encrypted_request.encode('utf-8'))
            await self.writer.drain()

            # Wait for the response with a timeout
            response = await asyncio.wait_for(future, timeout=30.0)

            if response.get('error'):
                return Result(**response['error'])
            else:
                return Result.ok(response.get('result'))

        except TimeoutError:
            self.futures.pop(call_id, None)
            return Result.default_internal_error("RPC call timed out.")
        except Exception as e:
            self.futures.pop(call_id, None)
            return Result.default_internal_error(f"RPC call failed: {e}")

    async def close(self):
        """Closes the connection."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            print("RPC Client: Connection closed.")

@export(mod_name=Name, name="test_rpc_client", test=False)
async def test_rpc_client(app: App, host: str = '127.0.0.1', port: int = 8000, tb_r_key: str = None):
    """An example of how to use the P2P RPC Client."""
    if tb_r_key is None:
        tb_r_key = os.getenv("TB_R_KEY")
        if tb_r_key is None:
            raise ValueError("TB_R_KEY environment variable is not set.")

    client = P2PRPCClient(app, host, port, tb_r_key)
    try:
        await client.connect()
        # Example: Call the 'list-users' function from the 'helper' module
        result = await client.call("helper", "list-users")
        result.print()
    finally:
        await client.close()
