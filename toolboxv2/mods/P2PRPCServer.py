import asyncio
import json
import os

from toolboxv2 import App, Result, get_app
from toolboxv2.utils.security.cryp import Code

# Define the module name and export function
Name = 'P2PRPCServer'
export = get_app(f"{Name}.Export").tb
version = "0.1.0"

class P2PRPCServer:
    def __init__(self, app: App, host: str, port: int, tb_r_key: str, function_access_config: dict = None):
        self.app = app
        self.host = host
        self.port = port
        self.server = None
        self.code = Code()

        if len(tb_r_key) < 24:
            raise ValueError("TB_R_KEY must be at least 24 characters long for security.")
        self.auth_key_part = tb_r_key[:24]
        self.identification_part_server = tb_r_key[24:]

        self.function_access_config = function_access_config if function_access_config is not None else {}

    async def handle_client(self, reader, writer):
        """Callback to handle a single client connection from a tcm instance."""
        addr = writer.get_extra_info('peername')
        print(f"RPC Server: New connection from {addr}")

        session_key = self.code.generate_symmetric_key()
        encrypted_session_key = self.code.encrypt_symmetric(session_key, self.auth_key_part)

        try:
            writer.write(len(encrypted_session_key).to_bytes(4, 'big'))
            writer.write(encrypted_session_key.encode('utf-8'))
            await writer.drain()

            len_data = await reader.readexactly(4)
            encrypted_challenge_len = int.from_bytes(len_data, 'big')
            encrypted_challenge = (await reader.readexactly(encrypted_challenge_len)).decode('utf-8')

            decrypted_challenge = self.code.decrypt_symmetric(encrypted_challenge, session_key)
            if decrypted_challenge != "CHALLENGE_ACK":
                raise ValueError("Invalid challenge received.")

            print(f"RPC Server: Authenticated client {addr}")

            while True:
                len_data = await reader.readexactly(4)
                msg_len = int.from_bytes(len_data, 'big')

                encrypted_msg_data = (await reader.readexactly(msg_len)).decode('utf-8')

                decrypted_msg_data = self.code.decrypt_symmetric(encrypted_msg_data, session_key)

                response = await self.process_rpc(decrypted_msg_data, session_key)

                encrypted_response = self.code.encrypt_symmetric(json.dumps(response), session_key)

                writer.write(len(encrypted_response).to_bytes(4, 'big'))
                writer.write(encrypted_response.encode('utf-8'))
                await writer.drain()

        except asyncio.IncompleteReadError:
            print(f"RPC Server: Connection from {addr} closed.")
        except Exception as e:
            print(f"RPC Server: Error with client {addr}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def process_rpc(self, msg_data: str, session_key: str) -> dict:
        """Processes a single RPC request and returns a response dictionary."""
        try:
            call = json.loads(msg_data)
            if call.get('type') != 'request':
                raise ValueError("Invalid message type")
        except (json.JSONDecodeError, ValueError) as e:
            return self.format_error(call.get('call_id'), -32700, f"Parse error: {e}")

        call_id = call.get('call_id')
        module = call.get('module')
        function = call.get('function')
        args = call.get('args', [])
        kwargs = call.get('kwargs', {})
        client_identification = call.get('identification_part')

        if not self.is_function_allowed(module, function, client_identification):
            error_msg = f"Function '{module}.{function}' is not allowed for identification '{client_identification}'."
            print(f"RPC Server: {error_msg}")
            return self.format_error(call_id, -32601, "Method not found or not allowed")

        print(f"RPC Server: Executing '{module}.{function}' for '{client_identification}'")
        try:
            result: Result = await self.app.a_run_any(
                (module, function),
                args_=args,
                kwargs_=kwargs,
                get_results=True
            )

            if result.is_error():
                return self.format_error(call_id, result.info.get('exec_code', -32000), result.info.get('help_text'), result.get())
            else:
                return {
                    "type": "response",
                    "call_id": call_id,
                    "result": result.get(),
                    "error": None
                }
        except Exception as e:
            print(f"RPC Server: Exception during execution of '{module}.{function}': {e}")
            return self.format_error(call_id, -32603, "Internal error during execution", str(e))

    def is_function_allowed(self, module: str, function: str, client_identification: str) -> bool:
        """Checks if a function is allowed for a given client identification."""
        if module not in self.function_access_config:
            return False

        allowed_functions_for_module = self.function_access_config[module]

        if function not in allowed_functions_for_module:
            return False

        # If the function is whitelisted, and there's a specific identification part,
        # you might want to add more granular control here.
        # For now, if it's in the whitelist, it's allowed for any identified client.
        # You could extend function_access_config to be:
        # {"ModuleName": {"function1": ["id1", "id2"], "function2": ["id3"]}}
        # For simplicity, current implementation assumes if module.function is in whitelist,
        # it's generally allowed for any authenticated client.
        return True

    def format_error(self, call_id, code, message, details=None) -> dict:
        """Helper to create a JSON-RPC error response object."""
        return {
            "type": "response",
            "call_id": call_id,
            "result": None,
            "error": {
                "code": code,
                "message": message,
                "details": details
            }
        }

    async def start(self):
        """Starts the TCP server."""
        self.server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        addr = self.server.sockets[0].getsockname()
        print(f"P2P RPC Server listening on {addr}")
        async with self.server:
            await self.server.serve_forever()

    def stop(self):
        """Stops the TCP server."""
        if self.server:
            self.server.close()
            print("P2P RPC Server stopped.")

@export(mod_name=Name, name="start_server", test=False)
async def start_rpc_server(app: App, host: str = '127.0.0.1', port: int = 8888, tb_r_key: str = None, function_access_config: dict = None):
    """Starts the P2P RPC server."""
    if tb_r_key is None:
        tb_r_key = os.getenv("TB_R_KEY")
        if tb_r_key is None:
            raise ValueError("TB_R_KEY environment variable is not set.")

    server = P2PRPCServer(app, host, port, tb_r_key, function_access_config)
    try:
        await server.start()
    except KeyboardInterrupt:
        server.stop()
