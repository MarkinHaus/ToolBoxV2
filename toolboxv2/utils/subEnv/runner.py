import asyncio
import json
import sys

from toolboxv2 import ApiResult, Result


# Subprocess Worker (client_worker.py)
async def worker_main(app):

    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

    writer = asyncio.StreamWriter(
        sys.stdout.buffer,
        protocol=asyncio.streams.FlowControlMixin,
        reader=None,
        loop=loop
    )

    while True:
        line = await reader.readline()
        if not line:
            break

        try:
            response = {'Nodata':None}
            command = json.loads(line.decode().strip())
            if command["type"] == "execute":
                if not app.mod_online(command["module"]):
                    await app.get_mod(command["module"])
                result = await app.a_run_function(
                   (command["module"] ,command["function"]),
                    args_=command["args"],
                    kwargs_=command["kwargs"]
                )
                if not isinstance(result, Result) and not isinstance(result, ApiResult):
                    result = Result.ok(
                        interface=app.interface_type,
                        data_info="Auto generated result",
                        data=result,
                        info="Function executed successfully"
                    ).set_origin(command["function"],)
                response = result.to_api_result().model_dump()
            elif command["type"] == "ping":
                response = {"type": "pong"}

            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()
        except Exception as e:
            error_response = Result.default_internal_error(str(e)).to_api_result().model_dump()
            writer.write(json.dumps(error_response).encode() + b"\n")
            await writer.drain()





if __name__ == "__main__":
    asyncio.run(worker_main())
