import asyncio
import sys


from .cli import main

if __name__ == "__main__":
    print("Starting From Main Guard")
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main(loop))
    sys.exit(0)
