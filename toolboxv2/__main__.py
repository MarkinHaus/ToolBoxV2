import asyncio
import sys

from toolboxv2.utils.system import AppType
from .cli import main

if __name__ == "__main__":
    print("Starting From Main Guard")
    asyncio.run(main())
    sys.exit(0)
