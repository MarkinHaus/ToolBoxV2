import asyncio
import difflib
import hashlib
import logging
import os
import tempfile

import websockets

from toolboxv2 import MainTool, FileHandler, App, Style


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "LiveFiles"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "WHITE"
        # ~ self.keys = {}
        self.tools = {
            "all": [["Version", "Shows current Version"]],
            "name": "LiveFiles",
            "Version": self.show_version, # TODO if functional replace line with [            "Version": show_version,]
        }
        self.file_content = ""
        self.file_hash = None
        self.temp_file = None
        # ~ FileHandler.__init__(self, "File name", app.id if app else __name__, keys=self.keys, defaults={})
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                        name=self.name, logs=self.logger, color=self.color, on_exit=self.on_exit)

    def on_start(self):
        self.logger.info(f"Starting LiveFiles")
        # ~ self.load_file_handler()

    def on_exit(self):
        self.logger.info(f"Closing LiveFiles")
        # ~ self.save_file_handler()

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    def set_file_path(self, file_content):
        self.print("Path: ")
        self.file_content = file_content
        self.file_hash = hashlib.sha256(self.file_content.encode()).hexdigest()

        # Create a temporary file and write the content into it
        self.temp_file = tempfile.TemporaryFile(mode='w+t')
        self.temp_file.write(self.file_content)
        self.temp_file.seek(0)

    async def watch_file(self):
        while True:
            # Read from the temporary file instead of the actual file
            new_content = self.temp_file.read()
            self.temp_file.seek(0)  # reset file pointer to the beginning

            new_hash = hashlib.sha256(new_content.encode()).hexdigest()
            if new_hash != self.file_hash:
                changes = self.identify_changes(self.file_content, new_content)
                self.file_content = new_content
                self.file_hash = new_hash
                await self.send_changes(changes)
            await asyncio.sleep(1)  # sleep for a while before checking again
    def identify_changes(self, old_content, new_content):
        self.print("Identifying changes")
        old_content_lines = old_content.splitlines()
        new_content_lines = new_content.splitlines()

        diff = difflib.unified_diff(old_content_lines, new_content_lines)

        changes = '\n'.join(list(diff))

        return changes

    async def send_changes(self, changes):
        async with websockets.connect('ws://localhost:500/app-live-test-DESKTOP-CI57V1L1515dc7f-c397-4a23-8ee4-1f23b8479d14CloudM-Signed') as websocket:
            await websocket.send(changes)

    async def write_to_file(self, new_content):
        with open(self.file_path, 'w') as file:
            file.write(new_content)
        new_hash = hashlib.sha256(new_content.encode()).hexdigest()
        if new_hash != self.file_hash:
            changes = self.identify_changes(self.file_content, new_content)
            self.file_content = new_content
            self.file_hash = new_hash
            await self.send_changes(changes)

    async def start_server(self, host, port):
        async def handler(websocket, path):
            async for message in websocket:
                await self.write_to_file(message)

        return await websockets.serve(handler, host, port)

def main():
    file_watcher = Tools()
    file_watcher.set_file_path('test content')
    file_watcher.watch_file()

if __name__ == "__main__":
    main()
