"""Console script for toolboxv2. Isaa CMD Tool"""
import difflib
import os
import random
import threading
import uuid

import chardet

from toolboxv2 import App, get_logger
from toolboxv2.mods.isaa_extars.isaa_modi import init_isaa, split_todo_list, generate_exi_dict, sys_print
from toolboxv2.utils.toolbox import ApiOb
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import os
import queue
from unidiff import PatchSet

NAME = "live-file"


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, q, path):
        super().__init__()
        self.q = q
        self.file_contents = self.initialize_file_contents(path)

    def initialize_file_contents(self, path):
        file_contents = {}
        for root, dirs, files in os.walk(path):
            print("rdf", root, dirs, files)
            for file in files:
                print("file", file)
                file_path = os.path.join(root, file)
                print("testing", file_path)
                if os.path.isfile(file_path):
                    contents, encoding = self.read_file(file_path)
                    file_contents[file_path] = (contents, encoding)
        return file_contents

    def read_file(self, path):
        try:
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    return file.read(), 'utf-8'
            except UnicodeDecodeError:
                rawdata = open(path, 'rb').read()
                result = chardet.detect(rawdata)
                encoding = result['encoding']

                with open(path, 'r', encoding=encoding) as file:
                    return file.read(), encoding
        except IOError:
            return "", None

    def on_modified(self, event):
        if os.path.isfile(event.src_path):
            new_contents, encoding = self.read_file(event.src_path)
            old_contents, _ = self.file_contents.get(event.src_path, ("", None))
            changes = compare_contents(old_contents, new_contents)
            info_dict = {
                'path': event.src_path,
                'event': 'modified',
                'changes': changes,
                'encoding': encoding
            }
            self.q.put(info_dict)
            self.file_contents[event.src_path] = (new_contents, encoding)


def compare_contents(old, new):
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff = difflib.unified_diff(old_lines, new_lines)
    return '\n'.join(diff)


def apply_diff(original, diff):
    patch_set = PatchSet(diff)
    original_lines = original.splitlines()

    for patched_file in patch_set:
        for hunk in patched_file:
            for line in hunk:
                if line.is_added:
                    original_lines.insert(line.target_line_no - 1, line.value)
                elif line.is_removed:
                    original_lines.pop(line.source_line_no - 1)

    return '\n'.join(original_lines)


def watch_path(path, q):
    event_handler = FileChangeHandler(q, path)
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=True)
    observer.start()
    return observer, event_handler


def start_watch(path):
    q = queue.Queue()
    observer, handler = watch_path(path, q)
    return q, observer, handler


def run(app: App, args):
    observations = {}

    def new_observer(path):
        changes, observer, handler = start_watch(path)
        observations[path] = [changes, observer, handler]
        return handler.file_contents

    def close_observer(path):
        _c, observer = observations[path]
        observer.close()
        del _c

    def close_all_observers():
        for path in observations.keys():
            close_observer(path)

    def get_changes():
        changes = []
        for key, value in observations.items():
            change, _, _ = value
            while not change.empty():
                changes.append(change.get())
        return changes

    ws_id_path = f".data/{app.id}/LiveFiles/"
    ws_id_file = f".data/{app.id}/LiveFiles/WsID.data"

    if not os.path.exists(ws_id_path):
        os.makedirs(ws_id_path, exist_ok=True)
    if not os.path.exists(ws_id_file):
        open(ws_id_file, "a").close()

    ws_id = ''

    if 'y' in input("Connect with old Connection-ID"):
        with open(ws_id_file, "r") as f:
            ws_id = f.read()

    if not ws_id:
        sys_print("PleasEnter Connection-ID:")
        # username = input("Username:")
        # password = input("Password:")
        # data = ApiOb()
        # data.data = {username, password}
        # print(data)
        # command = [data, ""]
        ws_id = input()  # app.run_any("cloudM", "log_in_user", command)
        print(ws_id)
        with open(ws_id_file, "w") as f:
            f.write(ws_id)

    logger = get_logger()

    sender, receiver = app.run_any("WebSocketManager", "srqw", ["ws://localhost:5000/ws", ws_id])

    seed_id = str(uuid.uuid4())
    rs = random.randint(4, len(seed_id) - 1)
    connaction_id = seed_id[3:] + '-' + seed_id[rs - 3:rs]

    def runner_receiver():

        running = True
        while running:
            while not receiver.empty():
                data = receiver.get()
                keys = data.keys()
                if 'exit' in keys:
                    running = False
                    sender.put({'exit': f"exit"})
                    close_all_observers()
                    exit(0)

                if not 'LiveFile' in keys:
                    continue

                if not 'vKey' in keys:
                    continue

                if data['vKey'] != connaction_id:
                    continue

                if 'new' in keys and 'path' in keys:
                    sender.put({'ChairData': True, "data": {"info": f"reading-path {data['path']}"}})
                    file_content = new_observer(data['path'])
                    sender.put({'ChairData': True, "data": {"file_content": file_content}}) # TODO: sync wtith isaa

                if 'close' in keys and 'path' in keys:
                    close_observer(data['path'])

                if 'write' in keys and 'path' in keys and 'data' in keys:
                    with open(data['path'], "w") as f0:
                        f0.write(data['data'])

                if 'write-diff' in keys and 'path' in keys and 'data' in keys:
                    with open(data['path'], "r") as f1:
                        og_file_data = f1.read()

                    new_data = apply_diff(og_file_data, data['path'])
                    with open(data['path'], "w") as f2:
                        f2.write(new_data)

                # write file functionality

    widget_runner = threading.Thread(target=runner_receiver)
    widget_runner.start()

    print("Starting with Validation Id:", connaction_id)
    running_s = True
    while running_s:
        changes = get_changes()
        if changes:
            sender.put({'ChairData': True, "data": {'changes': changes}})
        time.sleep(10)

    close_all_observers()
