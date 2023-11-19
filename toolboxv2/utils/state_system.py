"""The Task of the State System is :
 1 Kep trak of the current state of the ToolBox and its dependency's
 2 tracks the shasum of all mod and runnabael
 3 the version of all mod

 The state :
 {"utils":{"file_name": {"version":##,"shasum"}}
 ,"mods":{"file_name": {"version":##,"shasum":##,"src-url":##}}
 ,"runnable":{"file_name": {"version":##,"shasum":##,"src-url":##}}
 ,"api":{"file_name": {"version":##,"shasum"}}
 ,"app":{"file_name": {"version":##,"shasum":##,"src-url":##}}
 }

 trans form state from on to an other.
 """
from typing import Dict

import os
import hashlib
from typing import Dict

from toolboxv2 import App
from toolboxv2.utils.toolbox import get_app


class DefaultFilesFormatElement:
    version: str = "-1"
    shasum: str = "-1"
    provider: str = "-1"
    url: str = "-1"

    def __str__(self):
        return f"{self.version=}{self.shasum=}{self.provider=}{self.url=}|"


class TbState:
    utils: Dict[str, DefaultFilesFormatElement]
    mods: Dict[str, DefaultFilesFormatElement]
    runnable: Dict[str, DefaultFilesFormatElement]
    api: Dict[str, DefaultFilesFormatElement]
    app: Dict[str, DefaultFilesFormatElement]

    def __str__(self):
        fstr = "Utils\n"
        for name, item in self.utils.items():
            fstr += f"  {name} | {str(item)}\n"
        fstr += "Mods\n"
        for name, item in self.mods.items():
            fstr += f"  {name} | {str(item)}\n"
        fstr += "runnable\n"
        for name, item in self.runnable.items():
            fstr += f"  {name} | {str(item)}\n"
        fstr += "api\n"
        for name, item in self.api.items():
            fstr += f"  {name} | {str(item)}\n"
        fstr += "app\n"
        for name, item in self.app.items():
            fstr += f"  {name} | {str(item)}\n"
        return fstr


def calculate_shasum(file_path: str) -> str:
    BUF_SIZE = 65536

    sha_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        buf = file.read(BUF_SIZE)
        while len(buf) > 0:
            sha_hash.update(buf)
            buf = file.read(BUF_SIZE)

    return sha_hash.hexdigest()


def process_files(directory: str) -> TbState:
    state = TbState()

    state.utils = {}
    state.mods = {}
    state.runnable = {}
    state.api = {}
    state.app = {}
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                shasum = calculate_shasum(file_path)

                element = DefaultFilesFormatElement()
                element.shasum = shasum

                if 'utils' in root:
                    state.utils[file_name] = element
                elif 'mods' in root:
                    state.mods[file_name] = element
                elif 'runnable' in root:
                    state.runnable[file_name] = element
                elif 'api' in root:
                    state.api[file_name] = element
                elif 'app' in root:
                    state.app[file_name] = element

    return state


def get_state_from_app(app: App, simple_core_hub_url="https://SimpleCoreHub.com/Mods/",
                       github_url="https://github.com/MarkinHaus/ToolBoxV2/tree/master/toolboxv2/"):
    if simple_core_hub_url[-1] != '/':
        simple_core_hub_url += '/'

    simple_core_hub_url += 'api/'

    if github_url[-1] != '/':
        github_url += '/'

    state: TbState = process_files(app.start_dir)

    # and unit information
    # current time being units ar installed and managed via GitHub

    for file_name, file_data in state.utils.items():
        file_data.provider = "git"
        file_data.version = app.version
        file_data.url = github_url + "utils/" + file_name

    for file_name, file_data in state.api.items():
        file_data.provider = "git"
        file_data.version = app.version
        file_data.url = github_url + "api/" + file_name

    for file_name, file_data in state.app.items():
        file_data.provider = "git"
        file_data.version = app.version
        file_data.url = github_url + "app/" + file_name

    # and mods information
    # current time being mods ar installed and managed via SimpleCoreHub.com

    for file_name, file_data in state.mods.items():
        file_data.provider = "SimpleCore"
        try:
            file_data.version = app.get_mod(
                file_name.replace(".py", "")).version if file_name != "__init__.py" else app.version
        except Exception:
            file_data.version = "dependency"

        file_data.url = simple_core_hub_url + "mods/" + file_name

    return state


if __name__ == "__main__":
    # Provide the directory to search for Python files
    app = get_app()
    app.load_all_mods_in_file()
    state = get_state_from_app(app)
    print(str(state))
    # def get_state_from_app(app: App):
    #    """"""
