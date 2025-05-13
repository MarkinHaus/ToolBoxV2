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
import hashlib
import os
from dataclasses import asdict, dataclass

import yaml
from tqdm import tqdm

from ..extras.Style import Spinner
from .getting_and_closing_app import get_app


@dataclass
class DefaultFilesFormatElement:
    version: str = "-1"
    shasum: str = "-1"
    provider: str = "-1"
    url: str = "-1"

    def __str__(self):
        return f"{self.version=}{self.shasum=}{self.provider=}{self.url=}|".replace("self.", "")


@dataclass
class TbState:
    utils: dict[str, DefaultFilesFormatElement]
    mods: dict[str, DefaultFilesFormatElement]
    installable: dict[str, DefaultFilesFormatElement]
    runnable: dict[str, DefaultFilesFormatElement]
    api: dict[str, DefaultFilesFormatElement]
    app: dict[str, DefaultFilesFormatElement]

    def __str__(self):
        fstr = "Utils\n"
        for name, item in self.utils.items():
            fstr += f"  {name} | {str(item)}\n"
        fstr += "Mods\n"
        for name, item in self.mods.items():
            fstr += f"  {name} | {str(item)}\n"
        fstr += "Mods installable\n"
        for name, item in self.installable.items():
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
    utils = {}
    mods = {}
    runnable = {}
    installable = {}
    api = {}
    app = {}
    scann_dirs = ["utils", "mods", "runnable", "api", "app", "mods_sto"]
    for s_dir in scann_dirs:
        for root, _dirs, files in os.walk(directory+'/'+s_dir):
            for file_name in files:
                if file_name.endswith(".zip") and 'mods_sto' in root:
                    file_path = os.path.join(root, file_name)
                    shasum = calculate_shasum(file_path)

                    element = DefaultFilesFormatElement()
                    element.shasum = shasum
                    installable[file_name] = element

                if file_name.endswith(".py"):
                    file_path = os.path.join(root, file_name)
                    shasum = calculate_shasum(file_path)

                    element = DefaultFilesFormatElement()
                    element.shasum = shasum

                    if 'utils' in root:
                        utils[file_name] = element
                    elif 'mods' in root:
                        mods[file_name] = element
                    elif 'runnable' in root:
                        runnable[file_name] = element
                    elif 'api' in root:
                        api[file_name] = element
                    elif 'app' in root:
                        app[file_name] = element

    return TbState(
        utils=utils,
        mods=mods,
        installable=installable,
        runnable=runnable,
        api=api,
        app=app,
    )


def get_state_from_app(app, simple_core_hub_url="https://simplecore.app/",
                       github_url="https://github.com/MarkinHaus/ToolBoxV2/tree/master/toolboxv2/"):
    if simple_core_hub_url[-1] != '/':
        simple_core_hub_url += '/'

    if github_url[-1] != '/':
        github_url += '/'

    with Spinner("Scanning files"):
        state: TbState = process_files(app.start_dir)

    with tqdm(total=6, unit='chunk', desc='Building State data') as pbar:
        # and unit information
        # current time being units ar installed and managed via GitHub
        version = app.version
        pbar.write("working on utils files")
        for file_name, file_data in state.utils.items():
            file_data.provider = "git"
            file_data.version = version
            file_data.url = github_url + "utils/" + file_name
        pbar.update()
        pbar.write("working on api files")
        for file_name, file_data in state.api.items():
            file_data.provider = "git"
            file_data.version = version
            file_data.url = github_url + "api/" + file_name
        pbar.update()
        pbar.write("working on app files")
        for file_name, file_data in state.app.items():
            file_data.provider = "git"
            file_data.version = version
            file_data.url = github_url + "app/" + file_name

        # and mods information
        # current time being mods ar installed and managed via SimpleCoreHub.com
        all_mods = app.get_all_mods()
        pbar.update()
        pbar.write("working on mods files")
        for file_name, file_data in state.mods.items():
            file_data.provider = "SimpleCore"

            module_name = file_name.replace(".py", "")
            if module_name in all_mods:
                try:
                    file_data.version = app.get_mod(module_name).version if file_name != "__init__.py" else version
                except Exception:
                    file_data.version = "dependency"
            else:
                file_data.version = "legacy"

            file_data.url = simple_core_hub_url + "mods/" + file_name
        pbar.update()
        pbar.write("working on installable files")
        for file_name, file_data in state.installable.items():
            file_data.provider = "SimpleCore"
            try:
                file_data.version = file_name.replace(".py", "").replace(".zip", "").split("&")[-1].split("§")
            except Exception:
                file_data.version = "dependency"

            file_data.url = simple_core_hub_url + "installer/download/mods_sto/" + file_name
        pbar.update()
        pbar.write("Saving State Data")
        with open("tbState.yaml", "w") as config_file:
            yaml.dump(asdict(state), config_file)
        pbar.update()
    return state


if __name__ == "__main__":
    # Provide the directory to search for Python files
    app = get_app('state.system')
    app.load_all_mods_in_file()
    state = get_state_from_app(app)
    print(state)
    # def get_state_from_app(app: App):
    #    """"""
