import binascii
import hashlib
import logging
import math
import os
import time
import json

import requests

from toolboxv2 import MainTool, FileHandler, get_app, Style
from .UserInstances import UserInstances
from toolboxv2.utils.system.state_system import get_state_from_app, TbState
Name = 'CloudM'
version = "0.0.2"
export = get_app(f"{Name}.EXPORT").tb
no_test = export(mod_name=Name, test=False, version=version)
to_api = export(mod_name=Name, api=True, version=version)


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        t0 = time.perf_counter()
        self.version = version
        self.api_version = "404"
        self.name = "CloudM"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "CYAN"
        if app is None:
            app = get_app()
        self.user_instances = UserInstances()
        self.keys = {
            "URL": "comm-vcd~~",
            "URLS": "comm-vcds~",
            "TOKEN": "comm-tok~~",
        }
        self.tools = {
            "all": [
                ["Version", "Shows current Version"],
                ["api_Version", "Shows current Version"],
                [
                    "NEW", "crate a boilerplate file to make a new mod",
                    "add is case sensitive", "flags -fh for FileHandler",
                    "-func for functional bas Tools els class based"
                ],
                [
                    "download", "download a mod from MarkinHaus server",
                    "add is case sensitive"
                ],
                [
                    "#update-core", "update ToolBox from (git) MarkinHaus server ",
                    "add is case sensitive"
                ],
                [
                    "upload", "upload a mod to MarkinHaus server",
                    "add is case sensitive"
                ],
                ["first-web-connection", "set up a web connection to MarkinHaus"],
                ["create-account", "create a new account"],
                ["login", "login with Username & password"],
                ["api_create_user", "create a new user - api instance"],
                ["api_validate_jwt", "validate a  user - api instance"],
                ["validate_jwt", "validate a user"],
                ["api_log_in_user", "log_in user - api instance"],
                ["api_log_out_user", "log_out user - api instance"],
                ["api_email_waiting_list", "email_waiting_list user - api instance"],
                ["download_api_files", "download mods"],
                ["get-init-config", "get-init-config mods"],
                ["mod-installer", "installing mods via json url"],
                ["mod-remover", "remover mods via json url"],
                [
                    "wsGetI", "remover mods via json url", math.inf, 'get_instance_si_id'
                ],
                ["validate_ws_id", "remover mods via json url", math.inf],
                ["system_init", "Init system", math.inf, 'prep_system_initial'],
                ["close_user_instance", "close_user_instance", math.inf],
                ["get_user_instance", "get_user_instance only programmatic", math.inf],
                ["set_user_level", "set_user_level only programmatic", math.inf],
                ["make_installable", "crate pack for toolbox"],
            ],
            "name": "cloudM",
            "mod-installer": self.install_module,
            "mod-uninstaller": self.uninstall_module,
        }

        self.logger.info("init FileHandler cloudM")
        t1 = time.perf_counter()
        FileHandler.__init__(self, "modules.config", app.id if app else __name__,
                             self.keys, {
                                 "URL": '"https://simpelm.com/api"',
                                 "TOKEN": '"~tok~"',
                             })
        self.logger.info(f"Time to initialize FileHandler {time.perf_counter() - t1}")
        t1 = time.perf_counter()
        self.logger.info("init MainTool cloudM")
        MainTool.__init__(self,
                          load=self.load_open_file,
                          v=self.version,
                          tool=self.tools,
                          name=self.name,
                          logs=self.logger,
                          color=self.color,
                          on_exit=self.on_exit)

        self.logger.info(f"Time to initialize MainTool {time.perf_counter() - t1}")
        self.logger.info(
            f"Time to initialize Tools {self.name} {time.perf_counter() - t0}")

    def install_module(self, name):

        if isinstance(
            name, list
        ) and len(name) == 2:
            name = name[1]

        self.print(f"Installing module : {name}")

        mod_installer_url = self.get_file_handler(self.keys["URL"]) + fr"/install/installer\{name}-installer.json"
        self.print(f"Installer file url: {mod_installer_url}")
        try:
            pass
            #installer(mod_installer_url, self.app.debug)
        except Exception as e:
            self.print(f"Error : {e}")

    def uninstall_module(self, name):

        if isinstance(
            name, list
        ) and len(name) == 2:
            name = name[1]

        self.print(f"Installing module : {name}")

        mod_installer_url = self.get_file_handler(self.keys["URL"]) + fr"/install/installer\{name}-installer.json"
        self.print(f"Installer file url: {mod_installer_url}")
        try:
            pass
            # delete_package(mod_installer_url)
        except Exception as e:
            self.print(f"Error : {e}")

    def load_open_file(self):
        self.logger.info("Starting cloudM")
        self.load_file_handler()
        # self.get_version()

    def on_exit(self):
        self.save_file_handler()

    def get_version(self):  # Add root and upper and controll comander pettern
        version_command = self.app.config_fh.get_file_handler("provider::")

        url = version_command + "api/cloudm/show_version"

        try:
            self.api_version = requests.get(url, timeout=5).json()["res"]
            self.print(f"API-Version: {self.api_version}")
        except Exception as e:
            self.logger.error(Style.YELLOW(str(e)))
            self.print(
                Style.RED(
                    f" Error retrieving version from {url}\n\t run : cloudM first-web-connection\n"
                ))
            self.logger.error(f"Error retrieving version from {url}")

    def save_mod_snapshot(self, mod_name, provider=None, tb_state: TbState or None = None):
        if provider is None:
            provider = self.get_file_handler(self.keys["URL"])
        if tb_state is None:
            tb_state: TbState = get_state_from_app(self.app, simple_core_hub_url=provider)
        mod_data = tb_state.mods.get(mod_name)
        if mod_data is None:
            mod_data = tb_state.mods.get(mod_name + ".py")

        if mod_data is None:
            self.print(f"Valid ar : {list(tb_state.mods.keys())}")
            return None

        if not os.path.exists("./installer"):
            os.mkdir("./installer")

        json_data = {"Name": mod_name,
                     "mods": [mod_data.url],
                     "runnable": None,
                     "requirements": None,
                     "additional-dirs": None,
                     mod_name: {
                         "version": mod_data.version,
                         "shasum": mod_data.shasum,
                         "provider": mod_data.provider,
                         "url": mod_data.url
                     }}
        installer_path = f"./installer/{mod_name}-installer.json"
        if os.path.exists(installer_path):
            with open(installer_path, "r") as installer_file:
                file_data: dict = json.loads(installer_file.read())
                if len(file_data.get('mods', [])) > 1:
                    file_data['mods'].append(mod_data.url)
                file_data[mod_name] = json_data[mod_name]

                json_data = file_data

        with open(installer_path, "w") as installer_file:
            json.dump(json_data, installer_file)

        return json_data




# Create a hashed password
def hash_password(password):
    """Hash a password for storing."""
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt,
                                  100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt + pwdhash).decode('ascii')


# Check hashed password validity
def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    salt = stored_password[:64]
    stored_password = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha512', provided_password.encode('utf-8'),
                                  salt.encode('ascii'), 100000)
    pwdhash = binascii.hexlify(pwdhash).decode('ascii')
    return pwdhash == stored_password
