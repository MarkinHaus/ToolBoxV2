import binascii
import hashlib
import logging
import math
import os
import random
import sys
import time
import uuid
import json
from pathlib import Path

import requests
from toolboxv2 import MainTool, FileHandler, Style
from .UserInstanceManager import UserInstances
from toolboxv2.mods.CloudM.AuthManager import get_invitation
from toolboxv2.mods.CloudM.ModManager import installer
from toolboxv2.mods.DB.tb_adapter import Tools as DB
from toolboxv2.utils.state_system import get_state_from_app, TbState
from toolboxv2.utils.toolbox import get_app


Name = 'cloudM'
version = "0.0.2"
export = get_app(f"{Name}.EXPORT").tb
no_test = export(mod_name=Name, test=False, version=version)
to_api = export(mod_name=Name, api=True, version=version)


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        t0 = time.perf_counter()
        self.version = version
        self.api_version = "404"
        self.name = "cloudM"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "CYAN"
        self.app = app
        if app is None:
            self.app = get_app()
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
            "Version": self.show_version,
            "api_Version": self.show_version,
            "NEW": self.new_module,
            "create-account": self.create_account,
            "login": self.log_in,
            "#update-core": self.update_core,
            "mod-installer": self.install_module,
            "system_init": self.prep_system_initial,
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
            installer(mod_installer_url)
        except Exception as e:
            self.print(f"Error : {e}")

    def prep_system_initial(self, do_root=False):

        db: DB = self.app.get_mod('DB')

        if db.data_base is None or not db:
            self.print(
                "No redis instance provided from db run DB first-redis-connection")
            return "Pleas connect first to a redis instance"

        if not do_root:
            if 'y' not in input(
                Style.RED("Ar u sure : the deb will be cleared type y :")):
                return

        db.delete('*', matching=True)
        i = 0
        for _ in db.get('all').get():
            i += 1

        if i != 0:
            self.print("Pleas clean redis database first")
            return "Data in database"

        secret = str(random.randint(0, 100))
        for i in range(4):
            secret += str(uuid.uuid5(uuid.NAMESPACE_X500, secret))

        db.set("jwt-secret-cloudMService", secret)
        db.set("email_waiting_list", '[]')

        key = get_invitation().get()

        print("First key :" + key)
        self.print("Server Initialized for root user")
        return True

    def load_open_file(self):
        self.logger.info("Starting cloudM")
        self.load_file_handler()
        # self.get_version()

    def on_exit(self):
        self.save_file_handler()

    @to_api
    def show_version(self):
        self.print(f"Version: {self.version} {self.api_version}")
        return self.version

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

    @no_test
    def new_module(self, mod_name: str, *options): # updater wie AI Fuctonal and class based hybrie , file / folder |<futüre>| rust py
        self.logger.info(f"Crazing new module : {mod_name}")
        boilerplate = """import logging
from toolboxv2 import MainTool, FileHandler, App, Style


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "NAME"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "WHITE"
        # ~ self.keys = {}
        self.tools = {
            "all": [["Version", "Shows current Version"]],
            "name": "NAME",
            "Version": self.show_version, # TODO if functional replace line with [            "Version": show_version,]
        }
        # ~ FileHandler.__init__(self, "File name", app.id if app else __name__, keys=self.keys, defaults={})
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                        name=self.name, logs=self.logger, color=self.color, on_exit=self.on_exit)

    def on_start(self):
        self.logger.info(f"Starting NAME")
        # ~ self.load_file_handler()

    def on_exit(self):
        self.logger.info(f"Closing NAME")
        # ~ self.save_file_handler()

"""
        helper_functions_class = """
    def show_version(self):
        self.print("Version: ", self.version)
        return self.version
"""
        helper_functions_func = """
def get_tool(app: App):
    return app.AC_MOD


def show_version(_, app: App):
    welcome_f: Tools = get_tool(app)
    welcome_f.print(f"Version: {welcome_f.version}")
    return welcome_f.version

"""

        self.logger.info(f"crating boilerplate")
        if '-fh' in options:
            boilerplate = boilerplate.replace('pass', '').replace('# ~ ', '')
            self.logger.info(f"adding FileHandler")
        if '-func' in options:
            boilerplate += helper_functions_func
            self.logger.info(f"adding functional based")
        else:
            boilerplate += helper_functions_class
            self.logger.info(f"adding Class based")
        self.print(f"Test existing {self.api_version=} ")

        self.logger.info(f"Testing connection")

        # self.get_version()

        if self.api_version != '404':
            if self.download(["", mod_name]):
                self.print(
                    Style.Bold(Style.RED("MODULE exists-on-api pleas use a other name")))
                return False

        self.print("NEW MODULE: " + mod_name, end=" ")
        if os.path.exists(f"mods/" + mod_name +
                          ".py") or os.path.exists(f"mods_dev/" + mod_name +
                                                   ".py"):
            self.print(Style.Bold(Style.RED("MODULE exists pleas use a other name")))
            return False

        fle = Path("mods_dev/" + mod_name + ".py")
        fle.touch(exist_ok=True)
        with open(f"mods_dev/" + mod_name + ".py", "wb") as mod_file:
            mod_file.write(bytes(boilerplate.replace('NAME', mod_name),
                                 'ISO-8859-1'))

        self.print("Successfully created new module")
        return True

    @no_test
    def create_account(self):
        version_command = self.app.config_fh.get_file_handler("provider::")
        url = "https://simeplecore.app/app/signup"
        if version_command is not None:
            url = version_command + "/app/signup"
        # os.system(f"start {url}")

        try:
            import webbrowser
            webbrowser.open(url, new=0, autoraise=True)
        except Exception as e:
            self.logger.error(Style.YELLOW(str(e)))
            self.print(Style.YELLOW(str(e)))
            return False
        return True

    @no_test
    def log_in(self, input_):
        version_command = self.app.config_fh.get_file_handler("provider::")
        url = "https://simeplecore.app/app/login"
        if version_command is not None:
            url = version_command + "/app/login"

        if len(input_) == 3:
            username = input_[1]
            password = input_[2]

            data = {"username": username, "password": password}

            r = requests.post(url, json=data)
            self.print(r.status_code)
            self.print(str(r.content, 'utf-8'))
            token = r.json()["token"]
            error = r.json()["error"]

            if not error:
                claims = token.split(".")[1]
                import base64
                json_claims = base64.b64decode(claims + '==')
                claims = eval(str(json_claims, 'utf-8'))
                self.print(Style.GREEN(f"Welcome : {claims['username']}"))
                self.print(Style.GREEN(f"Email : {claims['email']}"))
                self.add_to_save_file_handler(self.keys["TOKEN"], token)
                self.print("Saving token to file...")

                self.on_exit()
                self.load_open_file()

                self.print("Saved")

                return True

            else:
                self.print(Style.RED(f"ERROR: {error}"))
        else:
            self.print(
                Style.RED(
                    f"ERROR: {input_} len {len(input_)} != 3 | login username password"))

        return False

    @no_test
    def update_core(self, dackup=False, name=""):
        self.print("Init Update..")
        if dackup:
            os.system("git fetch --all")
            d = f"git branch backup-master-{self.app.id}-{self.version}-{name}"
            os.system(d)
            os.system("git reset --hard origin/master")
        out = os.system("git pull")
        self.app.remove_all_modules()
        try:
            com = " ".join(sys.orig_argv)
        except AttributeError:
            com = "python3 "
            com += " ".join(sys.argv)

        if out == 0:
            self.app.print_ok()
        else:
            print("ther was an errer updateing...\n\n")
            print(Style.RED(f"Error-code: os.system -> {out}"))
            print(
                "if you changet local files type $ cloudM #update-core save {name}")
            print(
                "your changes will be saved to a branch named : backup-master-{app.id}-{self.version}-{name}"
            )
            print(
                "you can apply yur changes after the update with:\ngit stash\ngit stash pop"
            )

        if out == -1:
            os.system("git fetch --all")
            os.system("git reset --hard origin/master")

        if "update" not in com:
            print("Restarting...")
            os.system(com)
        exit(0)

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
