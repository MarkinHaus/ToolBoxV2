import os
import sys
from pathlib import Path

import requests
from toolboxv2 import Style, Result
from .AuthManager import get_invitation
from toolboxv2 import get_app, Code

Name = 'CloudM'
version = "0.0.2"
export = get_app(f"{Name}.EXPORT").tb
no_test = export(mod_name=Name, test=False, version=version)
to_api = export(mod_name=Name, api=True, version=version)


@no_test
def new_module(self, mod_name: str,
               *options):  # updater wie AI Fuctonal and class based hybrie , file / folder |<futüre>| rust py
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
    print(self, input_)
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

@no_test
def register_initial_root_user(app, tbef=None):

    root_key = app.config_fh.get_file_handler("Pk" + Code.one_way_hash("root", "dvp-k")[:8])

    if root_key is not None:
        return Result.default_user_error(info="root user already Registered")

    email = input("enter ure Email:")
    invitation = get_invitation().get()
    app.run_any(tbef.AUTHMANAGER.CRATE_LOCAL_ACCOUNT, username="root", email=email,
                     invitation=invitation)

@no_test
def clear_db(self, do_root=False):

    db = self.app.get_mod('DB', spec=self.spec)

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
    return True

@to_api
def show_version(self):
    self.print(f"Version: {self.version} {self.api_version}")
    return self.version
