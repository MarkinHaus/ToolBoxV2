import asyncio
import os
import sys
from pathlib import Path
from urllib.parse import quote

from toolboxv2 import Style, Result, tbef, App
from toolboxv2 import get_app, Code
from .AuthManager import get_invitation
from ...utils.extras.qr import print_qrcode_to_console

Name = 'CloudM'
version = "0.0.2"
export = get_app(f"{Name}.EXPORT").tb
no_test = export(mod_name=Name, test=False, version=version)
test_only = export(mod_name=Name, test=True, version=version, test_only=True)
to_api = export(mod_name=Name, api=True, version=version)


@no_test
def new_module(self, mod_name: str,
               *options):  # updater wie AI Functional and class based hybrid , file / folder |<futüre>| rust py
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
    url = "https://simeplecore.app/web/signup"
    if version_command is not None:
        url = version_command + "/web/signup"
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
def init_git(_):
    os.system("git init")


@no_test
def update_core(self, backup=False, name=""):
    self.print("Init Update..")
    if backup:
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
        print("their was an error updating...\n\n")
        print(Style.RED(f"Error-code: os.system -> {out}"))
        print(
            "if you changes local files type $ cloudM update_core save {name}")
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
def create_magic_log_in(app: App, username: str):
    user = app.run_any(tbef.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=username)
    key = "01#" + Code.one_way_hash(user.user_pass_sync, "CM", "get_magick_link_email")
    base_url = app.config_fh.get_file_handler("provider::") + (
        f':{app.args_sto.port}' if app.args_sto.host == 'localhost' else "")
    url = f"{base_url}/web/assets/m_log_in.html?key={quote(key)}&name={user.name}"
    return url


@no_test
async def register_initial_root_user(app: App, email=None):
    root_key = app.config_fh.get_file_handler("Pk" + Code.one_way_hash("root", "dvp-k")[:8])

    if root_key is not None:
        return Result.default_user_error(info="root user already Registered")

    if email is None:
        email = input("enter ure Email:")
    invitation = get_invitation(app=app).get()
    ret = app.run_any(tbef.CLOUDM_AUTHMANAGER.CRATE_LOCAL_ACCOUNT,
                      username="root",
                      email=email,
                      invitation=invitation, get_results=True)
    # awaiating user cration
    rport = await ret.aget()
    print(f"[{rport=}]")
    if rport.is_error():
        return rport
    await asyncio.sleep(1)
    user = await app.run_any(tbef.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username="root")
    print("User:")
    print(user)
    user = user.get()
    key = "01#" + Code.one_way_hash(user.user_pass_sync, "CM", "get_magick_link_email")
    base_url = app.config_fh.get_file_handler("provider::") + (
        f':{app.args_sto.port}' if app.args_sto.host == 'localhost' else "5000")
    url = f"{base_url}/web/assets/m_log_in.html?key={quote(key)}&name={user.name}"

    print_qrcode_to_console(url)

    print(url)
    return Result.ok(url)


@no_test
def clear_db(self, do_root=False):
    db = self.app.get_mod('DB', spec=self.spec)

    if db.data_base is None or not db:
        self.print(
            "No redis instance provided from db run DB first-redis-connection")
        return "Pleas connect first to a redis instance"

    if not do_root:
        if 'y' not in input(Style.RED("Ar u sure : the deb will be cleared type y :")):
            return

    db.delete('*', matching=True)
    i = 0
    for _ in db.get('all').get(default=[]):
        print(_)
        i += 1

    if i != 0:
        self.print("Pleas clean redis database first")
        return str(i) + " entry's Data in database"
    return True


@to_api
def show_version(self):
    self.print(f"Version: {self.version} {self.api_version}")
    return self.version


#@test_only
#async def tb_test_register_initial_root_user(app: App):
#    print("tb_test_register_initial_root_user 0")
#    if app is None:
#        app = get_app(from_="test_register_initial_root_user", name="debug-test")
#
#    print("tb_test_register_initial_root_user 1")
#    db = app.get_mod("DB")
#
#    print("tb_test_register_initial_root_user 2")
#    db.edit_programmable()
#
#    print("tb_test_register_initial_root_user 3")
#    s = await register_initial_root_user(app, "test@test.com")
#    print("tb_test_register2", s)
#    print("DONE running tb_test_register_initial_root_user")
