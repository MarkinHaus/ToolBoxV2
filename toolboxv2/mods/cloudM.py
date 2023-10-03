import binascii
import hashlib
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
import json
import urllib.request
import shutil
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm

import jwt
import requests
from toolboxv2 import MainTool, FileHandler, App, Style, ToolBox_over
from toolboxv2.utils.Style import extract_json_strings
from toolboxv2.utils.toolbox import get_app

from toolboxv2.mods import Restrictor, VirtualizationTool, welcome


class Tools(MainTool, FileHandler):

  def __init__(self, app=None):
    t0 = time.time()
    self.version = "0.0.1"
    self.api_version = "404"
    self.name = "cloudM"
    self.logger: logging.Logger or None = app.logger if app else None
    self.color = "CYAN"
    self.app_ = app
    if app is None:
      self.app_ = get_app()

    self.keys = {
      "URL": "comm-vcd~~",
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
      "name":
      "cloudM",
      "Version": self.show_version,
        "api_Version": self.show_version,
        "NEW": self.new_module,
        "upload": self.upload,
        "download": self.download,
        "first-web-connection": self.add_url_con,
        "create-account": self.create_account,
        "login": self.log_in,
        "api_create_user": self.create_user,
        "api_log_in_user": self.log_in_user,
        "api_log_out_user": self.log_out_user,
        "api_email_waiting_list": self.email_waiting_list,
        "api_validate_jwt": self.validate_jwt,
        "validate_jwt": self.validate_jwt
        , "download_api_files": self.download_api_files,
        "#update-core": self.update_core,
        "wsGetI": self.get_instance_si_id,
        "validate_ws_id": self.validate_ws_id,
        "mod-installer": installer,
        "system_init": self.prep_system_initial,
        "mod-remover": delete_package,
        "close_user_instance": self.close_user_instance,
        "get_user_instance": self.get_user_instance_wrapper,
        "set_user_level": self.set_user_level,
    }

    self.live_user_instances = {}
    self.user_instances = {}
    self.vt = None
    self.rt = None
    self.logger.info("init FileHandler cloudM")
    t1 = time.time()
    FileHandler.__init__(self, "modules.config", app.id if app else __name__,
                         self.keys, {
                           "URL": '"https://simpelm.com/api"',
                           "TOKEN": '"~tok~"',
                         })
    self.logger.info(f"Time to initialize FileHandler {time.time() - t1}")

    t1 = time.time()
    self.logger.info("init MainTool cloudM")
    MainTool.__init__(self,
                      load=self.load_open_file,
                      v=self.version,
                      tool=self.tools,
                      name=self.name,
                      logs=self.logger,
                      color=self.color,
                      on_exit=self.on_exit)

    self.logger.info(f"Time to initialize MainTool {time.time() - t1}")
    self.logger.info(
      f"Time to initialize Tools {self.name} {time.time() - t0}")

  def prep_system_initial(self, command, app):

    db = app.get_mod('DB')

    if db is None or not db:
      self.print(
        "No redis instance provided from db run DB first-redis-connection")
      return "Pleas connect first to a redis instance"

    if db.rcon is None:
      self.print(
        "No redis instance provided from db run DB first-redis-connection")
      return "Pleas connect first to a redis instance"

    if command[0] != 'do-root':
      if 'y' not in input(
          Style.RED("Ar u sure : the deb will be cleared type y :")):
        return
    db.clean_db()
    i = 0
    for _ in db.rcon.scan_iter():
      i += 1

    if i != 0:
      self.print("Pleas clean redis database first")
      return "Data in database"

    secret = str(random.randint(0, 100))
    for i in range(4):
      secret += str(uuid.uuid5(uuid.NAMESPACE_X500, secret))

    db.rcon.set("jwt-secret-cloudMService", secret)
    db.rcon.set("email_waiting_list", '[]')

    key = str(uuid.uuid4())

    print("First key :" + key)
    self.print("Server Initialized for root user")
    db.rcon.set(key, "Valid")
    return True

  def get_si_id(self, uid):
    ss_id = uid + 'SiID'
    # app . key generator
    # app . hash pepper and Salting
    # app . key generator
    self.print(f"APP: generated SiID")
    return ss_id

  def get_vt_id(self, uid):
    vt_id = uid + 'VTInstance'
    # app . key generator
    # app . hash pepper and Salting
    # app . key generator
    self.print(f"APP:{self.app_.id} generated from VTInstance:")
    return vt_id

  def get_web_socket_id(self, uid):
    ws_id = self.app_.id + uid + 'CloudM-Signed'
    # app . key generator
    # app . hash pepper and Salting
    # app . key generator
    self.print(f"APP:generated from webSocketID:")
    return ws_id

  def close_user_instance(self, uid):
    if self.get_si_id(uid) not in self.live_user_instances.keys():
      self.logger.warning("User instance not found")
      return "User instance not found"
    instance = self.live_user_instances[self.get_si_id(uid)]
    self.user_instances[instance['SiID']] = instance['webSocketID']
    self.app_.run_any(
      'db', 'set',
      ['', f"User::Instance::{uid}",
       json.dumps({"saves": instance['save']})])
    if not instance['live']:
      self.save_user_instances(instance)
      self.logger.info("No modules to close")
      return "No modules to close"
    for key, val in instance['live'].items():
      if key.startswith('v-'):
        continue
      try:
        val._on_exit()
      except Exception as e:
        self.logger.error(f"Error closing {key}, {str(e)}")
    del instance['live']
    instance['live'] = {}
    self.logger.info("User instance live removed")
    self.save_user_instances(instance)

  def validate_ws_id(self, command):
    ws_id = command[0]
    self.logger.info(f"validate_ws_id 1 {len(self.user_instances)}")
    if len(self.user_instances) == 0:
      data = self.app_.run_any('db', 'get',
                               [f"user_instances::{self.app_.id}"])
      self.logger.info(f"validate_ws_id 2 {type(data)} {data}")
      if isinstance(data, str):
        try:
          self.user_instances = json.loads(data)
          self.logger.info(Style.GREEN("Valid instances"))
        except Exception as e:
          self.logger.info(Style.RED(f"Error : {str(e)}"))
    self.logger.info(f"validate_ws_id ::{self.user_instances}::")
    for key in list(self.user_instances.keys()):
      value = self.user_instances[key]
      self.logger.info(f"validate_ws_id ::{value == ws_id}:: {key} {value}")
      if value == ws_id:
        return True, key
    return False, ""

  def delete_user_instance(self, uid):
    si_id = self.get_si_id(uid)
    if si_id not in self.user_instances.keys():
      return "User instance not found"
    if si_id in self.live_user_instances.keys():
      del self.live_user_instances[si_id]

    del self.user_instances[si_id]
    self.app_.run_any('db', 'del', ['', f"User::Instance::{uid}"])
    return "Instance deleted successfully"

  def set_user_level(self, command):

    users, keys = [(u['save'], _) for _, u in self.live_user_instances.items()]
    users_names = [u['username'] for u in users]
    for user in users:
      self.print(f"User: {user['username']} level : {user['level']}")

    rot_input = input("Username: ")
    if not rot_input:
      self.print(Style.YELLOW("Please enter a username"))
      return "Please enter a username"
    if rot_input not in users_names:
      self.print(Style.YELLOW("Please enter a valid username"))
      return "Please enter a valid username"

    user = users[users_names.index(rot_input)]

    self.print(Style.WHITE(f"Usr level : {user['level']}"))

    level = input("set level :")
    level = int(level)

    instance = self.live_user_instances[keys[users_names.index(rot_input)]]

    instance['save']['level'] = level

    self.save_user_instances(instance)

    self.print("done")

    return True

  def save_user_instances(self, instance):
    self.logger.info("Saving instance")
    self.user_instances[instance['SiID']] = instance['webSocketID']
    self.live_user_instances[instance['SiID']] = instance
    self.app_.run_any(
      'db', 'set',
      [f"user_instances::{self.app_.id}",
       json.dumps(self.user_instances)])

  def get_instance_si_id(self, command):
    si_id = command[0]
    if si_id in self.live_user_instances:
      return self.live_user_instances[si_id]
    return False

  def get_user_instance_wrapper(self, command):
    return self.get_user_instance(command[0])

  def get_user_instance(self,
                        uid: str,
                        username: str or None = None,
                        token: str or None = None,
                        hydrate: bool = True):
    # Test if an instance exist locally -> instance = set of data a dict

    instance = {
      'save': {
        'uid': uid,
        'level': 0,
        'mods': [],
        'username': username
      },
      'live': {},
      'webSocketID': self.get_web_socket_id(uid),
      'SiID': self.get_si_id(uid),
      'token': token
    }

    if instance['SiID'] in self.live_user_instances.keys():
      instance_live = self.live_user_instances[instance['SiID']]
      if 'live' in instance_live.keys():
        if instance_live['live'] and instance_live['save']['mods']:
          self.logger.info(Style.BLUEBG2("Instance returned from live"))
          return instance_live
        if instance_live['token']:
          instance = instance_live
          instance['live'] = {}

    if instance['SiID'] in self.user_instances.keys(
    ):  # der nutzer ist der server instanz bekannt
      instance['webSocketID'] = self.user_instances[instance['SiID']]

    chash_data = self.app_.run_any('db', 'get', [f"User::Instance::{uid}"])
    if chash_data:
      self.print(chash_data)
      try:
        instance['save'] = json.loads(chash_data)["saves"]
      except Exception as e:
        instance['save'] = chash_data["saves"]
        self.logger.error(Style.YELLOW(f"Error loading instance {e}"))

    self.logger.info(Style.BLUEBG(f"Init mods : {instance['save']['mods']}"))

    self.print(Style.MAGENTA(f"instance : {instance}"))

    #   if no instance is local available look at the upper instance.
    #       if instance is available download and install the instance.
    #   if no instance is available create a new instance
    # upper = instance['save']
    # # get from upper instance
    # # upper = get request ...
    # instance['save'] = upper
    if hydrate:
        instance = self.hydrate_instance(instance)
    self.save_user_instances(instance)

    return instance

  def get_restrictor(self):
    self.print(f"GET Restrictor {self.app_.id}")

    if self.rt is not None:
      return self.rt

    if not self.app_.mod_online("Restrictor"):
      self.rt = self.app_.inplace_load("Restrictor")
      return self.rt
    self.app_.new_ac_mod("Restrictor")
    self.rt = self.app_.AC_MOD
    return self.rt

  def get_virtualization(self):
    self.print(f"GET Virtualization {self.app_.id}")

    if self.vt is not None:
      return self.vt

    if not self.app_.mod_online("VirtualizationTool"):
      self.vt = self.app_.inplace_load("VirtualizationTool")
      return self.vt
    self.app_.new_ac_mod("VirtualizationTool")
    self.vt = self.app_.AC_MOD
    return self.vt

  def hydrate_instance(self, instance):

    # instance = {
    # 'save': {'uid':'INVADE_USER','level': -1, 'mods': []},
    # 'live': {},
    # 'webSocketID': 0000,
    # 'SiID': 0000,
    # }

    vt: VirtualizationTool.Tools = self.get_virtualization()
    rt: Restrictor.Tools = self.get_restrictor()

    chak = instance['live'].keys()
    level = instance['save']['level']

    # app . key generator
    user_instance_name = self.get_vt_id(instance['save']['uid'])

    for mod_name in instance['save']['mods']:

      if mod_name in chak:
        continue

      user_instance_name_mod = mod_name + '-' + user_instance_name

      mod = vt.get_instance(user_instance_name_mod)

      if mod is None:

        self.print(f"Crating v instance : {mod_name}")

        mod = vt.create_instance(user_instance_name_mod, mod_name)

        if mod is None:
          self.print(f"Creating Error Module {mod_name} not found")
          mod = welcome.Tools(
          )  # switch with an 404 mod and an worning message

      self.print(f"Received v instance : {mod.name}")
      tool_data = mod.tools["all"]

      for endpoint in tool_data:
        endpoint_name = endpoint[0]
        endpoint_len = len(endpoint)
        endpoint_level = 0
        endpoint_func_name = endpoint_name
        by = ToolBox_over + self.app_.id
        resid = self.app_.id + "resid"

        if endpoint_name.startswith("api_"):
          endpoint_func_name = endpoint_name[4:]

        if endpoint_len == 3:
          endpoint_level = endpoint[2]

        if endpoint_len >= 4:
          endpoint_level = endpoint[2]
          endpoint_func_name = endpoint[3]

        if endpoint_name.startswith("api_"):
          endpoint_level = -1

        restrict = True
        if isinstance(endpoint_level, int):
          restrict = level < endpoint_level

        if restrict:
          rt.restrict(mod, endpoint_name, by, resid, endpoint_func_name)

      instance['live'][mod_name] = mod
      instance['live']['v-' + mod_name] = user_instance_name_mod

    return instance

  def load_open_file(self):
    self.logger.info("Starting cloudM")
    self.load_file_handler()
    # self.get_version()

  def on_exit(self):
    self.save_file_handler()

  def show_version(self, c):
    self.print(f"Version: ,{self.version} {self.api_version}, {c}")
    return self.version

  def get_version(self):
    version_command = self.get_file_handler(self.keys["URL"])

    url = version_command + "/get/cloudm/run/Version?command=V:" + self.version

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

  def new_module(self, command):
    if len(command) < 3:
      print(f"Command {command} invalid : syntax new module-name ?-fh ?-func")
    self.logger.info(f"Crazing new module : {command[1]}")
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
    mod_name = command[1]
    if '-fh' in command:
      boilerplate = boilerplate.replace('pass', '').replace('# ~ ', '')
      self.logger.info(f"adding FileHandler")
    if '-func' in command:
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

  def upload(self, input_):
    version_command = self.get_file_handler(self.keys["URL"])
    url = "http://127.0.0.1:5000/api/upload-file"
    if version_command is not None:
      url = version_command + "/upload-file"
    try:
      if len(input_) >= 2:
        name = input_[1]
        os.system("cd")
        try:
          with open("./mods/" + name + ".py", "rb").read() as f:
            file_data = str(f, "utf-8")
        except IOError:
          self.print((Style.RED(
            f"File does not exist or is not readable: ./mods/{name}.py")))
          return

        if file_data:
          data = {
            "filename": name,
            "data": file_data,
            "content_type": "file/py"
          }

          try:

            def do_upload():
              r = requests.post(url, json=data)
              self.print(r.status_code)
              if r.status_code == 200:
                self.print("DON")
              self.print(r.content)

            threa = threading.Thread(target=do_upload)
            self.print("Starting upload threading")
            threa.start()

          except Exception as e:
            self.print(
              Style.RED(f"Error uploading (connoting to server) : {e}"))

      else:
        self.print((Style.YELLOW(f"SyntaxError : upload filename | {input_}")))
    except Exception as e:
      self.print(Style.RED(f"Error uploading : {e}"))
      return

  def download(self, input_):
    version_command = self.get_file_handler(self.keys["URL"])
    url = "http://127.0.0.1:5000/get/cloudm/run/download_api_files?command="
    if version_command is not None:
      url = version_command + "/get/cloudm/run/download_api_files?command="
    try:
      if len(input_) >= 1:
        name = input_[1]

        url += name

        try:
          data = requests.get(url).json()["res"]
          if str(data, "utf-8") == f"name not found {name}":
            return False
          with open("./mods/" + name, "a") as f:
            f.write(str(data, "utf-8"))
          self.print("saved file to: " + "./mods" + name)
          return True

        except Exception as e:
          self.print(Style.RED(f"Error download (connoting to server) : {e}"))
      else:
        self.print((Style.YELLOW(f"SyntaxError : download filename {input_}")))
    except Exception as e:
      self.print(Style.RED(f"Error download : {e}"))
    return False

  def download_api_files(self, command, app: App):
    filename = command[0]
    if ".." in filename:
      return "invalid command"
    self.print("download_api_files : ", filename)

    mds = app.get_all_mods()
    if filename in mds:
      self.logger.info(f"returning module {filename}")
      with open("./mods/" + filename + ".py", "rb") as f:
        d = f.read()
      return d

    self.logger.warning(f"Could not found module {filename}")
    return False

  def add_url_con(self, command):
    """
        Adds a url to the list of urls
        """
    if len(command) == 2:
      url = command[1]
    else:
      url = input(
        "Pleas enter URL of CloudM Backend default [https://simpelm.com/api] : "
      )
    if url == "":
      url = "https://simeplm.com/api"
    self.print(Style.YELLOW(f"Adding url : {url}"))
    self.add_to_save_file_handler(self.keys["URL"], url)
    return url

  def create_account(self):
    version_command = self.get_file_handler(self.keys["URL"])
    url = "https://simeplm/app/signup"
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

  def log_in(self, input_):
    version_command = self.get_file_handler(self.keys["URL"])
    url = "https://simeplm/cloudM/login"
    if version_command is not None:
      url = version_command + "/cloudM/login"

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

  def update_core(self, command, app: App):
    self.print("Init Update..")
    if "save" in command:
      os.system("git fetch --all")
      d = f"git branch backup-master-{app.id}-{self.version}-{command[-1]}"
      os.system(d)
      os.system("git reset --hard origin/master")
    out = os.system("git pull")
    app.reset()
    app.remove_all_modules()
    try:
      com = " ".join(sys.orig_argv)
    except AttributeError:
      com = "python3 "
      com += " ".join(sys.argv)

    if "update" not in com:
      print("Restarting...")
      os.system(com)

    if out == 0:
      app.print_ok()
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
    exit(0)

  def create_user(self, command, app: App):

    data = command[0].data

    username = data["username"]
    email = data["email"]
    password = data["password"]
    invitation_key = data["invitation"]

    invitation_data = app.run_any('db', 'get', [invitation_key])

    if invitation_data != "Valid":
      return "Invalid Invitation Key"

    app.run_any('db', 'del', ['', invitation_key])

    uid = str(uuid.uuid4())

    tb_token_jwt = app.run_any('db', 'get', ["jwt-secret-cloudMService"])
    if not tb_token_jwt:
      return "jwt - not found pleas register one"

    if test_if_exists(username, app):
      return "username already exists"

    if test_if_exists(email, app):
      return "email already exists"
    jwt_key = crate_sing_key(username, email, password, uid,
                             gen_token_time({"v": self.version}, 4380),
                             tb_token_jwt, app)
    app.MOD_LIST["db"].tools["set"](
      ["", f"user::{username}::{email}::{uid}", jwt_key])

    self.get_user_instance(uid, username, jwt_key, hydrate=False)

    return self.get_web_socket_id(uid)

  def log_out_user(self, command):
    data = command[0].data
    ws_id = data["webSocketID"]
    valid, key = self.validate_ws_id([ws_id])
    if valid:
      user_instance = self.live_user_instances[key]
      self.logger.info(f"Log out User : {user_instance['save']['username']}")
      for key, mod in user_instance['live'].items():
        self.logger.info(f"Closing {key}")
        if isinstance(mod, str):
          continue
        try:
          mod.on_exit()
        except Exception as e:
          self.logger.error(f"error closing mod instance {key}:{e}")
      self.close_user_instance(user_instance['save']['uid'])

      return "logout"

  def log_in_user(self, command, app: App):
    # if "db" not in list(app.MOD_LIST.keys()):
    #    return "Server has no database module"

    data = command[0].data

    username = data["username"]
    password = data["password"]

    tb_token_jwt = app.run_any('db', 'get', ["jwt-secret-cloudMService"])

    if not tb_token_jwt:
      return "The server is Not Initialized yet if u ar an admin run 'cloudM prep_system_initial'"

    user_data_token = app.run_any('db', 'get', [f"user::{username}::*"])

    user_data: dict = validate_jwt(user_data_token, tb_token_jwt, app.id)

    if type(user_data) is str:
      return user_data

    if "username" not in list(user_data.keys()):
      return "invalid Token"

    if "password" not in list(user_data.keys()):
      return "invalid Token"

    t_username = user_data["username"]
    t_password = user_data["password"]

    if t_username != username:
      return "username does not match"

    if not verify_password(t_password, password):
      return "invalid Password"

    self.print("user login successful : ", t_username)
    jwt_key = crate_sing_key(username, user_data["email"], "",
                             user_data["uid"],
                             gen_token_time({"v": self.version},
                                            4380), tb_token_jwt, app)

    self.get_user_instance(user_data["uid"], username, jwt_key, hydrate=False)

    return self.get_web_socket_id(user_data["uid"])

  def email_waiting_list(self, command, app: App):
    # if "db" not in list(app.MOD_LIST.keys()):
    #    return "Server has no database module"

    data = command[0].data

    email = data["email"]
    imp = ["email_waiting_list", [email]]
    tb_token_jwt = app.run_any('db', 'append_on_set', imp)

    out = "My apologies Unfortunately you could not be added to the Waiting list."
    if tb_token_jwt == imp:
      out = "You will receive an invitation email in a few days"

    if "already in list" in tb_token_jwt:
      out = "You are already in the list, please do not try to add yourself more than once."

    return f"{email}: {out}"

  def validate_jwt(self, command,
                   app: App):  # spec s -> validate token by server x ask max
    res = ''
    self.logger.debug(f'validate_ {type(command[0].data)} {command[0].data}')

    token = command[0].token
    data = command[0].data

    tb_token_jwt = app.run_any('db', 'get', ["jwt-secret-cloudMService"])
    res = validate_jwt(token, tb_token_jwt, app.id)

    if type(res) != str:
      return res
    if res in [
        "InvalidSignatureError", "InvalidAudienceError", "max-p", "no-db"
    ]:
      # go to next kown server to validate the signature and token
      version_command = self.get_file_handler(self.keys["URL"])
      url = "http://194.233.168.22:5000"  # "https://simeplm"
      if version_command is not None:
        url = version_command

      url += "/post/cloudM/run/validate_jwt?command="
      self.print(url)
      j_data = {
        "token": token,
        "data": {
          "server-x": app.id,
          "pasted": data["pasted"] + 1 if 'pasted' in data.keys() else 0,
          "max-p": data["max-p"] - 1 if 'max-p' in data.keys() else 3
        }
      }
      if j_data['data']['pasted'] > j_data['data']['max-p']:
        return "max-p"
      r = requests.post(url, json=j_data)
      res = r.json()

    return res


def test_if_exists(name: str, app: App):
  if "db" not in list(app.MOD_LIST.keys()):
    return "Server has no database module"

  db: MainTool = app.MOD_LIST["db"]

  get_db = db.tools["get"]

  return get_db([f"*::{name}"], app) != ""


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


def gen_token_time(massage: dict, hr_ex):
  massage['exp'] = datetime.now(tz=timezone.utc) + timedelta(hours=hr_ex)
  return massage


def crate_sing_key(username: str,
                   email: str,
                   password: str,
                   uid: str,
                   message: dict,
                   jwt_secret: str,
                   app: App or None = None):
  # Load an RSA key from a JWK dict.
  password = hash_password(password)
  message['username'] = username
  message['password'] = password
  message['email'] = email
  message['uid'] = uid
  # message['aud'] = app.id if app else "-1"

  jwt_ket = jwt.encode(message, jwt_secret, algorithm="HS512")
  return jwt_ket


def get_jwtdata(jwt_key: str, jwt_secret: str, aud):
  try:
    token = jwt.decode(jwt_key,
                       jwt_secret,
                       leeway=timedelta(seconds=10),
                       algorithms=["HS512"],
                       verify=False),  # audience=aud)
    return token
  except jwt.exceptions.InvalidSignatureError:
    return "InvalidSignatureError"
  except jwt.exceptions.InvalidAudienceError:
    return "InvalidAudienceError"


def validate_jwt(jwt_key: str, jwt_secret: str, aud) -> dict or str:
  if not jwt_key:
    return "No JWT Key provided"

  try:
    token = jwt.decode(jwt_key,
                       jwt_secret,
                       leeway=timedelta(seconds=10),
                       algorithms=["HS512"],
                       # audience=aud,
                       do_time_check=True,
                       verify=True)
    return token
  except jwt.exceptions.InvalidSignatureError:
    return "InvalidSignatureError"
  except jwt.exceptions.ExpiredSignatureError:
    return "ExpiredSignatureError"
  except jwt.exceptions.InvalidAudienceError:
    return "InvalidAudienceError"
  except jwt.exceptions.MissingRequiredClaimError:
    return "MissingRequiredClaimError"
  except Exception as e:
    return str(e)


def installer(url):
  if isinstance(url, list):
    for i in url:
      if i.strip().startswith('http'):
        url = i
        break
  with urllib.request.urlopen(url) as response:
    res = response \
        .read()
    soup = BeautifulSoup(res, 'html.parser')
    data = json.loads(extract_json_strings(soup.text)[0].replace('\n', ''))

  # os.mkdir(prfix)
  os.makedirs("mods", exist_ok=True)
  os.makedirs("runable", exist_ok=True)

  for mod_url in tqdm(data["mods"], desc="Mods herunterladen"):
    filename = os.path.basename(mod_url)
    urllib.request.urlretrieve(mod_url, f"mods/{filename}")

  for runnable_url in tqdm(data["runnable"], desc="Runnables herunterladen"):
    filename = os.path.basename(runnable_url)
    urllib.request.urlretrieve(runnable_url, f"runable/{filename}")

  shutil.unpack_archive(data["additional-dirs"], "./")

  # Herunterladen der Requirements-Datei
  requirements_url = data["requirements"]
  requirements_filename = f"{data['Name']}-requirements.txt"
  urllib.request.urlretrieve(requirements_url, requirements_filename)

  # Installieren der Requirements mit pip
  subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", requirements_filename])


def delete_package(url):
  if isinstance(url, list):
    for i in url:
      if i.strip().startswith('http'):
        url = i
        break
  with urllib.request.urlopen(url) as response:
    res = response \
        .read()
    soup = BeautifulSoup(res, 'html.parser')
    data = json.loads(extract_json_strings(soup.text)[0].replace('\n', ''))

  for mod_url in tqdm(data["mods"], desc="Mods löschen"):
    filename = os.path.basename(mod_url)
    file_path = os.path.join("mods", filename)
    if os.path.exists(file_path):
      os.remove(file_path)

  for runnable_url in tqdm(data["runnable"], desc="Runnables löschen"):
    filename = os.path.basename(runnable_url)
    file_path = os.path.join("runnable", filename)
    if os.path.exists(file_path):
      os.remove(file_path)

  additional_dir_path = os.path.join("mods",
                                     os.path.basename(data["additional-dirs"]))
  if os.path.exists(additional_dir_path):
    shutil.rmtree(additional_dir_path)

  # Herunterladen der Requirements-Datei
  requirements_url = data["requirements"]
  requirements_filename = f"{data['Name']}-requirements.txt"
  urllib.request.urlretrieve(requirements_url, requirements_filename)

  # Deinstallieren der Requirements mit pip
  with tempfile.NamedTemporaryFile(mode="w",
                                   delete=False) as temp_requirements_file:
    with open(requirements_filename) as original_requirements_file:
      for line in original_requirements_file:
        package_name = line.strip().split("==")[0]
        temp_requirements_file.write(f"{package_name}\n")

    temp_requirements_file.flush()
    subprocess.check_call([
      sys.executable, "-m", "pip", "uninstall", "-y", "-r",
      temp_requirements_file.name
    ])

  # Löschen der heruntergeladenen Requirements-Datei
  os.remove(requirements_filename)
  os.remove(temp_requirements_file.name)
