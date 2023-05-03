import binascii
import hashlib
import logging
import os
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
from json import JSONDecoder
from urllib import request

from bs4 import BeautifulSoup
from tqdm import tqdm

import jwt
import requests
import re
from toolboxv2 import MainTool, FileHandler, App, Style
from toolboxv2.util.Style import extract_json_strings


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        t0 = time.time()
        self.version = "0.0.1"
        self.api_version = "404"
        self.name = "cloudM"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "CYAN"
        self.keys = {
            "URL": "comm-vcd~~",
            "TOKEN": "comm-tok~~",
        }
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["NEW", "crate a boilerplate file to make a new mod", "add is case sensitive",
                     "flags -fh for FileHandler", "-func for functional bas Tools els class based"],
                    ["download", "download a mod from MarkinHaus server", "add is case sensitive"],
                    ["#update-core", "update ToolBox from (git) MarkinHaus server ",
                     "add is case sensitive"],
                    ["upload", "upload a mod to MarkinHaus server", "add is case sensitive"],
                    ["first-web-connection", "set up a web connection to MarkinHaus"],
                    ["create-account", "create a new account"],
                    ["login", "login with Username & password"],
                    ["create_user", "create a new user - api instance"],
                    ["validate_jwt", "validate a  user - api instance"],
                    ["log_in_user", "log_in user - api instance"],
                    ["download_api_files", "download mods"],
                    ["get-init-config", "get-init-config mods"],
                    ["mod-installer", "installing mods via json url"],
                    ["mod-remover", "remover mods via json url"],
                    ],
            "name": "cloudM",
            "Version": self.show_version,
            "NEW": self.new_module,
            "upload": self.upload,
            "download": self.download,
            "first-web-connection": self.add_url_con,
            "create-account": self.create_account,
            "login": self.log_in,
            "create_user": self.create_user,
            "log_in_user": self.log_in_user,
            "validate_jwt": self.validate_jwt,
            "download_api_files": self.download_api_files,
            "#update-core": self.update_core,
            "mod-installer": installer,
            "mod-remover": delete_package,
        }

        self.logger.info("init FileHandler cloudM")
        t1 = time.time()
        FileHandler.__init__(self, "modules.config", app.id if app else __name__, self.keys, {
            "URL": '"https://simpelm.com/api"',
            "TOKEN": '"~tok~"',
        })
        self.logger.info(f"Time to initialize FileHandler {time.time() - t1}")

        t1 = time.time()
        self.logger.info("init MainTool cloudM")
        MainTool.__init__(self, load=self.load_open_file, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logger, color=self.color, on_exit=self.on_exit)

        self.logger.info(f"Time to initialize MainTool {time.time() - t1}")
        self.logger.info(f"Time to initialize Tools {self.name} {time.time() - t0}")

    def load_open_file(self):
        self.logger.info("Starting cloudM")
        self.load_file_handler()
        self.get_version()

    def on_exit(self):
        self.save_file_handler()

    def show_version(self, c):
        self.print(f"Version: , {self.version}, {self.api_version}, {c}")
        return self.version

    def get_version(self):
        version_command = self.get_file_handler(self.keys["URL"])

        url = version_command + "/get/cloudm/run/Version?command=V:" + self.version

        try:
            self.api_version = requests.get(url).json()["res"]
            self.print(f"API-Version: {self.api_version}")
        except Exception:
            self.print(Style.RED(f" Error retrieving version from {url}\n\t run : cloudM first-web-connection\n"))
            self.logger.error(f"Error retrieving version from {url}")

    def new_module(self, command):
        if len(command) >= 1:
            print(f"Command {command} invalid : syntax new module-name ?-fh ?-func")
        self.logger.info(f"Crazing new module : {command[1]}")
        boilerplate = """import logging
from toolboxv2 import MainTool, FileHandler, App
from toolboxv2.Style import Style


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
        pass

    def on_exit(self):
        self.logger.info(f"Closing NAME")
        # ~ self.save_file_handler()
        pass

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
        if command in ['-fh']:
            boilerplate = boilerplate.replace('pass', '').replace('# ~ ', '')

            self.logger.info(f"adding FileHandler")
        if command in ['-func']:
            boilerplate += helper_functions_func
            self.logger.info(f"adding functional based")
        else:
            boilerplate += helper_functions_class
            self.logger.info(f"adding Class based")
        self.print(f"Test existing {self.api_version=} ", end='')

        self.logger.info(f"Testing connection")
        self.get_version()

        if self.api_version != '404':
            if self.download(["", mod_name]):
                self.print(Style.Bold(Style.RED("MODULE exists-on-api pleas use a other name")))
                return False

        self.print("NEW MODULE: " + mod_name, end=" ")
        if os.path.exists(f"mods/" + mod_name + ".py") or os.path.exists(f"mods_dev/" + mod_name + ".py"):
            self.print(Style.Bold(Style.RED("MODULE exists pleas use a other name")))
            return False

        # fle = Path("mods_dev/" + mod_name + ".py")
        # fle.touch(exist_ok=True)
        with open(f"mods_dev/" + mod_name + ".py", "wb") as mod_file:
            mod_file.write(
                bytes(boilerplate.replace('NAME', mod_name), 'ISO-8859-1')
            )

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
                    self.print((Style.RED(f"File does not exist or is not readable: ./mods/{name}.py")))
                    return

                if file_data:
                    data = {"filename": name,
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
                        self.print(Style.RED(f"Error uploading (connoting to server) : {e}"))

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
            url = input("Pleas enter URL of CloudM Backend default [https://simpelm.com/api] : ")
        if url == "":
            url = "https://simeplm.com/api"
        self.print(Style.YELLOW(f"Adding url : {url}"))
        self.add_to_save_file_handler(self.keys["URL"], url)
        return url

    def create_account(self):
        version_command = self.get_file_handler(self.keys["URL"])
        url = "https://simeplm/cloudM/create_acc_mhs"
        if version_command is not None:
            url = version_command + "/cloudM/create_acc_mhs"
        # os.system(f"start {url}")

        try:
            import webbrowser
            webbrowser.open(url, new=0, autoraise=True)
        except Exception:
            print("error")
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

            data = {"username": username,
                    "password": password}

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
            self.print(Style.RED(f"ERROR: {input_} len {len(input_)} != 3 | login username password"))

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
            print("if you changet local files type $ cloudM #update-core save {name}")
            print("your changes will be saved to a branch named : backup-master-{app.id}-{self.version}-{name}")
            print("you can apply yur changes after the update with:\ngit stash\ngit stash pop")

        if out == -1:
            os.system("git fetch --all")
            os.system("git reset --hard origin/master")
        exit(0)

    def create_user(self, command, app: App):
        if "db" not in list(app.MOD_LIST.keys()):
            return "Server has no database module"

        data = command[0].data

        username = data["username"]
        email = data["email"]
        password = data["password"]

        uid = str(uuid.uuid4())

        tb_token_jwt = app.run_any('db', 'get', ["jwt-secret-cloudMService"])
        if not tb_token_jwt:
            return "jwt - not found pleas register one"

        if test_if_exists(username, app):
            return "username already exists"

        if test_if_exists(email, app):
            return "email already exists"
        jwt_key = crate_sing_key(username, email, password, uid, gen_token_time({"v": self.version}, 4380),
                                 tb_token_jwt, app)
        app.MOD_LIST["db"].tools["set"](["", f"user::{username}::{email}::{uid}", jwt_key])
        return jwt_key

    def log_in_user(self, command, app: App):
        if "db" not in list(app.MOD_LIST.keys()):
            return "Server has no database module"

        data = command[0].data

        username = data["username"]
        password = data["password"]

        tb_token_jwt = app.run_any('db', 'get', ["jwt-secret-cloudMService"])

        if not tb_token_jwt:
            return "jwt scret - not found pleas register one"

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

        return crate_sing_key(username, user_data["email"], "", user_data["uid"],
                              gen_token_time({"v": self.version}, 4380),
                              tb_token_jwt, app)

    def validate_jwt(self, command, app: App):  # spec s -> validate token by server x ask max
        res = ''
        self.logger.debug(f'validate_ {type(command[0].data)} {command[0].data}')

        token = command[0].token
        data = command[0].data

        tb_token_jwt = app.run_any('db', 'get', ["jwt-secret-cloudMService"])
        res = validate_jwt(token, tb_token_jwt, app.id)

        if type(res) != str:
            return res
        if res in ["InvalidSignatureError", "InvalidAudienceError", "max-p", "no-db"]:
            # go to next kown server to validate the signature and token
            version_command = self.get_file_handler(self.keys["URL"])
            url = "http://194.233.168.22:5000"  # "https://simeplm"
            if version_command is not None:
                url = version_command

            url += "/post/cloudM/run/validate_jwt?command="
            self.print(url)
            j_data = {"token": token,
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


def test_if_exists(self, name: str, app: App):
    if "db" not in list(app.MOD_LIST.keys()):
        return "Server has no database module"

    db: MainTool = app.MOD_LIST["db"]

    get_db = db.tools["get"]

    return get_db([f"*::{name}"], app) != ""


# Create a hashed password
def hash_password(password):
    """Hash a password for storing."""
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'),
                                  salt, 100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt + pwdhash).decode('ascii')


# Check hashed password validity
def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    salt = stored_password[:64]
    stored_password = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha512',
                                  provided_password.encode('utf-8'),
                                  salt.encode('ascii'),
                                  100000)
    pwdhash = binascii.hexlify(pwdhash).decode('ascii')
    return pwdhash == stored_password


def gen_token_time(massage: dict, hr_ex):
    massage['exp'] = datetime.now(tz=timezone.utc) + timedelta(hours=hr_ex)
    return massage


def crate_sing_key(username: str, email: str, password: str, uid: str, message: dict, jwt_secret: str,
                   app: App or None = None):
    # Load an RSA key from a JWK dict.
    password = hash_password(password)
    message['username'] = username
    message['password'] = password
    message['email'] = email
    message['uid'] = uid
    message['aud'] = app.id if app else "-1"

    jwt_ket = jwt.encode(message, jwt_secret, algorithm="HS512")
    return jwt_ket


def get_jwtdata(jwt_key: str, jwt_secret: str, aud):
    try:
        token = jwt.decode(jwt_key, jwt_secret, leeway=timedelta(seconds=10),
                           algorithms=["HS512"], verify=False, audience=aud)
        return token
    except jwt.exceptions.InvalidSignatureError:
        return "InvalidSignatureError"
    except jwt.exceptions.InvalidAudienceError:
        return "InvalidAudienceError"


def validate_jwt(jwt_key: str, jwt_secret: str, aud) -> dict or str:
    if not jwt_key:
        return "No JWT Key provided"

    try:
        token = jwt.decode(jwt_key, jwt_secret, leeway=timedelta(seconds=10),
                           algorithms=["HS512"], audience=aud, do_time_check=True, verify=True)
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


# CLOUDM #update-core
# API_MANAGER start-api main a

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
        urllib.request.urlretrieve(runnable_url,  f"runable/{filename}")

    shutil.unpack_archive(data["additional-dirs"],  "./")

    # Herunterladen der Requirements-Datei
    requirements_url = data["requirements"]
    requirements_filename = f"{data['Name']}-requirements.txt"
    urllib.request.urlretrieve(requirements_url, requirements_filename)

    # Installieren der Requirements mit pip
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_filename])

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

    additional_dir_path = os.path.join("mods", os.path.basename(data["additional-dirs"]))
    if os.path.exists(additional_dir_path):
        shutil.rmtree(additional_dir_path)

    # Herunterladen der Requirements-Datei
    requirements_url = data["requirements"]
    requirements_filename = f"{data['Name']}-requirements.txt"
    urllib.request.urlretrieve(requirements_url, requirements_filename)

    # Deinstallieren der Requirements mit pip
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_requirements_file:
        with open(requirements_filename) as original_requirements_file:
            for line in original_requirements_file:
                package_name = line.strip().split("==")[0]
                temp_requirements_file.write(f"{package_name}\n")

        temp_requirements_file.flush()
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-r", temp_requirements_file.name])

    # Löschen der heruntergeladenen Requirements-Datei
    os.remove(requirements_filename)
    os.remove(temp_requirements_file.name)
