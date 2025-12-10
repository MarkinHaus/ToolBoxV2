import binascii
import hashlib
import logging
import os
import time

import requests

from toolboxv2 import FileHandler, MainTool, Style, get_app

from toolboxv2.utils.system.state_system import find_highest_zip_version
from .UserInstances import UserInstances

Name = 'CloudM'
version = "0.0.4"
export = get_app(f"{Name}.EXPORT").tb
no_test = export(mod_name=Name, test=False, version=version)
to_api = export(mod_name=Name, api=True, version=version)


class Tools(MainTool, FileHandler):
    version = version

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
            ],
            "name": "cloudM",
            "Version": self.get_version,
            "show_version": self.s_version,
            "get_mod_snapshot": self.get_mod_snapshot,
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

    async def load_open_file(self):
        self.logger.info("Starting cloudM")
        self.load_file_handler()
        from toolboxv2.mods.Minu.examples import initialize
        initialize(self.app)
        await self.app.session.login()

    def s_version(self):
        return self.version

    def on_exit(self):
        self.save_file_handler()

    def get_version(self):  # Add root and upper and controll comander pettern
        version_command = self.app.config_fh.get_file_handler("provider::")

        url = version_command + "/api/Cloudm/show_version"

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
        return self.version

    def get_mod_snapshot(self, mod_name):
        if mod_name is None:
            return None
        self.print("")
        return find_highest_zip_version(mod_name, version_only=True)


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
