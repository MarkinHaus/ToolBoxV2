import os
from abc import ABC
from typing import Any

from toolboxv2 import MainTool, FileHandler, Result, get_app
from .reddis_instance import MiniRedis
from .local_instance import MiniDictDB
from .types import DatabaseModes, AuthenticationTypes
from toolboxv2.utils.types import ToolBoxInterfaces
from ...utils.cryp import Code

Name = "DB"


class DB(ABC):

    def get(self, query: str) -> Result:
        """get data"""

    def set(self, query:str, value) -> Result:
        """set data"""

    def append_on_set(self, query:str, value) -> Result:
        """append set data"""

    def delete(self, query:str, matching=False) -> Result:
        """delete data"""


class Tools(MainTool, FileHandler):
    version = "0.0.2"

    export = get_app(from_="DB.EXPORT").tb

    def __init__(self, app=None):
        self.name = Name
        self.logs = app.logger if app else None
        self.color = "YELLOWBG"

        self.keys = {"url": "redis:url~"}
        self.encoding = 'utf-8'

        self.data_base: MiniRedis or MiniDictDB or DB or None = None
        self.mode = DatabaseModes.LC
        self.url = None
        self.passkey = None
        self.user_name = None
        self.password = None

        MainTool.__init__(self,
                          v=self.version,
                          name=self.name,
                          logs=self.logs,
                          color=self.color)

    @export(
        mod_name=Name,
        name="Version",
        version=version
    )
    def get_version(self):
        return self.version

    @export(
        mod_name=Name,
        helper="Get data from an Database instance",
        version=version,
        interface=ToolBoxInterfaces.internal
    )
    def get(self, query: str) -> Result:

        if self.data_base is None:
            return Result.default_internal_error(info="No database connection")

        query = self.crytography(query)

        if self.mode.value == "LOCAL_DICT" or self.mode.value == "LOCAL_REDDIS" or self.mode.value == "REMOTE_REDDIS":
            return self.data_base.get(query)

        if self.mode.value == "REMOTE_DICT":
            return Result.custom_error(info="Not Implemented yet")  # TODO: add remote dict storage

        return Result.default_internal_error(info="Database is not configured")

    @export(
        mod_name=Name,
        helper="Set data to an Database instance",
        version=version,
        interface=ToolBoxInterfaces.internal
    )
    def set(self, query: str, data: Any) -> Result:
        if self.data_base is None:
            return Result.default_internal_error(info="No database connection")

        query = self.crytography(query)
        data = self.crytography(data)

        if self.mode.value == "LOCAL_DICT" or self.mode.value == "LOCAL_REDDIS" or self.mode.value == "REMOTE_REDDIS":
            return self.data_base.set(query, data)

        if self.mode.value == "REMOTE_DICT":
            return Result.custom_error(info="Not Implemented yet")  # TODO: add remote dict storage

        return Result.default_internal_error(info="Database is not configured")

    @export(
        mod_name=Name,
        helper="Delete data from an Database instance",
        version=version,
        interface=ToolBoxInterfaces.internal
    )
    def delete(self, query: str, matching=False) -> Result:
        if self.data_base is None:
            return Result.default_internal_error(info="No database connection")

        query = self.crytography(query)

        if self.mode.value == "LOCAL_DICT" or self.mode.value == "LOCAL_REDDIS" or self.mode.value == "REMOTE_REDDIS":
            try:
                return self.data_base.delete(query, matching)
            except ValueError and KeyError:
                return Result.default_user_error(info=f"'{query=}' not in database")

        if self.mode.value == "REMOTE_DICT":
            return Result.custom_error(info="Not Implemented yet")  # TODO: add remote dict storage

        return Result.default_internal_error(info="Database is not configured")

    @export(
        mod_name=Name,
        helper="append data to an Database instance subset",
        version=version,
        interface=ToolBoxInterfaces.internal
    )
    def append_on_set(self, query: str, data: Any) -> Result:
        if self.data_base is None:
            return Result.default_internal_error(info="No database connection")

        query = self.crytography(query)
        data = self.crytography(data)

        if self.mode.value == "LOCAL_DICT" or self.mode.value == "LOCAL_REDDIS" or self.mode.value == "REMOTE_REDDIS":
            return self.data_base.append_on_set(query, data)

        if self.mode.value == "REMOTE_DICT":
            return Result.custom_error(info="Not Implemented yet")  # TODO: add remote dict storage

        return Result.default_internal_error(info="Database is not configured")

    @export(mod_name=Name, initial=True, helper="init database")
    def initialize_database(self) -> Result:
        if self.data_base is not None:
            return Result.default_user_error(info="Database is already configured")

        if self.mode.value == DatabaseModes.LC.value:
            self.data_base = MiniDictDB()
        elif self.mode.value == DatabaseModes.LR.value:
            self.data_base = MiniRedis()
        elif self.mode.value == DatabaseModes.RR.value:
            self.data_base = MiniRedis()
        else:
            return Result.default_internal_error(info="Not implemented")

        self._autoresize()

        self.logger.info(f"Running DB in mode : {self.mode.value}")

    def _autoresize(self):

        if self.data_base is None:
            return Result.default_internal_error(info="No data_base instance specified")

        auth = self.data_base.auth_type
        evaluation = "An unknown authentication error occurred"

        if auth.value == AuthenticationTypes.Uri.name:
            url = self.url
            if self.url is None:
                url = os.getenv("DB_CONNECTION_URI")
            evaluation = self.data_base.initialize(url)

        if auth.value == AuthenticationTypes.PassKey.name:
            passkey = self.passkey
            if self.passkey is None:
                passkey = os.getenv("DB_PASSKEY")
            evaluation = self.data_base.initialize(passkey)

        if auth.value == AuthenticationTypes.UserNamePassword.name:
            user_name = self.user_name
            if self.user_name is None:
                user_name = os.getenv("DB_USERNAME")
            evaluation = self.data_base.initialize(user_name, input(":Password:"))

        if isinstance(evaluation, bool) and evaluation:
            return Result.ok()
        return Result.default_internal_error(info=evaluation)

    @staticmethod
    def crytography(data) -> str:
        # Code().encode_code(data)
        return data

    @export(mod_name=Name, interface=ToolBoxInterfaces.native)
    def edit_programmable(self):
        pass

    @export(mod_name=Name, interface=ToolBoxInterfaces.cli)
    def edit_cli(self):
        pass

    @export(mod_name=Name, interface=ToolBoxInterfaces.remote)
    def edit_dev_web_ui(self):
        pass

    # TODO: init db default local save and load

    # TODO: switch to reddis or other data base

    # TODO: save sensitive data to database {passkey,username,password,url}

    # TODO: enable crytography for data base
