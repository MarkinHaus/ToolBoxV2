import abc
import os
from abc import ABC
from typing import Any, Optional

from dotenv import load_dotenv

from toolboxv2 import FileHandler, MainTool, Result, get_app
from toolboxv2.utils.security.cryp import Code
from toolboxv2.utils.system.types import ToolBoxInterfaces

from .blob_instance import BlobDB
from .local_instance import MiniDictDB
from .reddis_instance import MiniRedis
from .types import AuthenticationTypes, DatabaseModes

Name = "DB"
export = get_app(from_="DB.EXPORT").tb

load_dotenv(verbose=True)

def pre_function(*args, **kwargs) -> tuple[list, dict]:
    # Verarbeitung der args mit der encode_code-Methode
    encoded_args = map(lambda x: str(x) if type(x).__name__ in ['str', 'int', 'dict', 'list', 'tupel'] else x, args)

    # Verarbeitung der kwargs mit der encode_code-Methode
    encoded_kwargs = {k: str(v) if k == 'data' else v for k, v in kwargs.items()}

    return list(encoded_args), encoded_kwargs


def post_function(result: Result) -> Result:
    if result.get() is None:
        return result
    if result.is_error():
        return result
    return result

# decoded_data = []
# if isinstance(result.get(), list):
#     for data in result.get():
#         decoded_data.append(Code().decode_code(data))
#
# else:
#     decoded_data = Code().decode_code(result.get())
#
# print("Decoded data", decoded_data)

# if decoded_data == "Error decoding":
#     return result.lazy_return('intern', result, info=f"post fuction decoding error")

# return result.ok(data=decoded_data, data_info=result.result.data_info, info=result.info.help_text,
# interface=result.result.data_to)

class DB(ABC):
    @abc.abstractmethod
    def get(self, query: str) -> Result:
        """get data"""

    @abc.abstractmethod
    def set(self, query: str, value) -> Result:
        """set data"""

    @abc.abstractmethod
    def append_on_set(self, query: str, value) -> Result:
        """append set data"""

    @abc.abstractmethod
    def delete(self, query: str, matching=False) -> Result:
        """delete data"""

    @abc.abstractmethod
    def if_exist(self, query: str) -> bool:
        """return True if query exists"""

    @abc.abstractmethod
    def exit(self) -> Result:
        """Close DB connection and optional save data"""


class Tools(MainTool, FileHandler):
    version = "0.0.3"

    def __init__(self, app=None):
        self.name = "DB"
        self.color = "YELLOWBG"

        self.keys = {"mode": "db~mode~~:"}
        self.encoding = 'utf-8'

        self.data_base: Optional[MiniRedis , MiniDictDB , DB , None] = None
        self.mode = DatabaseModes.crate(
            os.getenv("DB_MODE_KEY", "LC") if 'test' not in get_app("DB_MODE_KEY").id else os.getenv("DB_MODE_KEY_TEST",
                                                                                                     "LC"))
        self.url = None
        self.passkey = None
        self.user_name = None
        self.password = None

        MainTool.__init__(self,
                          load=self.initialize_database,
                          v=self.version,
                          name=self.name,
                          color=self.color,
                          on_exit=self.close_db)

    @export(
        mod_name=Name,
        name="Version",
        version=version,
    )
    def get_version(self):
        return self.version

    @export(
        mod_name=Name,
        helper="Get data from an Database instance",
        version=version,
        interface=ToolBoxInterfaces.internal,
        post_compute=post_function,
        test=False,
    )
    def get(self, query: str) -> Result:

        if self.data_base is None:
            return Result.default_internal_error(info="No database connection")


        return self.data_base.get(query)


    @export(
        mod_name=Name,
        helper="Test if an key is in the Database instance",
        version=version,
        interface=ToolBoxInterfaces.internal,
        post_compute=post_function,
        test=False,
    )
    def if_exist(self, query: str) -> Result:

        if self.data_base is None:
            return Result.default_internal_error(info="No database connection")

        return Result.ok(data=self.data_base.if_exist(query)).set_origin(f"{self.mode.value}.DB.if_exist")

    @export(
        mod_name=Name,
        helper="Set data to an Database instance",
        version=version,
        interface=ToolBoxInterfaces.internal,
        pre_compute=pre_function,
        test=False,
    )
    def set(self, query: str, data: Any) -> Result:
        if self.data_base is None:
            return Result.default_internal_error(info="No database connection")

        if data is None:
            return Result.default_user_error(info="None value is not valid value must have a to str & be serialise")

        return self.data_base.set(query, data)

    @export(
        mod_name=Name,
        helper="Delete data from an Database instance",
        version=version,
        interface=ToolBoxInterfaces.internal,
        test=False,
    )
    def delete(self, query: str, matching:bool=False) -> Result:

        if self.data_base is None:
            return Result.default_internal_error(info="No database connection")

        if self.mode.value == "LOCAL_DICT" or self.mode.value == "LOCAL_REDDIS" or self.mode.value == "REMOTE_REDDIS":
            try:
                return self.data_base.delete(query, matching)
            except ValueError:
                return Result.default_user_error(info=f"'{query=}' not in database ValueError")
            except KeyError:
                return Result.default_user_error(info=f"'{query=}' not in database KeyError")

        return Result.default_internal_error(info="Database is not configured")

    @export(
        mod_name=Name,
        helper="append data to an Database instance subset",
        version=version,
        interface=ToolBoxInterfaces.internal,
        test=False,
    )
    def append_on_set(self, query: str, data: Any) -> Result:
        if self.data_base is None:
            return Result.default_internal_error(info="No database connection")

        if data is None:
            return Result.default_user_error(info="None value is not valid value must have a to str & be serialise")

        return self.data_base.append_on_set(query, data)

    def initialize_database(self) -> Result:
        if self.data_base is not None:
            return Result.default_user_error(info="Database is already configured")

        if self.mode.value == DatabaseModes.LC.value:
            self.data_base = MiniDictDB()
        elif self.mode.value == DatabaseModes.CB.value:
            self.data_base = BlobDB()
        elif self.mode.value == DatabaseModes.LR.value or self.mode.value == DatabaseModes.RR.value:
            self.data_base = MiniRedis()
        else:
            return Result.default_internal_error(info="Not implemented")
        a = self._autoresize()
        if a.log(prifix="initialize_database: ").is_error():
            raise RuntimeError("DB Autoresize Error " + a.print(show=False))

        self.app.logger.info(f"Running DB in mode : {self.mode.value}")
        self.print(f"Running DB.{self.spec} in mode : {self.mode.value}")
        print(self.get('all-k'))
        return Result.ok()

    def _autoresize(self):

        if self.data_base is None:
            return Result.default_internal_error(info="No data_base instance specified")
        auth = self.data_base.auth_type
        evaluation = Result.default_internal_error(info="An unknown authentication error occurred")
        if auth.value == AuthenticationTypes.Uri.value:
            url = self.url
            if self.url is None:
                url = os.getenv("DB_CONNECTION_URI")
            if url is None:
                raise ValueError("Could not find DB connection URI in environment variable DB_CONNECTION_URI")
            evaluation = self.data_base.initialize(url)
        if auth.value == AuthenticationTypes.PassKey.value:
            passkey = self.passkey
            if self.passkey is None:
                passkey = os.getenv("DB_PASSKEY")

            if passkey is None:
                raise ValueError("Could not find DB connection passkey in environment variable DB_PASSKEY")
            evaluation = self.data_base.initialize(passkey)
        if auth.value == AuthenticationTypes.UserNamePassword.value:
            user_name = self.user_name
            if self.user_name is None:
                user_name = os.getenv("DB_USERNAME")

            if user_name is None:
                raise ValueError("Could not find DB connection user_name in environment variable DB_USERNAME")
            evaluation = self.data_base.initialize(user_name, input(":Password:"))
        if auth.value == AuthenticationTypes.location.value:
            local_key = Code.DK()() #self.app.config_fh.get_file_handler("LocalDbKey")
            if self.mode.value == DatabaseModes.CB.value:
                evaluation = self.data_base.initialize()
            if self.mode.value == DatabaseModes.LC.value:
                evaluation = self.data_base.initialize(self.app.data_dir, local_key)
        if not evaluation.is_error():
            return Result.ok()
        return Result.default_internal_error(info=evaluation.get())

    def close_db(self) -> Result:
        if self.data_base is None:
            return Result.default_user_error(info="Database is not configured therefor cand be closed")
        result = self.data_base.exit().print()
        self.data_base = None
        return result

    @export(mod_name=Name, interface=ToolBoxInterfaces.native, samples=[{"mode": DatabaseModes.crate("LC")}])
    def edit_programmable(self, mode: DatabaseModes = DatabaseModes.LC):
        if mode is None:
            self.app.logger.warning("No mode parsed")
            return Result.default_user_error(info="mode is None")
        if mode.name not in ["LC", "LR", "RR", "CB"]:
            return Result.default_user_error(info=f"Mode not supported used : {mode.name}")
        self.mode = mode
        if self.data_base is None:
            return Result.ok(data=self.initialize_database()).lazy_return(data=f"mode change to {mode}")
        return self.close_db().lazy_return("intern",
                                           data=self.initialize_database()
                                           ).lazy_return(data=f"mode change to {mode}")

    @export(mod_name=Name, interface=ToolBoxInterfaces.cli,
        samples=[{"mode": "LC"}])
    def edit_cli(self, mode: str = "LC"):
        if mode is None:
            self.app.logger.warning("No mode parsed")
            return Result.default_user_error(info="mode is None")
        if mode not in ["LC", "LR", "RR", "CB"]:
            return Result.default_user_error(info="Mode not supported")
        self.mode = DatabaseModes.crate(mode)
        if self.data_base is None:
            return Result.ok(data=self.initialize_database()).lazy_return(data=f"mode change to {mode}")
        return self.close_db().lazy_return("intern",
                                           data=self.initialize_database()
                                           ).lazy_return(data=f"mode change to {mode}")

    @export(mod_name=Name, interface=ToolBoxInterfaces.remote, api=False, helper="Avalabel modes: LC CB LR RR",
        samples=[{"mode": "LC"}])
    def edit_dev_web_ui(self, mode: str = "LC"):
        if mode is None:
            self.app.logger.warning("No mode parsed")
            return Result.default_user_error(info="mode is None")
        if mode not in ["LC", "LR", "RR", "CB"]:
            return Result.default_user_error(info="Mode not supported")
        self.mode = DatabaseModes.crate(mode)
        if self.data_base is None:
            return Result.ok(data=self.initialize_database()).lazy_return(data=f"mode change to {mode}")
        return self.close_db().lazy_return("intern",
                                           data=self.initialize_database()
                                           ).lazy_return(data=f"mode change to {mode}")


    @export(mod_name=Name, interface=ToolBoxInterfaces.internal, test_only=True)
    def test(self):
        test_key = "test_key"
        test_key2 = "test_key"
        test_data = {"test": "data"}
        test_data2 = ["test", "data"]

        result = self.set(test_key, test_data)
        assert result.is_ok()

        result = self.get(test_key)
        assert result.is_ok()
        assert result.get() == test_data

        result = self.append_on_set(test_key2, test_data2)
        assert result.is_ok()
        result = self.append_on_set(test_key2, ["test","data2"])
        assert result.is_ok()
        result = self.get(test_key2)
        assert result.is_ok()
        assert result.get() == ["test", "data", "test2"]

        result = self.delete(test_key2)
        assert result.is_ok()
        result = self.delete(test_key)
        assert result.is_ok()

        result = self.get(test_key)
        assert result.is_error()

        result = self.get(test_key2)
        assert result.is_error()


