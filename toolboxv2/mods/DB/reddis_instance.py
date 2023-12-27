import redis

from toolboxv2 import Result, get_logger
from .types import AuthenticationTypes


class MiniRedis:

    auth_type = AuthenticationTypes.Uri

    def __init__(self):
        self.encoding = 'utf-8'
        self.rcon = None

    def initialize(self, uri: str):
        try:
            self.rcon = redis.from_url(uri)
            return True
        except Exception as e:
            return e

    def get(self, key: str) -> Result:
        data = []
        data_info = ""
        if self.rcon is None:
            return Result.default_user_error(info='Pleas run first-redis-connection to connect to a reddis instance')

        if key == 'all':
            data_info = "Returning all data available "
            for key_ in self.rcon.scan_iter():
                val = self.rcon.get(key_)
                data.append((key_, val))

        elif key == "all-k":
            data_info = "Returning all keys "
            for key_ in self.rcon.scan_iter():
                data.append(key_)
        else:
            data_info = "Returning subset of keys "
            for key_ in self.rcon.scan_iter(key):
                val = self.rcon.get(key_)
                data.append(val)

        if not data:
            return Result.default_internal_error(info=f"No data found for key {key}")

        return Result.ok(data=data, data_info=data_info + key)

    def set(self, key: str, value) -> Result:
        if self.rcon is None:
            return Result.default_user_error(info='Pleas run first-redis-connection to connect to a reddis instance')
        try:
            self.rcon.set(key, value)
            return Result.ok()
        except TimeoutError as e:
            get_logger().error(f"Timeout by redis DB : {e}")
            return Result.default_internal_error(info=e)
        except Exception as e:
            return Result.default_internal_error(info="Fatal Error: "+str(e))

    def append_on_set(self, key: str, value: list) -> Result:

        if self.rcon is None:
            return Result.default_internal_error(info='Pleas run first-redis-connection')

        val = self.rcon.get(key)

        if val:
            val = eval(val)
            for new_val in value:
                if new_val in val:
                    return Result.default_user_error(info="Error value: " + str(new_val) + "already in list")
                val.append(new_val)
        else:
            val = value

        self.rcon.set(key, str(val))

        return Result.ok()

    def delete(self, key, matching=False) -> Result:
        if self.rcon is None:
            return Result.default_user_error(info='Pleas run first-redis-connection to connect to a reddis instance')

        del_list = []
        n = 0

        if matching:
            for key_ in self.rcon.scan_iter():
                # Check if the key contains the substring
                if key_ in str(key, 'utf-8'):
                    n += 1
                    # Delete the key if it contains the substring
                    v = self.rcon.delete(key_)
                    del_list.append((key_, v))
        else:
            v = self.rcon.delete(key)
            del_list.append((key, v))
            n += 1

        return Result.ok(data=del_list, data_info=f"Data deleted successfully removed {n} items")
