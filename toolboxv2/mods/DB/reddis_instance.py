import time
from typing import Optional

import redis

from toolboxv2 import Result, get_logger
from .types import AuthenticationTypes


class MiniRedis:
    auth_type = AuthenticationTypes.Uri

    def __init__(self):
        self.encoding = 'utf-8'
        self.rcon: Optional[redis.Redis] = None

    def initialize(self, uri: str):
        try:
            self.rcon: redis.Redis = redis.from_url(uri)
            return Result.ok(data=True).set_origin("Reddis DB")
        except Exception as e:
            return Result.default_internal_error(data=e).set_origin("Reddis DB")

    def get(self, key: str) -> Result:
        data = []
        if self.rcon is None:
            return (Result.default_user_error(info='Pleas run first-redis-connection to connect to a reddis instance')
                    .set_origin("Reddis DB"))

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
            return Result.ok(info=f"No data found for key {key}", data=None, data_info=data_info).set_origin(
                "Reddis DB")

        return Result.ok(data=data, data_info=data_info + key).set_origin("Reddis DB")

    def if_exist(self, query: str):
        if self.rcon is None:
            return Result.default_user_error(
                info='Pleas run first-redis-connection to connect to a reddis instance').set_origin("Reddis DB")
        if not query.endswith('*'):
            return self.rcon.exists(query)
        i = 0
        for _ in self.rcon.scan_iter(query):
            i += 1
        return i

    def set(self, key: str, value) -> Result:
        if self.rcon is None:
            return Result.default_user_error(
                info='Pleas run first-redis-connection to connect to a reddis instance').set_origin("Reddis DB")
        try:
            self.rcon.set(key, value)
            return Result.ok().set_origin("Reddis DB")
        except TimeoutError as e:
            get_logger().error(f"Timeout by redis DB : {e}")
            return Result.default_internal_error(info=e).set_origin("Reddis DB")
        except Exception as e:
            return Result.default_internal_error(info="Fatal Error: " + str(e)).set_origin("Reddis DB")

    def append_on_set(self, key: str, value: list) -> Result:

        if self.rcon is None:
            return Result.default_internal_error(info='Pleas run first-redis-connection').set_origin("Reddis DB")

        val = self.rcon.get(key)

        if val:
            val = eval(val)
            if not isinstance(val, list):
                return Result.default_user_error(info="Error key: " + str(key) + " is not a set",
                                                 exec_code=-4).set_origin("Reddis DB")
            for new_val in value:
                if new_val in val:
                    return Result.default_user_error(info="Error value: " + str(new_val) + " already in list",
                                                     exec_code=-5).set_origin("Reddis DB")
                val.append(new_val)
        else:
            val = value

        self.rcon.set(key, str(val))

        return Result.ok().set_origin("Reddis DB")

    def delete(self, key, matching=False) -> Result:
        if self.rcon is None:
            return Result.default_user_error(
                info='Pleas run first-redis-connection to connect to a reddis instance').set_origin("Reddis DB")

        del_list = []
        n = 0

        if matching:
            for key_ in self.rcon.scan_iter(key):
                # Check if the key contains the substring
                v = self.rcon.delete(key_)
                del_list.append((key_, v))
        else:
            v = self.rcon.delete(key)
            del_list.append((key, v))
            n += 1

        return Result.ok(data=del_list, data_info=f"Data deleted successfully removed {n} items").set_origin(
            "Reddis DB")

    def exit(self) -> Result:
        if self.rcon is None:
            return Result.default_user_error(info="No reddis connection active").set_origin("Reddis DB")
        t0 = time.perf_counter()
        logger = get_logger()
        try:
            self.rcon.save()
        except Exception as e:
            logger.warning(f"Saving failed {e}")
        try:
            self.rcon.quit()
        except Exception as e:
            logger.warning(f"Saving quit {e}")
        try:
            self.rcon.close()
        except Exception as e:
            logger.warning(f"Saving close {e}")
        return Result.ok(data_info=f"closing time in ms {time.perf_counter() - t0:.2f}", info="Connection closed",
                         data=time.perf_counter() - t0).set_origin("Reddis DB")
