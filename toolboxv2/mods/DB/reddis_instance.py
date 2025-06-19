import json
import time
from typing import Any

try:
    import redis
except ImportError:
    def redis() -> None:
        return None
    redis.Redis = None

from toolboxv2 import Result, get_logger

from .types import AuthenticationTypes


def sync_redis_databases(source_url, target_url):
    """Synchronize keys from the source Redis database to the target Redis database.
    This function scans all keys in the source DB and uses DUMP/RESTORE to replicate data to the target.

    Args:
        source_url (str): The Redis URL of the source database.
        target_url (str): The Redis URL of the target database.

    Returns:
        int: The number of keys successfully synchronized.
    """
    try:
        src_client = redis.from_url(source_url)
        tgt_client = redis.from_url(target_url)
    except Exception as e:
        print(f"Error connecting to one of the Redis instances: {e}")
        return 0

    total_synced = 0
    cursor = 0
    try:
        while True:
            cursor, keys = src_client.scan(cursor=cursor, count=100)
            for key in keys:
                try:
                    serialized_value = src_client.dump(key)
                    if serialized_value is None:
                        continue
                    # Restore key with TTL=0 and replace existing key
                    tgt_client.restore(key, 0, serialized_value, replace=True)
                    total_synced += 1
                except Exception as e:
                    print(f"Error syncing key {key}: {e}")
            if cursor == 0:
                break
    except Exception as scan_error:
        print(f"Error during scanning keys: {scan_error}")

    print(f"Synced {total_synced} keys from {source_url} to {target_url}")
    return total_synced



class MiniRedis:
    auth_type = AuthenticationTypes.Uri

    def __init__(self):
        self.encoding = 'utf-8'
        self.rcon: redis.Redis | None = None

    def initialize(self, uri: str):
        try:
            self.rcon: redis.Redis = redis.from_url(uri)
            return Result.ok(data=True).set_origin("Reddis DB")
        except Exception as e:
            return Result.default_internal_error(data=e, info="install redis using pip").set_origin("Reddis DB")

    def get(self, key: str) -> Result:
        data = []
        if self.rcon is None:
            return (Result.default_user_error(info='Pleas run initialize to connect to a reddis instance')
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
            return Result.ok(info="No data found for key", data=None, data_info=data_info).set_origin(
                "Reddis DB")

        if len(data) == 1:
            data = data[0]
        return Result.ok(data=data, data_info=data_info).set_origin("Reddis DB")

    def if_exist(self, query: str):
        if self.rcon is None:
            return Result.default_user_error(
                info='Pleas run initialize to connect to a reddis instance').set_origin("Reddis DB")
        if not query.endswith('*'):
            return self.rcon.exists(query)
        i = 0
        for _ in self.rcon.scan_iter(query):
            i += 1
        return i

    def set(self, key: str, value) -> Result:
        if self.rcon is None:
            return Result.default_user_error(
                info='Pleas run initialize to connect to a reddis instance').set_origin("Reddis DB")
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
            return Result.default_internal_error(info='Pleas run initialize').set_origin("Reddis DB")

        if not isinstance(value, list):
            value: list[Any] = [value]

        db_val: str | None = self.rcon.get(key)

        if db_val:
            if isinstance(db_val, bytes):
                db_val = db_val.decode('utf-8')
            if db_val.startswith('[') and db_val.endswith(']'):
                set_val: list = [s.strip() for s in db_val[1:-1].split(',')]
                set_val = [s[1:-1] for s in set_val]
            else:
                set_val: list = json.loads(db_val.replace("'", '"')).get('set', [])
            if not isinstance(set_val, list):
                return Result.default_user_error(info="Error key: " + str(key) + " is not a set",
                                                 exec_code=-4).set_origin("Reddis DB")
            for new_val in value:
                if new_val in set_val:
                    return Result.default_user_error(info="Error value: " + str(new_val) + " already in list",
                                                     exec_code=-5).set_origin("Reddis DB")
                set_val.append(new_val)
            save_val = json.dumps({'set':set_val})
        elif value:
            save_val = json.dumps({'set': value})
        else:
            save_val: str = '{"set": []}'

        self.rcon.set(key, save_val)

        return Result.ok().set_origin("Reddis DB")

    def delete(self, key, matching=False) -> Result:
        if self.rcon is None:
            return Result.default_user_error(
                info='Pleas run initialize to connect to a reddis instance').set_origin("Reddis DB")

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
