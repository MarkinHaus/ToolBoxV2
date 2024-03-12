import os
import shelve
from cachetools import TTLCache


class FileCache:
    def __init__(self, folder='', filename='cache.db'):
        self.filename = filename
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    def get(self, key):
        with shelve.open(self.filename) as db:
            return db.get(key)

    def set(self, key, value):
        with shelve.open(self.filename, writeback=True) as db:
            db[key] = value


class MemoryCache:
    def __init__(self, maxsize=100, ttl=300):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value
