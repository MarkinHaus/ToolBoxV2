from toolboxv2 import MainTool, FileHandler, App, Style
import redis


class MiniRedis:

    def __init__(self):
        self.data = {}

    def scan_iter(self, serch=''):
        return [key for key in list(self.data.keys()) if key.startswith(serch)]

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value

    def delete(self, key):
        return self.data.pop(key)


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.1"
        self.name = "DB"
        self.logs = app.logger if app else None
        self.color = "YELLOWBG"
        self.keys = {
            "url": "redis:url~"
        }
        self.encoding = 'utf-8'
        self.rcon = None
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["first-redis-connection", "set up a web connection to MarkinHaus"],
                    ["get", "get key and value from redis"],
                    ["append_on_set", "append_on_list"],
                    ["set", "set key value pair redis"],
                    ["del", "set key value pair redis"],
                    ["all", "get all (Autocompletion helper)"],
                    ["all-k", "get all-k (get all keys) (Autocompletion helper)"]
                    ],
            "name": "DB",
            "Version": self.show_version,
            "first-redis-connection": self.add_url_con,
            "get": self.get_keys,
            "set": self.set_key,
            "del": self.delete_key,
            "append_on_set": self.append_on_set,

        }

        FileHandler.__init__(self, "db.data", app.id if app else __name__, keys=self.keys,
                             defaults={"url": 'redis://default:id@url.com:port'})
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logs, color=self.color, on_exit=self.on_exit)

    def show_version(self):
        self.print("Version: ", self.version)

    def on_start(self):
        self.load_file_handler()
        version_command = self.get_file_handler(self.keys["url"])
        if version_command is not None and version_command != 'redis://default:id@url.com:port':
            self.rcon = redis.from_url(version_command)
        else:
            self.print("No url found starting localhost server not secure!!!")
            self.print(Style.RED("do not go live"))
            self.logger.warning("Local MiniRedis not secure")
            self.rcon = MiniRedis()

    def on_exit(self):
        self.save_file_handler()

    def add_url_con(self, command):
        if len(command) == 2:
            url = command[1]
        else:
            url = input("Pleas enter URL of Redis Backend default: ")
        self.add_to_save_file_handler(self.keys["url"], url)
        self.rcon = redis.from_url(url)
        return True

    def get_keys(self, command_, app: App):

        if self.rcon is None:
            return 'Pleas run first-redis-connection'

        command = command_[0]
        if command == "get":
            command = command_[1]

        if command == "all":
            for key in self.rcon.scan_iter():
                val = self.rcon.get(key)
                self.print(f"{key} = {val}")
        elif command == "all-k":
            for key in self.rcon.scan_iter():
                self.rcon.get(key)
                self.print(f"{key}")
        else:
            val = ""
            for key in self.rcon.scan_iter(command):
                val = self.rcon.get(key)

            # self.print(self.check(command, app), val, app.id)
            if self.check(command, app):
                if isinstance(val, str):
                    return val
                else:
                    return str(val, 'utf-8')
        return command

    def append_on_set(self, command):

        if self.rcon is None:
            return 'Pleas run first-redis-connection'

        val = self.rcon.get(command[0])

        print(val)

        if val:
            val = eval(val)
            for new_val in command[1]:
                if new_val in val:
                    return "Error value: " + str(new_val) + "already in list"
                val.append(new_val)
        else:
            val = command[1]

        self.rcon.set(command[0], str(val))

        return command

    def check(self, request, app: App):
        if app.id == "tb.config":
            return True
        return True  # not "secret".upper() in request.upper()

    def set_key(self, command):
        if self.rcon is None:
            return 'Pleas run first-redis-connection'
        try:
            if len(command) == 3:
                key = command[1]
                val = command[2]
                self.rcon.set(key, val)
                self.print(f"key: {key} value: {val} DON 3")
            elif len(command) == 2:
                key = command[0]
                val = command[1]
                self.rcon.set(key, val)
                self.print(f"key: {key} value: {val} DON 2")
            else:
                self.print("set {key} {value}, ")
                self.print(f"{command=}")
            return True
        except TimeoutError as e:
            self.logger.error(f"Timeout by redis DB : {e}")
            return False

    def clean_db(self):
        for key in self.rcon.scan_iter():
            self.rcon.delete(key)

    def delete_key(self, ind):
        if self.rcon is None:
            return 'Pleas run first-redis-connection'
        del_list = []

        if len(ind) == 2:
            key = ind[1]
            e = self.rcon.delete(key)
            self.print(f"{e}]")
            self.print(f"key: {key} DEL")
            del_list.append(key)

        elif len(ind) == 3:
            key_ = ind[1]
            all = ind[2] == "*"
            if all:
                # Use the scan method to iterate over all keys
                self.print("")
                for key in self.rcon.scan_iter():
                    # Check if the key contains the substring
                    self.print(f"test: {key} ", end="\r")
                    if key_ in str(key, 'utf-8'):
                        # Delete the key if it contains the substring
                        self.rcon.delete(key)
                        self.print(f"DEL")
                        del_list.append(key)
                    else:
                        self.print(f" next", end=" ")
        else:
            self.print("del {key} || del {key} *")

        return del_list

# BP in #addLodeFucktionBulprint
