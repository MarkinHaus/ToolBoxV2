import os

from toolboxv2.util import get_logger
from toolboxv2.Style import Style
from toolboxv2.cryp import Code


class FileHandler(Code):

    def __init__(self, filename, name='mainTool', keys=None, defaults=None):
        if defaults is None:
            defaults = {}
        if keys is None:
            keys = {}
        assert filename.endswith(".config") or filename.endswith(".data"), \
            f"filename must end with .config or .data {filename=}"
        self.file_handler_save = {}
        self.file_handler_load = {}
        self.file_handler_key_mapper = {}
        self.file_handler_filename = filename
        self.file_handler_storage = None
        self.file_handler_max_loaded_index_ = 0
        self.file_handler_file_prefix = f".{filename.split('.')[1]}/{name.replace('.', '-')}/"
        # self.load_file_handler()
        self.set_defaults_keys_file_handler(keys, defaults)

    def _open_file_handler(self, mode: str, rdu):
        logger = get_logger()
        logger.info(Style.Bold(Style.YELLOW(f"Opening file in mode : {mode}")))
        if self.file_handler_storage:
            self.file_handler_storage.close()
            self.file_handler_storage = None
        try:
            self.file_handler_storage = open(self.file_handler_file_prefix + self.file_handler_filename, mode)
            self.file_handler_max_loaded_index_ += 1
        except FileNotFoundError:
            if self.file_handler_max_loaded_index_ >= 5:
                print(Style.RED(f"pleas create this file to prosed : {self.file_handler_file_prefix}"
                                f"{self.file_handler_filename}"))
                logger.critical(f"{self.file_handler_file_prefix} {self.file_handler_filename} FileNotFoundError cannot"
                                f" be Created")
                exit(0)
            self.file_handler_max_loaded_index_ += 1
            logger.info(Style.YELLOW(f"Try Creating File: {self.file_handler_file_prefix}{self.file_handler_filename}"))

            if not os.path.exists(f"{self.file_handler_file_prefix}"):
                os.makedirs(f"{self.file_handler_file_prefix}")

            with open(self.file_handler_file_prefix + self.file_handler_filename, 'a'):
                logger.info(Style.GREEN("File created successfully"))
                self.file_handler_max_loaded_index_ = -1
            rdu()

    def open_s_file_handler(self):
        self._open_file_handler('w+', self.open_s_file_handler)
        return self

    def open_l_file_handler(self):
        self._open_file_handler('r+', self.open_l_file_handler)
        return self

    def save_file_handler(self):
        get_logger().info(
            Style.BLUE(
                f"init Saving (S) {self.file_handler_filename} "
            )
        )
        if self.file_handler_storage:
            get_logger().warning(
                f"WARNING file is already open (S): {self.file_handler_filename} {self.file_handler_storage}")

        self.open_s_file_handler()

        get_logger().info(
            Style.BLUE(
                f"Elements to save : ({len(self.file_handler_save.keys())})"
            )
        )

        for key in self.file_handler_save.keys():
            data = self.file_handler_save[key]
            get_logger().info(
                Style.BLUE(
                    f"writing to file : {key} : {len(data)} char(s)"
                )
            )
            self.file_handler_storage.write(key + str(data))
            self.file_handler_storage.write('\n')

        self.file_handler_storage.close()
        self.file_handler_storage = None

        get_logger().info(
            Style.BLUE(
                f"closing file : {self.file_handler_filename} "
            )
        )

        return self

    def add_to_save_file_handler(self, key: str, value: str):
        if len(key) != 10:
            get_logger(). \
                warning(
                Style.YELLOW(
                    'WARNING: key length is not 10 characters'
                )
            )
            return False
        # if key not in self.file_handler_save.keys():
        #     print(Style.YELLOW(f"{key} wos not found in file set new"))
        #     w = 'None'
        # else:
        #     w = self.file_handler_save[key]
        if key not in self.file_handler_load.keys():
            if key in self.file_handler_key_mapper:
                key = self.file_handler_key_mapper[key]

        self.file_handler_load[key] = value
        self.file_handler_save[key] = self.encode_code(value)

        # return w, self.decode_code(w)
        return True

    def load_file_handler(self):
        get_logger().info(
            Style.BLUE(
                f"loading {self.file_handler_filename} "
            )
        )
        if self.file_handler_storage:
            get_logger().warning(
                Style.YELLOW(
                    f"WARNING file is already open (L) {self.file_handler_filename}"
                )
            )
        self.open_l_file_handler()

        for line in self.file_handler_storage:
            line = line[:-1]
            heda = line[:10]
            self.file_handler_save[heda] = line[10:]
            enc = self.decode_code(line[10:])
            self.file_handler_load[heda] = enc

        self.file_handler_storage.close()
        self.file_handler_storage = None

        return self

    def get_file_handler(self, obj: str) -> str or None:
        logger = get_logger()
        if obj not in self.file_handler_load.keys():
            if obj in self.file_handler_key_mapper:
                obj = self.file_handler_key_mapper[obj]
        logger.info(Style.ITALIC(Style.GREY(f"Collecting data from storage key : {obj}")))
        self.file_handler_max_loaded_index_ = -1
        for objects in self.file_handler_load.items():
            self.file_handler_max_loaded_index_ += 1
            if obj == objects[0]:

                try:
                    if len(objects[1]) > 0:
                        return eval(objects[1])
                    logger.warning(
                        Style.YELLOW(
                            f"No data  {obj}  ; {self.file_handler_filename}"
                        )
                    )
                except ValueError:
                    logger.error(f"ValueError Loading {obj} ; {self.file_handler_filename}")
                except SyntaxError:
                    logger.critical(
                        Style.RED(
                            f"SyntaxError Loading {obj} ; {self.file_handler_filename}"
                            f" {len(objects[1])}, {type(objects[1])}"
                        )
                    )
                    pass  # print(Style.YELLOW(f"Data frc : {obj} ; {objects[1]}"))
                except NameError:
                    return str(objects[1])

        if obj in list(self.file_handler_save.keys()):
            r = self.decode_code(self.file_handler_save[obj])
            logger.info(f"returning Default for {obj}")
            return r

        logger.info(f"no data found")
        return None

    def set_defaults_keys_file_handler(self, keys: dict, defaults: dict):
        list_keys = iter(list(keys.keys()))
        df_keys = defaults.keys()
        for key in list_keys:
            self.file_handler_key_mapper[key] = keys[key]
            self.file_handler_key_mapper[keys[key]] = key
            if key in df_keys:
                self.file_handler_load[keys[key]] = str(defaults[key])
                self.file_handler_save[keys[key]] = self.encode_code(defaults[key])
            else:
                self.file_handler_load[keys[key]] = "None"

    def delete_file(self):
        os.remove(self.file_handler_file_prefix + self.file_handler_filename)
        get_logger().warning(Style.GREEN(f"File deleted {self.file_handler_file_prefix + self.file_handler_filename}"))
