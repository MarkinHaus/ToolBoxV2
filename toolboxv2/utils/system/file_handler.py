import json
import os

from .tb_logger import get_logger

from ..extras.Style import Style

from ..security.cryp import Code


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
        self.file_handler_file_prefix = (f".{filename.split('.')[1]}/"
                                         f"{name.replace('.', '-')}/")
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
            if self.file_handler_max_loaded_index_ == 2:
                os.makedirs(self.file_handler_file_prefix, exist_ok=True)
            if self.file_handler_max_loaded_index_ == 3:
                os.makedirs(".config/mainTool", exist_ok=True)
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
        except OSError and PermissionError as e:
            raise e

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

        self.file_handler_storage.write(json.dumps(self.file_handler_save))

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
        if key not in self.file_handler_load.keys():
            if key in self.file_handler_key_mapper:
                key = self.file_handler_key_mapper[key]

        self.file_handler_load[key] = value
        self.file_handler_save[key] = self.encode_code(value)
        return True

    def remove_key_file_handler(self, key: str):
        if key == 'Pka7237327':
            print("Cant remove Root Key")
            return
        if key in self.file_handler_load.keys():
            del self.file_handler_load[key]
        if key in self.file_handler_save.keys():
            del self.file_handler_save[key]

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

        try:

            self.file_handler_save = json.load(self.file_handler_storage)
            for key, line in self.file_handler_save.items():
                self.file_handler_load[key] = self.decode_code(line)

        except json.decoder.JSONDecodeError and Exception:

            for line in self.file_handler_storage:
                line = line[:-1]
                heda = line[:10]
                self.file_handler_save[heda] = line[10:]
                enc = self.decode_code(line[10:])
                self.file_handler_load[heda] = enc

            self.file_handler_save = {}

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
                        return eval(objects[1]) if isinstance(objects[1], str) else objects[1]
                    logger.warning(
                        Style.YELLOW(
                            f"No data  {obj}  ; {self.file_handler_filename}"
                        )
                    )
                except ValueError:
                    logger.error(f"ValueError Loading {obj} ; {self.file_handler_filename}")
                except SyntaxError:
                    if isinstance(objects[1], str):
                        return objects[1]
                    logger.warning(
                        Style.YELLOW(
                            f"Possible SyntaxError Loading {obj} ; {self.file_handler_filename}"
                            f" {len(objects[1])} {type(objects[1])}"
                        )
                    )
                    return objects[1]
                    # print(Style.YELLOW(f"Data frc : {obj} ; {objects[1]}"))
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


"""
Klasse: FileHandler
Die FileHandler Klasse ist ein Werkzeug zum Verwalten von Dateien, insbesondere zum Speichern, Laden und Verwalten von Schlüssel-Wert-Paaren in Konfigurationsdateien. Sie erbt von der Code Klasse und verfügt über Funktionen zum Öffnen, Schließen, Speichern, Laden und Löschen von Dateien. Es gibt auch Funktionen, um Werte basierend auf ihren Schlüsseln hinzuzufügen, abzurufen oder Standardwerte festzulegen.

Funktionen:

    __init__(self, filename, name='mainTool', keys=None, defaults=None): Initialisiert die FileHandler-Klasse mit dem angegebenen Dateinamen, Namen und optionalen Schlüsseln und Standardwerten.

    _open_file_handler(self, mode: str, rdu): Eine interne Funktion, die zum Öffnen einer Datei in einem bestimmten Modus verwendet wird. Diese Funktion sollte nicht von außen aufgerufen werden.

    open_s_file_handler(self): Öffnet die Datei im Schreibmodus.

    open_l_file_handler(self): Öffnet die Datei im Lesemodus.

    save_file_handler(self): Speichert die in der Klasse gespeicherten Schlüssel-Wert-Paare in der Datei.

    add_to_save_file_handler(self, key: str, value: str): Fügt ein neues Schlüssel-Wert-Paar hinzu, das in der Datei gespeichert werden soll.

    load_file_handler(self): Lädt die Schlüssel-Wert-Paare aus der Datei in die Klasse.

    get_file_handler(self, obj: str) -> str or None: Gibt den Wert für den angegebenen Schlüssel zurück, falls vorhanden.

    set_defaults_keys_file_handler(self, keys: dict, defaults: dict): Setzt die Standardwerte und Schlüssel für die Klasse.

    delete_file(self): Löscht die Datei, auf die sich die Klasse bezieht.

Die Funktionen, die von außen aufgerufen werden sollten, sind __init__, open_s_file_handler, open_l_file_handler, save_file_handler, add_to_save_file_handler, load_file_handler, get_file_handler, set_defaults_keys_file_handler und delete_file. Die Funktion _open_file_handler sollte nicht direkt aufgerufen werden, da sie eine interne Hilfsfunktion ist.

"""
