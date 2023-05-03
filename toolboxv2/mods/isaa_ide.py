import os
import shutil

from toolboxv2 import MainTool


def list_s_str_f(s):
    if isinstance(s, str):
        if s[0] != '/':
            s = '/' + s
        return s
    if len(s) == 0:
        return ""
    if len(s) == 1:
        return s[0]
    return s[1]


class Tools(MainTool):
    def __init__(self, app=None):
        self.version = "0.1"
        self.name = "isaa_ide"
        self.logs = app.logger if app else None
        self.color = "YELLOW"
        self.tools = {
            "all": [
                ["create", "Erstellt ein neues Element oder Objekt"],
                ["delete", "Löscht ein vorhandenes Element oder Objekt."],
                ["list", "Zeigt eine Liste aller verfügbaren Elemente oder Objekte an."],
                ["move", "Verschiebt ein Element oder Objekt von einer Position oder einem Ort zu einem anderen."],
                ["insert-edit", "Fügt ein neues Element an einer bestimmten Position ein oder bearbeitet ein vorhandenes Element."],
                ["search", "Sucht nach einem bestimmten Element oder Objekt in der Liste oder Sammlung."],
                ["copy", "Erstellt eine Kopie eines vorhandenen Elements oder Objekts."],
            ],
            "create": ["create(path)", "Create a file or directory at the specified path."],
            "delete": ["delete(path)", "Delete the file or directory at the specified path."],
            "list": ["list(path)", "List the contents of the directory at the specified path."],
            "move": ["move(src, dest)", "Move a file or directory from the source path to the destination path."],
            "insert-edit": ["insert_edit(file, start, end, text)",
                            "Insert or edit text in a file starting at the specified line."],
            "search": ["search(path, text)", "Search for files containing the specified text in the specified path."],
            "copy": ["copy(src, dest)", "Copy a file from the source path to the destination path."]
        }

        self.scope = "isaa-directory/"

        self.open_file = ""

        MainTool.__init__(self, load=None, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logs, color=self.color, on_exit=lambda: "")

    def create(self, path):
        """
        Create a file or directory at the specified path.
        """
        p = list_s_str_f(path)
        path = self.scope + p
        self.print("create " + path)

        # Überprüfen, ob path bereits existiert
        if os.path.exists(path) and not (path.endswith("/") or path.endswith("\\")):
            return f"Error: {path} already exists."

        # path in Verzeichnisse aufteilen
        dirs, filename = os.path.split(path)
        if not filename and not (path.endswith("/") or path.endswith("\\")):  # Wenn path endet mit "/"
            dirs, filename = os.path.split(dirs)
        # Baumstruktur erstellen
        curdir = ""
        for d in dirs.split(os.path.sep):
            curdir = os.path.join(curdir, d)
            if not os.path.exists(curdir):
                os.makedirs(curdir)

        # Datei erstellen
        filepath = os.path.join(curdir, filename)
        if filename:
            open(filepath, 'a').close()
            return f"File created at {filepath}"
        else:
            return f"Directory created at {curdir}"

    def delete(self, path):
        """
        Delete the file or directory at the specified path.
        """
        p = list_s_str_f(path)
        path = self.scope + p
        self.print("delete "+path)
        if not os.path.exists(path):
            return f"Error: {path} does not exist."

        if "." in path.split("/")[-1]:
            os.remove(path)
            return f"File deleted at {path}"
        else:
            shutil.rmtree(path)
            return f"Directory deleted at {path}"

    def list(self, path):
        """
        List the contents of the directory at the specified path.
        """
        path = self.scope + list_s_str_f(path)
        self.print("list "+path)
        if not os.path.exists(path):
            return f"Error: {path} does not exist."

        return "\n".join(os.listdir(path))

    def move(self, command):
        """
        move a file or directory to the specified path.
        """
        if len(command) < 2:
            return "Invalid command"
        if len(command) == 3:
            command = command[1:]
        src, dest = command
        src = self.scope + src
        dest = self.scope + dest

        self.print("move to " + dest)

        if not os.path.exists(src):
            return f"Error: {src} does not exist."

        if os.path.exists(dest):
            return f"Error: {dest} already exists."

        dest_dir = os.path.dirname(dest)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        shutil.move(src, dest)

        if os.path.exists(src) or not os.path.exists(dest):
            return f"Error: Failed to move {src} to {dest}."

        return f"{src} moved to {dest}"

    def insert_edit(self, command):
        """
        writes or edit text in a file starting at the specified line.
        """
        if len(command) < 1:
            return "Invalid command"
        if len(command) == 3:
            command = command[1:]
        file_, text = command
        file = self.scope + list_s_str_f(file_)
        self.print("insert_edit "+file)
        if not os.path.exists(file):
            self.create(list_s_str_f(file_))

        with open(file, 'w') as f:
            f.writelines(text)

        return f"File content updated"

    def search(self, command):
        """
        Search for a keyword in a file or directory at the specified path.
        """
        if len(command) < 2:
            return "Invalid command"
        if len(command) == 3:
            command = command[1:]
        path, keyword = command
        path = self.scope + list_s_str_f(path)
        self.print("search "+path)
        if not os.path.exists(path):
            return f"Error: {path} does not exist."

        if "." in path.split("/")[-1]:
            # search for keyword in file
            with open(path, 'r') as f:
                contents = f.read()
                if keyword in contents:
                    return f"Found keyword '{keyword}' in file at {path}"
                else:
                    return f"Keyword '{keyword}' not found in file at {path}"
        else:
            # search for keyword in directory
            found = False
            for root, dirs, files in os.walk(path):
                for name in files:
                    with open(os.path.join(root, name), 'r') as f:
                        contents = f.read()
                        if keyword in contents:
                            found = True
                            self.print(f"Found keyword '{keyword}' in file at {os.path.join(root, name)}")
            if found:
                return f"Found keyword '{keyword}' in files in directory at {path}"
            else:
                return f"Keyword '{keyword}' not found in files in directory at {path}"

    def read(self, path):
        """
        Read the contents of the file at the specified path.
        """
        path = self.scope + list_s_str_f(path)
        self.print("read")
        if not os.path.exists(path):
            return f"Error: {path} does not exist."

        if "." not in path.split("/")[-1]:
            return f"Error: {path} is not a file."

        with open(path, 'r') as file:
            contents = file.read()

        return contents

    def copy(self, command):
        """
        Copy a file from the source path to the destination path.
        """
        if len(command) < 2:
            return "Invalid command"
        if len(command) == 3:
            command = command[1:]
        source_path, dest_path = command
        source_path = self.scope + list_s_str_f(source_path)
        dest_path = self.scope + list_s_str_f(dest_path)
        self.print("copy to "+source_path)
        if not os.path.exists(source_path):
            return f"Error: {source_path} does not exist."

        if not os.path.isfile(source_path):
            return f"Error: {source_path} is not a file."

        if os.path.exists(dest_path):
            return f"Error: {dest_path} already exists."

        shutil.copy(source_path, dest_path)
        return f"File copied from {source_path} to {dest_path}"

    def process_input(self, input_str: str):
        """
        process_input to intact witch file.
        """
        input_str = input_str.replace('"', '')
        input_str = input_str.replace("'", '')
        input_list = input_str.split(" ")

        command = input_list[0]

        if command == "create":
            path = input_list[1]
            return self.create([path])

        elif command == "delete":
            path = input_list[1]
            return self.delete([path])

        elif command == "list":
            path = input_list[1]
            return self.list([path])

        elif command == "read":
            path = input_list[1]
            return self.read([path])

        elif command == "move":
            source = input_list[1]
            destination = input_list[2]
            return self.move([source, destination])

        elif command in "write":
            path = input_list[1]
            if isinstance(input_list[2], str):
                input_list[2] = 0
            if isinstance(input_list[3], str):
                input_list[3] = -1
            start_line = int(input_list[2])
            end_line = int(input_list[3])
            text = " ".join(input_list[4:])
            return self.insert_edit([path, start_line, end_line, text])

        elif command == "search":
            path = input_list[1]
            keyword = input_list[2]
            return self.search([path, keyword])

        elif command == "copy":
            source = input_list[1]
            destination = input_list[2]
            return self.copy([source, destination])

        else:
            return "Invalid command. Please try again."
