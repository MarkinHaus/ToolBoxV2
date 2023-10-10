import os
import shutil

import fnmatch
from toolboxv2 import MainTool, App


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
                ["insert-edit",
                 "Fügt ein neues Element an einer bestimmten Position ein oder bearbeitet ein vorhandenes Element."],
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
            "copy": ["copy(src, dest)", "Copy a file from the source path to the destination path."],
            "add_tools_to_config": ["add_tools_to_config(config, isaa or app)", "Copy a file from the source path to the destination path."]
        }

        self.scope = "isaa_work/"
        if not os.path.exists(self.scope):
            os.mkdir(self.scope)

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
        if not filename and not (path.endswith("/") or path.endswith("\\\\")):  # Wenn path endet mit "/"
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
        self.print("delete " + path)
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
        self.print("list " + path)
        if not os.path.exists(path):
            return f"Error: {path} does not exist."

        return "\n".join(os.listdir(path))

    def move(self, source_path, destination_path):
        """
        move a file or directory to the specified path.
        """
        src = self.scope + source_path
        dest = self.scope + destination_path

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

    def write(self, filename, text):
        """
        writes or edit text in a file starting at the specified line.
        IMPORTANT strict to write to a file do not format the action in json separate filename and text by using ',' !
        """
        file_ = list_s_str_f(filename)
        file = self.scope + file_
        self.print("write " + file)
        if not os.path.exists(file):
            self.create(file_)

        with open(file, 'w') as f:
            f.writelines(text)

        return f"File content updated"

    def search_for_content(self, path, search_text):
        """
        Search for a keyword in a file or directory at the specified path.
        """
        path = self.scope + list_s_str_f(path)
        self.print("search " + path)
        if not os.path.exists(path):
            return f"Error: {path} does not exist."

        if "." in path.split("/")[-1]:
            # search for keyword in file
            with open(path, 'r') as f:
                contents = f.read()
                if search_text in contents:
                    return f"Found keyword '{search_text}' in file at {path}"
                else:
                    return f"Keyword '{search_text}' not found in file at {path}"
        else:
            # search for keyword in directory
            found = False
            for root, dirs, files in os.walk(path):
                for name in files:
                    with open(os.path.join(root, name), 'r') as f:
                        try:
                            contents = f.read()
                        except UnicodeDecodeError:
                            contents = ''
                        if search_text in contents:
                            found = True
                            self.print(f"Found keyword '{search_text}' in file at {os.path.join(root, name)}")
            if found:
                return f"Found keyword '{search_text}' in files in directory at {path}"
            else:
                return f"Keyword '{search_text}' not found in files in directory at {path}"

    def find(self, directory, pattern):
        """
        finde a file or directory with a pattern and a given directory
            Patterns are Unix shell style:

    *       matches everything
    ?       matches any single character
    [seq]   matches any character in seq
    [!seq]  matches any char not in seq

        """
        self.print("search " + directory)
        directory = self.scope + list_s_str_f(directory)
        findings = []
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    self.print(f'Found file: {filename}')
                    findings.append(filename.replace(self.scope, ""))
            for basename in dirs:
                if fnmatch.fnmatch(basename, pattern):
                    folder_name = os.path.join(root, basename)
                    findings.append(folder_name.replace(self.scope, ""))
        if not findings:
            self.print('No files found that match the pattern.')
            return 'No files found that match the pattern.'
        return 'Found: '+', '.join(findings)

    def read(self, path):
        """
        Read the contents of the file at the specified path.
        """
        self.print(f"read0 {path}")
        path = self.scope + list_s_str_f(path)
        self.print(f"read {path}")
        if not os.path.exists(path):
            return f"Error: {path} does not exist."

        if "." not in path.split("/")[-1]:

            return os.listdir(path)

        with open(path, 'r') as file:
            contents = file.read()

        if not contents:
            contents = f"The file : {path} is empty"

        return contents

    def copy(self, source_path, destination_path):
        """
        Copy a file from the source path to the destination path.
        """

        source_path = self.scope + list_s_str_f(source_path)
        dest_path = self.scope + list_s_str_f(destination_path)
        self.print("copy to " + source_path)
        if not os.path.exists(source_path):
            return f"Error: {source_path} does not exist."

        if not os.path.isfile(source_path):
            return f"Error: {source_path} is not a file."

        if os.path.exists(dest_path):
            return f"Error: {dest_path} already exists."

        shutil.copy(source_path, dest_path)
        return f"File copied from {source_path} to {dest_path}"

    def add_tools_to_config(self, config, isaa):

        if isinstance(isaa, App):
            isaa = isaa.get_mod('isaa')

        tools_info = {
            "create": """"Inputs: path""",
            "delete": """Inputs: path"}""",
            "list": """Inputs: path"}""",
            "move": """Inputs: {'source_path':source_path, 'destination_path':destination_path}""",
            "write": """Inputs: {'filename':filename, 'text':text}""",
            "search_for_content": """Inputs: {'path':path, 'search_text':search_text}""",
            "copy": """Inputs: {'source_path':source_path, 'destination_path':destination_path}""",
            "find": """Inputs: {'directory':directory, 'pattern':pattern}""",
            "read": """Inputs: path""",
        }

        # Loop through the tools_info dictionary and add each tool to isaa
        for tool_name, tool_info in tools_info.items():
            isaa.add_tool(tool_name, getattr(self, tool_name), getattr(self, tool_name).__doc__, tool_info, config)

# isaa Modify the code so that there is a cativ file to which every access is made ess sol a replace method for the file exist.
##code
