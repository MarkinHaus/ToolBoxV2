import inspect
import logging
from toolboxv2 import MainTool, FileHandler, App, Style
import string

from toolboxv2.utils.toolbox import get_app


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "MinimalHtml"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "WHITE"
        # ~ self.keys = {}
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["add_group", "Init a template group with an namespace"],
                    ["add_collection_to_group", "add a collection to group with an namespace"],
                    ["generate_html", "generate html for a collection in a group run witch group and collection name"],
                    ["fuse_to_string", "genrate generate_html to string for keyword usage"],
                    ],
            "name": "MinimalHtml",
            "Version": self.show_version,
            "add_group": self.add_group,
            "add_collection_to_group": self.add_collection_to_group,
            "generate_html": self.generate_html,
            "fuse_to_string": self.fuse_to_string,
        }
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logger, color=self.color, on_exit=self.on_exit)

        self.seves = {}
        self.groups = {}
        self.file_chas = {}

    def on_start(self):
        self.logger.info(f"Starting MinimalHtml")

    def on_exit(self):
        self.logger.info(f"Closing MinimalHtml")

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    def read_template_file(self, file_path):

        if file_path is None:
            return ""

        if file_path in self.file_chas.keys():
            self.print(f"Returning from chash")
            return self.file_chas[file_path]

        self.print(f"Reading template file at {file_path}")
        with open(file_path, 'r') as file:
            content = file.read()

        self.file_chas[file_path] = content

        return content

    def add_group(self, command):
        if command is None:
            return
        if isinstance(command, list):
            command = command[0]
        if command not in self.groups:
            self.groups[command] = {}
            self.print(f"New Group addet name {command}:{list(self.groups.keys())}")

    def add_collection_to_group_wrapper(self, command):
        if command is None:
            return
        self.add_collection_to_group(command[0], command[1])

    def add_collection_to_group(self, group_name, collection):
        if group_name is None:
            return self.return_result(help_text="Invalid Input")
        if collection is None:
            return self.return_result(help_text="Invalid Input")
        if group_name not in self.groups.keys():
            raise ValueError(f"add Group '{group_name}' not found in {list(self.groups.keys())}")

        self.groups[group_name][collection['name']] = collection['group']

    def generate_html(self, group_name: str, collection_name: str):
        if group_name not in self.groups.keys():
            if 'test' not in self.app.id:
                self.logger.error(f"Group '{group_name}' not found in {self.groups.keys()}")
            return f"Group '{group_name}' not found in {self.groups.keys()}"

        groups = self.groups[group_name]
        if collection_name not in groups.keys():
            if 'test' not in self.app.id:
                self.logger.error(f"Section '{collection_name}' not found in group '{group_name}'")
            return f"Section '{collection_name}' not found in group '{group_name}'"

        collections = groups[collection_name]
        # [{'name': 'titel', 'file_path': '/web/1/init0/titel.html', 'kwargs': {'test-name': 'titel1'}}]

        html_elements = []
        i = 0
        for element in collections:
            i += 1
            self.print(f"Generating element {element['name']}")
            if 'file_path' in element:
                template_content = self.read_template_file(element['file_path'])
            elif 'template' in element:
                template_content = element['template']
            else:
                raise ValueError("Invalid arguments nor file_path or template provided")
            template = string.Template(template_content)
            html_element = template.substitute(**element['kwargs'])
            html_elements.append({'name': element['name'], 'html_element': html_element})

        self.print(f"Addet {i} element{'s' if i > 1 else ''} {group_name}:{collection_name}")

        return html_elements

    def fuse_to_string(self, html_elements, join_chat=''):
        if html_elements is None:
            return
        self.print(f"Fusing to string witch {join_chat}::{' '.join([element['name'] for element in html_elements])}")
        return join_chat.join([element['html_element'] for element in html_elements])


if __name__ == '__main__':
    get_app('debug')
    # Usage
    generator = Tools()

    # Add group
    generator.add_group("navigation")

    # Sample data
    # /web/1/init0/titel.html -> <h1>test $test-name</h1>
    titels = {'name': "titels",
              'group': [
                  {'name': 'titel1', 'file_path': './web/0/welcome/welcome.html', 'kwargs': {'test_name': 'titel1'}},
                  {'name': 'titel2', 'file_path': './web/0/welcome/welcome.html', 'kwargs': {'test_name': 'titel2'}}]}

    generator.add_collection_to_group("navigation", titels)

    titels_html_elements = generator.generate_html("navigation", "titels")
    # titels_html_elements = '<h1>test titel1</h1><h1>test titel2</h1>'

    # /web/1/init0/nav.html -> <ul> <input type="text" id="Search" placeholder="Search@Everything"> <div> $titels </div> </ul>
    nav = {'name': "Hadder", 'group': [
        {'name': 'nav', 'file_path': './web/0/isaa_installer/ii.html',
         'kwargs': {'titels': generator.fuse_to_string(titels_html_elements)}}]}
    generator.add_collection_to_group("navigation", nav)

    navigation_html_elements = generator.generate_html("navigation", "Hadder")
    # titels_html_elements = '<ul> <input type="text" id="Search" placeholder="Search@Everything"> <div> <h1>test titel1</h1><h1>test titel2</h1> </div> </ul>'

    # Example: Generate HTML for the navigation group and nav section
    print(titels_html_elements)
    print(navigation_html_elements)
