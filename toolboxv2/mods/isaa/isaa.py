import json
import logging
import os
import random
import shlex
import sys
import threading
import uuid
from inspect import signature
from typing import Tuple, Any

import openai
import replicate
import torch
from bs4 import BeautifulSoup
from duckduckgo_search import ddg, ddg_answers, ddg_news
from langchain import PromptTemplate, LLMChain, OpenAI, HuggingFaceHub, ConversationChain, GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, tool as LCtool, load_tools, load_huggingface_tool
from langchain.chains import ConversationalRetrievalChain
from langchain.agents.agent_toolkits import FileManagementToolkit
# Model
from langchain.chat_models import ChatOpenAI
from langchain.tools import AIPluginTool
from transformers import pipeline
import gpt4all

from toolboxv2 import MainTool, FileHandler, App, Spinner
from .AgentUtils import *
from toolboxv2.utils.toolbox import get_app

try:
    import inquirer

    INQUIRER = True
except ImportError:
    INQUIRER = False
# Loaders
# Splitters
# Embedding Support
# Summarizer we'll use for Map Reduce
# Data Science

pipeline_arr = [
    # 'audio-classification',
    # 'automatic-speech-recognition',
    # 'conversational',
    # 'depth-estimation',
    # 'document-question-answering',
    # 'feature-extraction',
    # 'fill-mask',
    # 'image-classification',
    # 'image-segmentation',
    # 'image-to-text',
    # 'ner',
    # 'object-detection',
    'question-answering',
    # 'sentiment-analysis',
    'summarization',
    # 'table-question-answering',
    'text-classification',
    # 'text-generation',
    # 'text2text-generation',
    # 'token-classification',
    # 'translation',
    # 'visual-question-answering',
    # 'vqa',
    # 'zero-shot-classification',
    # 'zero-shot-image-classification',
    # 'zero-shot-object-detection',
    # 'translation_en_to_de',
    # 'fill-mask'
]


def get_ip():
    response = requests.get('https://api64.ipify.org?format=json').json()
    return response["ip"]


@concurrent.process(timeout=12)
def get_location():
    ip_address = get_ip()
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_data = f"city: {response.get('city')},region: {response.get('region')},country: {response.get('country_name')},"

    return location_data


def extract_code(x):
    data = x.split('```')
    if len(data) == 3:
        text = data[1].split('\n')
        code_type = text[0]
        code = '\n'.join(text[1:])
        return code, code_type
    if len(data) > 3:
        print(x)
    return '', ''


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        if app is None:
            app = get_app()
        self.version = "0.0.2"
        self.name = "isaa"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "VIOLET2"
        self.config = {'genrate_image-init': False,
                       'agents-name-list': [],
                       "DEFAULTMODEL0": "gpt-4",
                       "DEFAULTMODEL1": "gpt-3.5-turbo-0613",
                       "DEFAULTMODEL2": "text-davinci-003",
                       "DEFAULTMODELCODE": "code-davinci-edit-001",
                       "DEFAULTMODELSUMMERY": "text-curie-001",
                       }
        self.per_data = {}
        self.keys = {
            "KEY": "key~~~~~~~",
            "Config": "config~~~~"
        }
        self.initstate = {}
        self.mas_text_summaries_dict = [[], []]
        self.genrate_image = image_genrating_tool
        extra_path = ""
        if self.toolID:
            extra_path = f"/{self.toolID}"
        self.observation_term_mem_file = f".data/{app.id}/Memory{extra_path}/observationMemory/"
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["api_run", "name inputs"],
                    ["add_api_key", "Adds API Key"],
                    ["image", "genarate image input"],
                    ["api_initIsaa", "init isaa wit dif functions", 0, 'init_isaa_wrapper'],
                    ["add_task", "Agent Chin add - Task"],
                    ["api_save_task", "Agent Chin save - Task", 0, "save_task"],
                    ["api_load_task", "Agent Chin load - Task", 1, "load_task"],
                    ["api_get_task", "Agent Chin get - Task", 0, "get_task"],
                    ["api_list_task", "Agent Chin list - Task", 0, "list_task"],
                    ["api_start_widget", "api_start_widget", 0, "start_widget"],
                    ["api_get_use", "get_use", 0, "get_use"],
                    ["generate_task", "generate_task", 0, "generate_task"],
                    ["init_cli", "init_cli", 0, "init_cli"],
                    ["run_chain_cli", "run_chain_cli", 0, "run_chain_cli"],
                    ["remove_chain_cli", "remove_chain_cli", 0, "remove_chain_cli"],
                    ["create_task_cli", "create_task_cli", 0, "create_task_cli"],
                    ["optimise_task_cli", "optimise_task_cli", 0, "optimise_task_cli"],
                    ["run_create_task_cli", "run_create_task_cli", 0, "run_create_task_cli"],
                    ["run_describe_chains_cli", "run_describe_chains", 0, "run_describe_chains"],
                    ["run_auto_chain_cli", "run_auto_chain_cli", 0, "run_auto_chain_cli"],
                    ["save_to_mem", "save_to_mem", 0, "save_to_mem"],
                    ],
            "name": "isaa",
            "Version": self.show_version,
            "api_run": self.run_isaa_wrapper,
            "image": self.genrate_image_wrapper,
            "api_initIsaa": self.init_isaa_wrapper,
            "api_start_widget": self.start_widget,
            "add_task": self.add_task,
            "save_task": self.save_task,
            "load_task": self.load_task,
            "get_task": self.get_task,
            "list_task": self.list_task,
            "api_get_use": self.get_use,
            "generate_task": self.generate_task,
            "init_cli": self.init_cli,
            "run_chain_cli": self.run_chain_cli,
            "remove_chain_cli": self.remove_chain_cli,
            "create_task_cli": self.create_task_cli,
            "optimise_task_cli": self.optimise_task_cli,
            "run_create_task_cli": self.run_create_task_cli,
            "run_describe_chains_cli": self.run_describe_chains,
            "run_auto_chain_cli": self.run_auto_chain_cli,
            "save_to_mem": self.save_to_mem,
        }
        self.working_directory = "C:\\Users\\Markin\\Isaa"
        self.app_ = app
        self.print_stream = print
        self.agent_collective_senses = False
        self.global_stream_override = False
        self.pipes_device = 1
        self.lang_chain_tools_dict = {}
        self.agent_chain = AgentChain(directory=f".data/{app.id}{extra_path}/chains")
        self.agent_memory = AIContextMemory(extra_path=extra_path)
        self.summarization_mode = 2  # 0 to 3  0 huggingface 1 text 2 opnai 3 gpt
        self.summarization_limiter = 102000
        self.speak = lambda x, *args, **kwargs: x
        self.scripts = Scripts(f".data/{app.id}{extra_path}/ScriptFile")
        self.ac_task = None

        self.price = {
            'all': 0,
            'input': 0,
            'output': 1,
            'consumption': [],
            'price_consumption': 0,
            'model_consumption': {},
            'history': {},
        }

        self.tools_dict = {

        }

        FileHandler.__init__(self, f"isaa{extra_path.replace('/', '-')}.config", app.id if app else __name__)
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=None, color=self.color, on_exit=self.on_exit)

        self.toolID = ""
        MainTool.toolID = ""

    def add_task(self, name, task):
        self.agent_chain.add_task(name, task)

    def save_task(self, name=None):
        self.agent_chain.save_to_file(name)

    def load_task(self, name=None):
        self.agent_chain.load_from_file(name)

    def get_task(self, name):
        return self.agent_chain.get(name)

    def get_augment(self, task_name=None, exclude=None):
        return {
            "tools": self.tools_dict,
            "Agents": self.serialize_all(exclude=exclude),
            "customFunctions": self.scripts.scripts,
            "tasks": self.agent_chain.save_to_dict(task_name)
        }

    def init_from_augment(self, augment, agent_name: str or AgentConfig='self', exclude=None):
        if isinstance(agent_name, str):
            agent = self.get_agent_config_class(agent_name)
        elif isinstance(agent_name, AgentConfig):
            agent = agent_name
        else:
            return ValueError(f"Invalid Type {type(agent_name)} accept ar : str and AgentConfig")
        a_keys = augment.keys()

        if "tools" in a_keys:
            tools = augment['tools']
            print("tools:", tools)
            self.init_tools(agent, tools)
            self.print("tools initialized")

        if "Agents" in a_keys:
            agents = augment['Agents']
            self.deserialize_all(agents, agent, exclude=exclude)
            self.print("Agents crated")

        if "customFunctions" in a_keys:
            custom_functions = augment['customFunctions']
            self.scripts.scripts = custom_functions
            self.print("customFunctions saved")

        if "tasks" in a_keys:
            tasks = augment['tasks']
            if isinstance(tasks, str):
                tasks = json.loads(tasks)
            if tasks:
                self.agent_chain.load_from_dict(tasks)
                self.print("tasks chains restored")

    def init_tools(self, self_agent, tools):  # not  in unit test

        plugins = [
            # SceneXplain
            # "https://scenex.jina.ai/.well-known/ai-plugin.json",
            # Weather Plugin for getting current weather information.
            #    "https://gptweather.skirano.repl.co/.well-known/ai-plugin.json",
            # Transvribe Plugin that allows you to ask any YouTube video a question.
            #    "https://www.transvribe.com/.well-known/ai-plugin.json",
            # ASCII Art Convert any text to ASCII art.
            #    "https://chatgpt-plugin-ts.transitive-bullshit.workers.dev/.well-known/ai-plugin.json",
            # DomainsGPT Check the availability of a domain and compare prices across different registrars.
            # "https://domainsg.pt/.well-known/ai-plugin.json",
            # PlugSugar Search for information from the internet
            #    "https://websearch.plugsugar.com/.well-known/ai-plugin.json",
            # FreeTV App Plugin for getting the latest news, include breaking news and local news
            #    "https://www.freetv-app.com/.well-known/ai-plugin.json",
            # Screenshot (Urlbox) Render HTML to an image or ask to see the web page of any URL or organisation.
            # "https://www.urlbox.io/.well-known/ai-plugin.json",
            # OneLook Thesaurus Plugin for searching for words by describing their meaning, sound, or spelling.
            # "https://datamuse.com/.well-known/ai-plugin.json", -> long loading time
            # Shop Search for millions of products from the world's greatest brands.
            # "https://server.shop.app/.well-known/ai-plugin.json",
            # Zapier Interact with over 5,000+ apps like Google Sheets, Gmail, HubSpot, Salesforce, and thousands more.
            "https://nla.zapier.com/.well-known/ai-plugin.json",
            # Remote Ambition Search millions of jobs near you
            # "https://remoteambition.com/.well-known/ai-plugin.json",
            # Kyuda Interact with over 1,000+ apps like Google Sheets, Gmail, HubSpot, Salesforce, and more.
            # "https://www.kyuda.io/.well-known/ai-plugin.json",
            # GitHub (unofficial) Plugin for interacting with GitHub repositories, accessing file structures, and modifying code. @albfresco for support.
            #     "https://gh-plugin.teammait.com/.well-known/ai-plugin.json",
            # getit Finds new plugins for you
            "https://api.getit.ai/.well_known/ai-plugin.json",
            # WOXO VidGPT Plugin for create video from prompt
            "https://woxo.tech/.well-known/ai-plugin.json",
            # Semgrep Plugin for Semgrep. A plugin for scanning your code with Semgrep for security, correctness, and performance issues.
            # "https://semgrep.dev/.well-known/ai-plugin.json",
        ]

        # tools = {  # Todo save tools to file and loade from usaage data format : and isaa_extras
        #    "lagChinTools": ["ShellTool", "ReadFileTool", "CopyFileTool",
        #                     "DeleteFileTool", "MoveFileTool", "ListDirectoryTool"],
        #    "huggingTools": [],
        #    "Plugins": ["https://nla.zapier.com/.well-known/ai-plugin.json"],
        #    "Custom": [],
        # }

        if 'Plugins' not in tools.keys():
            tools['Plugins'] = []
        if 'lagChinTools' not in tools.keys():
            tools['lagChinTools'] = []
        if 'huggingTools' not in tools.keys():
            tools['huggingTools'] = []

        for plugin_url in set(tools['Plugins'] + self_agent.plugins):
            get_logger().info(Style.BLUE(f"Try opening plugin from : {plugin_url}"))
            try:
                plugin_tool = AIPluginTool.from_plugin_url(plugin_url)
                get_logger().info(Style.GREEN(f"Plugin : {plugin_tool.name} loaded successfully"))
                self.lang_chain_tools_dict[plugin_tool.name + "-usage-information"] = plugin_tool
            except Exception as e:
                get_logger().error(Style.RED(f"Could not load : {plugin_url}"))
                get_logger().error(Style.GREEN(f"{e}"))

        for tool in load_tools(list(set(tools['lagChinTools'] + self_agent.lag_chin_tools)),
                               self.get_llm_models(self_agent.model_name)):
            self.lang_chain_tools_dict[tool.name] = tool
        for tool in set(tools['huggingTools'] + self_agent.hugging_tools):
            self.lang_chain_tools_dict[tool.name] = load_huggingface_tool(tool, self.config['HUGGINGFACEHUB_API_TOKEN'])

        # Add custom Tools

        self.add_lang_chain_tools_to_agent(self_agent, self_agent.tools)

        mem = self.get_context_memory()

        def get_relevant_informations(*args):
            x = ' '.join(args)
            ress = mem.get_context_for(x)

            task = f"Act as an summary expert your specialties are writing summary. you are known to think in small and " \
                   f"detailed steps to get the right result. Your task : write a summary reladet to {x}\n\n{ress}"
            res = self.run_agent(self.get_default_agent_config('think').set_model_name('gpt-3.5-turbo-0613'), task)

            if res:
                return res

            return ress

        def ad_data(*args):
            x = ' '.join(args)
            mem.add_data('main', x)

            return 'added to memory'

        if 'memory' not in self_agent.tools.keys():
            self.add_tool("memory", get_relevant_informations, "a tool to get similar information from your memories."
                                                               " useful to get similar data. ",
                          "memory(<related_information>)",
                          self_agent)

            self.add_tool("save_data_to_memory", ad_data, "tool to save data to memory,"
                                                          " write the data as specific"
                                                          " and accurate as possible.",
                          "save_data_to_memory(<store_information>)",
                          self_agent)

        self.tools_dict = tools

    def serialize_all(self, exclude=None):
        if exclude is None:
            exclude = []
        data = {}
        for agent_name in self.config['agents-name-list']:
            agent = self.get_agent_config_class(agent_name)
            agent_data = agent.serialize()
            for e in exclude:
                del agent_data[e]
            data[agent.name] = agent_data
        return data

    def deserialize_all(self, data, s_agent, exclude=None):
        for key, agent_data in data.items():
            agent = self.get_agent_config_class(key)
            agent.deserialize(agent_data, reste_task=True, agent_config=agent, exclude=exclude)
            ac_tools = {}
            for tool_name in agent.tools.keys():
                if tool_name in s_agent.tools.keys():
                    ac_tools[tool_name] = s_agent.tools[tool_name]
                else:
                    self.print(Style.YELLOW(f"Tools {tool_name} not found"))

    def generate_task(self, subject, variables=None, context=None):

        if context is None:
            context = []
        if variables is None:
            variables = []

        self_agent = self.get_agent_config_class('self')

        agent_context_de = f"""
Handle als Entscheidungsagenten du sollst, basierend auf einer Auswahl an Aufgaben und dem Kontext entscheiden, ob und welche Aufgabe für das Subjekt X angewendet werden soll. Wenn keine Aufgabe eine Erfolgswahrscheinlichkeit von über 80% für die beste Aufgabe aufweist, soll der Agent angeben, dass keine Aufgabe das Ziel erreicht, und das System wird eine passende Aufgabe erstellen.
Befehl: Entscheide, welche Aufgabe für {subject} basierend auf dem Kontext {context} {variables} angewendet werden soll. Wenn keine Aufgabe eine Erfolgswahrscheinlichkeit von über 80% für die beste Aufgabe aufweist, gib an, dass keine Aufgabe das Ziel erreicht, und erstelle eine passende Aufgabe.
Verfügbare aufgaben : {str(self.agent_chain)}
Aufgaben Name oder None:"""

        # task_name = self.mini_task_completion(agent_context_de)
        # task_name_l = task_name.lower()
        # if not (task_name_l != "None".lower() or len(task_name) > 1):
        #    self.init_config_var_initialise('chains-keys', [l.lower() for l in self.agent_chain.chains.keys()])
        #    if task_name_l in self.config['chains-keys']:
        #        return task_name  # Agent selected a valid task
        #
        # self.print_stream(f"Agent Evaluation: System cant detect valid task : {task_name}")
        self.print_stream(f"Pleas Open The Task editor or the isaa task creator")
        tools, names = self_agent.generate_tools_and_names_compact()
        ta_c = self.mini_task_completion(
            f"Handle als Entscheidungsagenten Überlege, wie komplex die Aufgabe ist und welche Fähigkeiten dafür "
            f"benötigt werden. Es gibt verschiedene Tools, Die du zu auswahl hast"
            f", wie zum Beispiel ein Text2Text Taschenrechner. Wähle zwischen einem "
            f"Tool oder einem Agenten für diese Aufgabe. Die verfügbaren Tools sind "
            f"{names}. Hier sind ihre Beschreibungen: {tools}. Es stehen auch folgende "
            f"Agenten zur Verfügung: {self.config['agents-name-list']}. Wenn weder ein "
            f"Agent noch ein Tool zum Thema '{subject}' passen, hast du noch eine weitere Option: "
            f"Gib 'Create-Agent' ein. Bitte beachte den Kontext: {context} {variables}. "
            f"Was ist dein gewählter Tool- oder Agentenname oder möchtest du einen "
            f"Agenten erstellen?"
            f"Ausgabe:")

        if not ta_c:
            ta_c = 'crate-task'

        self.print(ta_c)

        return {'isaa': ta_c}  ## TODO test

    def list_task(self):
        return str(self.agent_chain)

    def run_isaa_wrapper(self, command):
        self.print(f"Running isaa wrapper {command}")
        # if len(command) < 1:
        #    return "Unknown command"
        #
        # return self.run_agent(command[0].data['name'], command[0].data['text'])
        return """Um alle `h`-Elemente (Überschriften) in einem `div` auszuwählen, können Sie in Ihrer CSS-Datei oder im `<style>`-Bereich Ihres HTML-Dokuments den folgenden CSS-Selektor verwenden:

```css
div h1, div h2, div h3, div h4, div h5, div h6 {
  /* Hier können Sie Ihre gewünschten Stile hinzufügen */
}
```"""

    def genrate_image_wrapper(self, command):
        if len(command) != 1:
            return "Unknown command"

        return self.genrate_image(command[0], self.app_)

    def get_use(self, command, app):

        uid, err = self.get_uid(command, app)

        if err:
            return "Invalid Token"

        self.logger.debug("Instance get_user_instance")

        # user_instance = self.get_user_instance(uid, app)
        use = command[0].data['use']
        if use == "agent":
            return {"isaaUseResponse": self.config["agents-name-list"]}
        if use == "tool":
            return {"isaaUseResponse": list(self.get_agent_config_class('self').tools.keys())}
        if use == "chain":
            return {"isaaUseResponse": list(set(self.agent_chain.live_chains + self.agent_chain.chains))}
        if use == "function":
            return {"isaaUseResponse": list(self.scripts.scripts.keys())}

    def start_widget(self, command, app):

        uid, err = self.get_uid(command, app)

        if err:
            return "Invalid Token"

        self.logger.debug("Instance get_user_instance")

        user_instance = self.get_user_instance(uid, app)

        self.logger.debug("Instace Recived")

        sender, receiver = self.app_.run_any("WebSocketManager", "srqw",
                                             ["ws://localhost:5000/ws", user_instance["webSocketID"]])

        widget_id = str(uuid.uuid4())[25:]

        def print_ws(x):
            sender.put(json.dumps({"Isaa": x}))

        self.print_stream = print_ws

        group_name = user_instance["webSocketID"] + "-IsaaSWidget"
        collection_name = user_instance["webSocketID"] + '-' + widget_id + "-IsaaSWidget"

        self.app_.run_any("MinimalHtml", "add_group", [group_name])

        widget_data = {'name': collection_name, 'group': [
            {'name': 'nav', 'file_path': './app/1/simpchat/simpchat.html',
             'kwargs': {'chatID': widget_id}}]}

        self.app_.run_any("MinimalHtml", "add_collection_to_group", [group_name, widget_data])

        isaa_widget_html_element = self.app_.run_any("MinimalHtml", "generate_html", [group_name, collection_name])

        print(isaa_widget_html_element)

        # Initialize the widget ui
        ui_html_content = self.app_.run_any("WebSocketManager", "construct_render",
                                            command=isaa_widget_html_element[0]['html_element'],
                                            element_id="widgetChat",
                                            externals=["/app/1/simpchat/simpchat.js"])

        # Initial the widget backend
        # on receiver { task: '', IChain': {
        #             "args": "Present the final report $final_report",
        #             "name": "execution",
        #             "return": "$presentation",
        #             "use": "agent"
        #         } }

        def runner():

            uesd_mem = {}
            chain_data = {}
            chain_ret = []

            running = True
            while running:
                while not receiver.empty():
                    data = receiver.get()

                    if 'exit' in data:
                        running = False
                    self.logger.info(f'Received Data {data}')

                    # if 'widgetID' not in data.keys():
                    #    continue
                    #
                    # self.logger.info(f'widgetID found in Data keys Valid:{data["widgetID"] != widget_id}')
                    #
                    # if data['widgetID'] != widget_id:
                    #    continue

                    try:
                        if "type" in data.keys():
                            if 'id' not in data.keys():
                                continue
                            # if data['id'] != widget_id:
                            #    continue
                            if data["type"] == "textWidgetData":
                                chain_data[data["context"]] = data["text"]
                                sender.put({"ChairData": True, "data": {'res': f"Text in {data['context']}"}})
                        elif 'task' in data.keys() and 'IChain' in data.keys():
                            chain_ret, chain_data, uesd_mem = self.execute_thought_chain(data['task'], [data["IChain"]],
                                                                                         chain_ret=chain_ret,
                                                                                         chain_data=chain_data,
                                                                                         uesd_mem=uesd_mem,
                                                                                         chain_data_infos=True,
                                                                                         config=self.get_agent_config_class(
                                                                                             "self"))

                            sender.put({"ChairData": True, "data": {'res': chain_ret[-1][-1]}})
                        elif 'subject' in data.keys():
                            context = self.agent_memory.get_context_for(data['subject'])
                            res = self.generate_task(data['subject'], str(chain_data), context)
                            sender.put({"ChairData": True, "data": {'res': res}})

                    except Exception as e:
                        sender.put({'error': f"Error e", 'res': str(e)})
                        sender.put('exit')
            sender.put('exit')

        widget_runner = threading.Thread(target=runner)
        widget_runner.start()

        self.print(ui_html_content)

        return ui_html_content

    def init_isaa_wrapper(self, command, app):

        uid, err = self.get_uid(command, app)

        if err:
            return "Invalid Token"

        self.print("Init Isaa Instance")

        modis = command[0].data['modis']

        if 'global_stream_override' in modis:
            self.global_stream_override = True

        self.init_isaa()

    def init_isaa(self):

        sys.setrecursionlimit(1500)

        qu_init_t = threading.Thread(target=self.init_all_pipes_default)
        qu_init_t.start()

        mem_init_t = threading.Thread(target=self.get_context_memory().load_all)
        mem_init_t.start()

        self_agent_config: AgentConfig = self.get_agent_config_class("self")

        mem = self.get_context_memory()

        def get_relevant_informations(x):
            ress = mem.get_context_for(x)

            task = f"Act as an summary expert your specialties are writing summary. you are known to think in small and " \
                   f"detailed steps to get the right result. Your task : write a summary reladet to {x}\n\n{ress}"
            res = self.run_agent(self.get_default_agent_config('think').set_model_name('gpt-3.5-turbo-0613'), task)

            if res:
                return res

            return ress

        def ad_data(x):
            mem.add_data('main', x)

            return 'added to memory'

        self.add_tool("memory", get_relevant_informations, "a tool to get similar information from your memories."
                                                           " useful to get similar data. ",
                      "memory(<related_information>)",
                      self_agent_config)

        self.add_tool("save_data_to_memory", ad_data, "tool to save data to memory,"
                                                      " write the data as specific"
                                                      " and accurate as possible.",
                      "save_data_to_memory(<store_information>)",
                      self_agent_config)

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    def on_start(self):
        self.print("Isaa starting init fh, env, config")
        self.load_file_handler()
        self.agent_chain.load_from_file()
        # self.load_keys_from_env()
        self.scripts.load_scripts()
        config = self.get_file_handler(self.keys["Config"])
        if config is not None:
            if isinstance(config, str):
                config = json.loads(config)
            if isinstance(config, dict):
                self.config = {**self.config, **config}
        if 'price' in self.config.keys():
            if isinstance(self.config['price'], dict):
                self.logger.info("Persist Price from Last session")
                self.price = self.config['price']
        if not os.path.exists(f".data/{get_app().id}/isaa/"):
            os.mkdir(f".data/{get_app().id}/isaa/")

    def load_keys_from_env(self):
        self.config['WOLFRAM_ALPHA_APPID'] = os.getenv('WOLFRAM_ALPHA_APPID')
        self.config['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        self.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
        self.config['REPLICATE_API_TOKEN'] = os.getenv('REPLICATE_API_TOKEN')
        self.config['IFTTTKey'] = os.getenv('IFTTTKey')
        self.config['SERP_API_KEY'] = os.getenv('SERP_API_KEY')
        self.config['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
        self.config['PINECONE_API_ENV'] = os.getenv('PINECONE_API_ENV')
        self.config['DEFAULTMODEL0'] = os.getenv("DEFAULTMODEL0", "gpt-4")
        self.config['DEFAULTMODEL1'] = os.getenv("DEFAULTMODEL1", "gpt-3.5-turbo-0613")
        self.config['DEFAULTMODEL2'] = os.getenv("DEFAULTMODEL2", "text-davinci-003")
        self.config['DEFAULTMODELCODE'] = os.getenv("DEFAULTMODELCODE", "code-davinci-edit-001")
        self.config['DEFAULTMODELSUMMERY'] = os.getenv("DEFAULTMODELSUMMERY", "text-curie-001")

    def webInstall(self, user_instance, construct_render) -> str:
        self.print('Installing')
        return construct_render(content="./app/0/isaa_installer/ii.html",
                                element_id="Installation",
                                externals=["/app/0/isaa_installer/ii.js"],
                                from_file=True)

    def on_exit(self):
        self.show_usage()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        if now in self.price['history'].keys():
            self.price['input'] += self.price['history'][now]['i']
            self.price['output'] += self.price['history'][now]['o']
            self.price['price_consumption'] += self.price['history'][now]['c']

        self.price['history'][now] = {
            'i': self.price['input'],
            'o': self.price['output'],
            'c': self.price['price_consumption'],
        }

        self.price['input'] = 0
        self.price['output'] = 0
        self.price['price_consumption'] = 0

        self.config['price'] = self.price
        self.config['augment'] = self.get_augment()
        del self.config['augment']['tasks']
        for key in list(self.config.keys()):
            if key.startswith("LLM-model-"):
                del self.config[key]
            if key.startswith("agent-config-"):
                del self.config[key]
            if key.endswith("_pipeline"):
                del self.config[key]
            if key.endswith("-init"):
                self.config[key] = False
            if key == 'agents-name-list':
                self.config[key] = []
        self.add_to_save_file_handler(self.keys["Config"], json.dumps(self.config))
        self.save_file_handler()
        self.agent_chain.save_to_file()
        self.scripts.save_scripts()

    def init_config_var_initialise(self, key: str, value):
        key_i = key + '-init'
        if key_i not in self.config.keys():
            self.config[key_i] = False
        if not self.config[key_i]:
            self.config[key] = value
            self.config[key_i] = True

    def init_config_var_reset(self, key):
        key = key + '-init'
        self.config[key] = False

    def init_all_pipes_default(self):
        self.init_pipeline('question-answering', "deepset/roberta-base-squad2")
        self.init_pipeline('summarization', "pinglarin/summarization_papers")
        self.init_pipeline('text-classification', "distilbert-base-uncased-finetuned-sst-2-english")

    def init_pipeline(self, p_type, model):
        if not p_type in self.initstate.keys():
            self.initstate[p_type] = False

        if not self.initstate[p_type]:
            self.logger.info(f"init {p_type} pipeline")
            if self.pipes_device >= 1 and torch.cuda.is_available():
                if torch.cuda.device_count() < self.pipes_device:
                    self.print("device count exceeded ava-label ar")
                    for i in range(1, torch.cuda.device_count()):
                        self.print(torch.cuda.get_device_name(i - 1))

                self.config[f"{p_type}_pipeline"] = pipeline(p_type, model=model, device=self.pipes_device - 1)
            else:
                self.logger.warning("Cuda is not available")
                self.config[f"{p_type}_pipeline"] = pipeline(p_type, model=model)
            self.logger.info("Done")
            self.initstate[p_type] = True

    def question_answering(self, question, context, model="deepset/roberta-base-squad2", **kwargs):
        self.init_pipeline('question-answering', model)
        qa = {
            'question': question,
            'context': context
        }
        return self.config["question-answering_pipeline"](qa, **kwargs)

    def summarization(self, text, model="pinglarin/summarization_papers", **kwargs):
        # if isinstance(text, str):
        #     print(f"\t\tsummarization({len(text)})")
        # if isinstance(text, list):
        #     print(f"\t\tsummarization({len(text) * len(text[0])})")
        self.init_pipeline('summarization', model)
        try:
            summary_ = self.config["summarization_pipeline"](text, **kwargs)
        except IndexError as e:
            if isinstance(text, str):
                h = len(text) // 2
                self.logger.warning(f'Summarization text to log split in to tex len : {len(text)} splitt to {h}')
                summary_text_ = self.summarization(text[:h], **kwargs)[0]['summary_text']
                summary_ = self.summarization(text[h:], **kwargs)
                summary_[0]['summary_text'] = summary_text_ + '\n' + summary_[0]['summary_text']
            if isinstance(text, list):
                old_cap = len(text[0])
                new_cap = int(old_cap * .95)

                print(f"\tCould not generate summary old cap : {old_cap} new cap : {new_cap}")

                new_text = []
                str_text = ' '.join(text)
                num_tokens = new_cap / 2.0

                if num_tokens > 1020:
                    new_cap = int(new_cap / (num_tokens / 1020))
                    print(f"\t\t2New cap : {new_cap}")

                while len(str_text) > new_cap:
                    new_text.append(str_text[:new_cap])
                    str_text = str_text[new_cap:]
                if str_text:
                    new_text.append(str_text)
                summary_ = self.summarization(new_text, **kwargs)
            else:
                summary_ = f"text type invalid {type(text)} valid ar str and list"

        return summary_

    def text_classification(self, text, model="distilbert-base-uncased-finetuned-sst-2-english", **kwargs):
        self.init_pipeline('text-classification', model)
        return self.config["text-classification_pipeline"](text, **kwargs)

    def toolbox_interface(self):
        @LCtool("toolbox", return_direct=False)
        def function(query: str) -> str:
            """Using The toolbox for interacting with toolbox mods -> modules_name function_name arguments ... """
            data = query.split(' ')
            if len(data) < 2:
                return "invalid syntax"
            data += [""]
            try:
                return self.app_.run_any(data[0], data[1], [""] + data[2:])
            except Exception as e:
                return "Das hat leider nicht geklappt ein Fehler" \
                       " ist bei der ausfürgung des Tools aufgetreten Fehler meldung : " + str(e)

        return function

    def toolbox_information_interface(self):
        @LCtool("toolbox_infos", return_direct=False)
        def function(query: str) -> str:
            """Get information about toolbox mods -> Ask to list Avalabel
             mods get-mod-list, or Ask of Spezifika mod infos mod_names"""

            infos = "invalid syntax"

            if "get-mod-list" in query.lower():
                infos = ' '.join(self.app_.MACRO[8:])

            for modname in self.app_.MACRO:
                if modname in query.lower():
                    infos = str(self.app_.HELPER)

            return infos

        return function

    def generate_image(self):
        @LCtool("Image", return_direct=False)
        def function(query: str) -> str:
            """Generate image with Stable diffusion"""
            try:
                image_genrating_tool(query, self.app_)
            except NameError as e:
                return "Das hat leider nicht geklappt ein Fehler tip versuche es auf englisch, benutze synonyme" \
                       " ist bei der ausfürgung des Tools aufgetreten Fehler meldung : " + str(e)
            return "Das bild wird in kürze angezeigt"

        return function

    def free_llm_model(self, names: list[str]):
        for model in names:
            self.initstate[f'LLM-model-{model}-init'] = False
            del self.config[f'LLM-model-{model}']

    def load_llm_models(self, names: list[str]):
        for model in names:
            if f'LLM-model-{model}-init' not in self.initstate.keys():
                self.initstate[f'LLM-model-{model}-init'] = False

            if not self.initstate[f'LLM-model-{model}-init']:
                self.initstate[f'LLM-model-{model}-init'] = True
                if '/' in model:
                    self.config[f'LLM-model-{model}'] = HuggingFaceHub(repo_id=model,
                                                                       huggingfacehub_api_token=self.config[
                                                                           'HUGGINGFACEHUB_API_TOKEN'])
                    self.print(f'Initialized HF model : {model}')
                elif model.startswith('gpt4all#'):
                    m = gpt4all.GPT4All(model.replace('gpt4all#', ''))
                    self.config[f'LLM-model-{model}'] = m
                    self.print(f'Initialized gpt4all model : {model}')
                elif model.startswith('gpt'):
                    self.config[f'LLM-model-{model}'] = ChatOpenAI(model_name=model,
                                                                   openai_api_key=self.config['OPENAI_API_KEY'],
                                                                   streaming=True)
                    self.print(f'Initialized OpenAi model : {model}')
                else:
                    self.config[f'LLM-model-{model}'] = OpenAI(model_name=model,
                                                               openai_api_key=self.config['OPENAI_API_KEY'])
                    self.print(f'Initialized OpenAi : {model}')

    def get_llm_models(self, name: str):
        if f'LLM-model-{name}' not in self.config.keys():
            self.load_llm_models([name])
        return self.config[f'LLM-model-{name}']

    def add_tool(self, name, func, dis, form, config: AgentConfig, lagchaintool=False):

        if name is None:
            self.print(Style.RED('Error no name specified'))
            return
        if func is None:
            self.print(Style.RED(f'Error no func specified {Style.CYAN(f"Tool {name} not active")}'))
            return
        if dis is None:
            self.print(Style.RED(f'Error no dis specified {Style.CYAN(f"Tool {name} not active")}'))
            return
        if form is None:
            self.print(Style.RED(f'Error no form specified {Style.CYAN(f"Tool {name} not active")}'))
            return
        if config is None:
            self.print(Style.RED(f'Error no config specified {Style.CYAN(f"Tool {name} not active")}'))
            return

        self.print(f"ADDING TOOL:{name} to {config.name}")

        tool = {name: {"func": func, "description": dis, "format": form}}
        if lagchaintool:
            tool[name]['langchain-tool'] = func

        config.tools.update(tool)

    def add_lang_chain_tools_to_agent(self, agent, tools=None):

        if tools is None:
            tools = {}
        for key, _tool in self.lang_chain_tools_dict.items():
            try:
                tools[key] = {"func": _tool, "description": _tool.description, "format": f"{key}({_tool.args})",
                              'langchain-tool': True}
            except Exception as e:
                self.logger.error(Style.YELLOW(Style.Bold(f"Error in add tool : {key} {e}")))
                self.print(Style.RED(f"Tools:{key} Not available"))

        agent.set_tools(tools)

    def create_agent_class(self, name="BP"):
        return AgentConfig(self, name)

    def get_default_agent_config(self, name="Normal") -> AgentConfig:
        config = self.create_agent_class(name)
        if name != "Normal":
            if os.path.exists(f".data/{get_app().id}/Memory/{name}.agent"):
                config = AgentConfig.load_from_file(self, name)

        def toggel(x):
            x = x.lower()
            if x in config.available_modes:
                config.mode = x
                return f"Switched to {config.mode}"

            return f"Switched to {config.mode}"

        config.name = name

        if self.global_stream_override:
            config.stream = True

        def run_agent(agent_name, text, mode_over_lode: bool or str = False):
            text = text.replace("'", "").replace('"', '')
            if agent_name:
                return self.run_agent(agent_name, text, mode_over_lode=mode_over_lode)
            return "Provide Information in The Action Input: fild or function call"

        def search_text(x):
            search = GoogleSearchAPIWrapper()
            x = x.replace("'", "").replace('"', '')
            print(Style.CYAN(x))
            responses = ddg(x)
            qa = ddg_answers(x)
            responses_yz = search.run(x)
            response = self.mas_text_summaries(responses_yz, min_length=600)

            if responses:
                for res in responses[:4]:
                    response += f"\ntitle:{res['title']}\nhref:{res['href']}\n" \
                                f"body:{self.mas_text_summaries(res['body'], min_length=600)}\n\n"

            if qa:
                for res in qa[:4]:
                    response += f"\nurl:{res['url']}\n" \
                                f"text:{self.mas_text_summaries(res['text'], min_length=600)}\n\n"

            print(response)
            if len(response) == 0:
                return "No data found"
            return response

        def search_news(x):

            x = x.replace("'", "").replace('"', '')
            responses = ddg_news(x, max_results=5)

            if not responses:
                return "No News"
            response = ""
            for res in responses:
                response += f"\ntitle:{res['title']}\n" \
                            f"date:{res['date']}\n" \
                            f"url:{res['url']}\n" \
                            f"source:{res['source']}\n" \
                            f"body:{self.mas_text_summaries(res['body'], min_length=1000)}\n\n"
            if len(response) == 0:
                return "No data found"
            return response

        def browse_url(text):

            text = text.replace("'", "").replace('"', '')
            if text.startswith("http:") or text.startswith("https:"):
                url = text.split("|")[0]
                question = text.split("|")[1:]
                res = browse_website(url, question, self.mas_text_summaries)
                return res
            return f"{text[:30]} is Not a Valid url. please just type <url>"

        def memory_search(x):
            ress = self.get_context_memory().get_context_for(x)

            task = f"Act as an summary expert your specialties are writing summary. you are known to think in small and " \
                   f"detailed steps to get the right result. Your task : write a summary reladet to {x}\n\n{ress}"
            res = self.run_agent('thinkm', task)

            if res:
                return res

            return ress

        if name == "self":
            config.mode = "free"
            config.model_name = self.config['DEFAULTMODEL0']  # "gpt-4"
            config.max_iterations = 6
            config.personality = """
Resourceful: Isaa is able to efficiently utilize its wide range of capabilities and resources to assist the user.
Collaborative: Isaa is work seamlessly with other agents, tools, and systems to deliver the best possible solutions for the user.
Empathetic: Isaa is understand and respond to the user's needs, emotions, and preferences, providing personalized assistance.
Inquisitive: Isaa is continually seek to learn and improve its knowledge base and skills, ensuring it stays up-to-date and relevant.
Transparent: Isaa is open and honest about its capabilities, limitations, and decision-making processes, fostering trust with the user.
Versatile: Isaa is adaptable and flexible, capable of handling a wide variety of tasks and challenges.
                  """
            config.goals = "Isaa's primary goal is to be a digital assistant designed to help the user with various " \
                           "tasks and challenges by leveraging its diverse set of capabilities and resources."
            config.tools = {
                "memory_search": {"func": lambda x: memory_search(x),
                                  "description": "Serch for simmilar memory imput <context>"},
                "search_web": {"func": lambda x: run_agent('search', x),
                               "description": "Run agent to search the web for information's"
                    , "format": "search(<task>)"},
                # "write-production-redy-code": {"func": lambda x: run_agent('think',
                #                                                            f"Act as a Programming expert your specialties are coding."
                #                                                            f" you are known to think in small and detailed steps to get"
                #                                                            f" the right result.\n\nInformation's:"
                #                                                            f" {config.edit_text.text}\n\n Your task : {x}\n\n"
                #                                                            f"write an production redy code"),
                #                                "description": "Run agent to generate code."
                #     , "format": "write-production-redy-code(<task>)"},
                "mode_switch": {"func": lambda x: toggel(x),
                                "description": f"switch the mode of the agent avalabel ar : {config.available_modes}"
                    , "format": "mode_switch(<mode>)"},
                "think": {"func": lambda x: run_agent('thinkm', x),
                          "description": "Run agent to solve a text based problem"
                    , "format": "programming(<task>)"},

                "image-generator": {"func": lambda x: image_genrating_tool(x, self.app_),
                                    "description": "Run to generate image"
                    , "format": "reminder(<detaild_discription>)"},
                "mini_task": {"func": lambda x: self.mini_task_completion(x),
                              "description": "programmable pattern completion engin. use text davici args:str only"
                    , "format": "reminder(<detaild_discription>)"},

            }
            add_shell_tool(self, config)

        if "tools" in name:
            tools = {}
            for key, _tool in self.lang_chain_tools_dict.items():
                tools[key] = {"func": _tool, "description": _tool.description, "format": f"{key}({_tool.args})"}
            config. \
                set_mode("tools") \
                .set_model_name(self.config['DEFAULTMODEL0']) \
                .set_max_iterations(12) \
                .set_completion_mode("chat") \
                .set_tools(tools)
        # "gpt-3.5-turbo-0613"
        if name == "todolist":

            def priorisirung(x):
                config.step_between = "take time to gently think about the problem and consider the whole picture."
                if len(config.task_list) != 0:
                    config.step_between = config.task_list[0]
                return run_agent(config.name, x, mode_over_lode="talk")

            config. \
                set_model_name(self.config['DEFAULTMODEL1']). \
                set_max_iterations(4). \
                set_mode('tools'). \
                set_tools({"Thinck": {"func": lambda x: priorisirung(x),
                                      "description": "Use Tool to perform complex resenig"}}). \
                set_personality("""As a proactive agent, I can identify and take on tasks without constant prompting
                    or supervision. I am organized, efficiently handling information and resources while following a
                    structured approach to planning and managing tasks. Adaptable, I can respond to changes and adjust my
                    strategies and plans accordingly. I focus on solving problems and overcoming obstacles rather than
                    dwelling on difficulties. I am communicative, effectively exchanging information and fostering
                    collaboration. I pay attention to details without losing sight of the big picture."""). \
                set_goals("""I have a clear understanding of the desired goal and can break it down into concrete and
                    measurable steps (goal clarity). I can prioritize tasks based on their importance and urgency,
                    ensuring the most critical tasks are completed first (prioritization). I can manage time effectively,
                    ensuring tasks are completed within a reasonable timeframe (time management). I can efficiently use
                    available resources and identify and procure additional resources when needed (resource management).
                    I regularly monitor the progress of tasks and make adjustments to ensure the goal is achieved (
                    progress monitoring). I am constantly striving to improve my skills and processes to increase
                    efficiency and effectiveness in achieving goals (continuous improvement)."""). \
                task_list = ["Make a todo list."
                             "Go through each todo and consider whether that item can be done in one step."
                             "can be done. If not, divide this item into smaller sub-items."
                             "Find relevant information on each todo and estimate how long each task will take"
                             "takes"
                             "Create the final todo list in format\n"
                             "TODO <name_of_list>\n'todo_item': ['resources_that_will_be_used',"
                             "'time_to_be_claimed_of_the_todo', 'subitem':'description']",
                             "Return only the todo list."]

            config.short_mem.max_length = 3000

        if name == "search":
            config.mode = "tools"
            config.model_name = self.config['DEFAULTMODEL1']
            config.completion_mode = "chat"
            config.set_agent_type("zero-shot-react-description")
            config.max_iterations = 6
            config.verbose = True

            config.personality = """
            Resourceful: The Search Agent should be adept at finding relevant and reliable information from various sources on the web.
            Analytical: The Search Agent should be skilled at analyzing the retrieved information and identifying key points and themes.
            Efficient: The Search Agent should be able to quickly search for and summarize information, providing users with accurate and concise results.
            Adaptive: The Search Agent should be able to adjust its search and summarization strategies based on the user's query and the available information.
            Detail-Oriented: The Search Agent should pay close attention to the details of the information it finds, ensuring accuracy and relevance in its summaries."""

            config.goals = """
            1. Information Retrieval: The primary goal of the Search Agent is to find relevant and reliable information on the web in response to user queries.
            2. Text Summarization: The Search Agent should be able to condense the retrieved information into clear and concise summaries, capturing the most important points and ideas.
            3. Relevance Identification: The Search Agent should be able to assess the relevance of the information it finds, ensuring that it meets the user's needs and expectations.
            4. Source Evaluation: The Search Agent should evaluate the credibility and reliability of its sources, providing users with trustworthy information.
            5. Continuous Improvement: The Search Agent should continuously refine its search algorithms and summarization techniques to improve the quality and relevance of its results over time."""

            config.short_mem.max_length = 3500
            config.tools = {
                "memory_search": {"func": lambda x: memory_search(x),
                                  "description": "Search for memory  <context>"},
                "browse_url": {"func": lambda x: browse_url(x),
                               "description": "browse web page via URL syntax <url>"},
                "search_web": {"func": lambda x: search_text(x),
                               "description": "Use Duck Duck go to search the web systax <key word>"},
                # "search_news": {"func": lambda x: search_news(x),
                #                 "description": "Use Duck Duck go to search the web for new get time"
                #                                "related data systax <key word>"}
                # "chain_search_web": {"func": lambda x: run_agent('chain_search_web', x),
                #                     "description": "Run chain agent to search in the web for informations, Only use for complex mutistep tasks"
                #    , "chain_search_web": "search(<task>)"},
                # "chain_search_url": {"func": lambda x: run_agent('chain_search_url', x),
                #                     "description": "Run chain agent to search by url for informations provide mutibel urls, Only use for complex mutistep tasks"
                #    , "format": "chain_search_url(<task,url1,url...>)"},
                # "chain_search_memory": {"func": lambda x: run_agent('chain_search_memory', x),
                #                        "description": "Run chain agent to search in the memory for informations, Only use for complex mutistep tasks"
                #    , "format": "chain_search_memory(<task>)"},
            }

            config.task_list: list[str] = ["Erfülle die Aufgae in so wenigen Schritten und so Bedacht wie Möglich"]

        if name == "think":
            config. \
                set_mode("free") \
                .set_max_iterations(1) \
                .set_completion_mode("chat").set_model_name(self.config['DEFAULTMODEL0'])

            config.stop_sequence = ["\n\n\n"]

        if name == "TaskCompletion":
            config. \
                set_mode("free") \
                .set_max_iterations(1) \
                .set_completion_mode("text") \
                .set_model_name(self.config['DEFAULTMODEL1'])
            config.add_system_information = False
            config.stop_sequence = ["\n"]

        if name == "liveInterpretation":
            config. \
                set_mode("live") \
                .set_max_iterations(12) \
                .set_completion_mode("chat") \
                .set_model_name(self.config['DEFAULTMODEL0']).stream = True
            config.stop_sequence = ["!X!"]

        if name == "isaa-chat-web":
            config. \
                set_mode("talk") \
                .set_max_iterations(1) \
                .set_completion_mode("chat")

        if name == "summary":
            config. \
                set_mode("free") \
                .set_max_iterations(2) \
                .set_completion_mode("text") \
                .set_model_name(self.config['DEFAULTMODELSUMMERY']) \
                .set_pre_task("Write a Summary of the following text :")
            # 'text-curie-001'
            # text-davinci-003 text-babbage-001 curie
            config.stop_sequence = ["\n\n\n"]
            config.stream = False
            config.add_system_information = False

        if name == "thinkm":
            config. \
                set_mode("free") \
                .set_model_name(self.config['DEFAULTMODEL1']) \
                .set_max_iterations(1) \
                .set_completion_mode("chat")

            config.stop_sequence = ["\n\n\n"]

        if name == "code":
            config. \
                set_mode("free") \
                .set_model_name(self.config['DEFAULTMODELCODE']) \
                .set_max_iterations(3) \
                .set_completion_mode("edit")

        if name.startswith("chain_search"):

            config.mode = "tools"
            config.model_name = self.config['DEFAULTMODEL1']

            config.set_agent_type("self-ask-with-search")
            config.max_iterations = 4

            config.personality = """
            Innovative: Employ advanced search techniques to retrieve relevant and reliable information from diverse web sources.
            Analytical: Analyze found data, identifying key points and themes.
            Efficient: Rapidly search and summarize information, delivering precise and accurate results.
            Adaptive: Modify search and summarization strategies based on the user query and available data.
            Detail-Oriented: Maintain a keen focus on the details of the information, ensuring accuracy and relevance in the summaries."""

            config.goals = """
            1. Information Retrieval: Primary goal is to locate pertinent and reliable web information in response to user queries.
            2. Text Summarization: Condense retrieved data into clear and concise summaries, encapsulating the most crucial points and ideas.
            3. Relevance Identification: Assess the relevance of found information, ensuring it meets the user's needs and expectations.
            4. Source Evaluation: Evaluate the credibility and reliability of the sources, providing users with trustworthy information.
            5. Continuous Improvement: Refine search algorithms and summarization techniques continually to enhance result quality and relevance over time."""
            config.short_mem.max_length = 3500

            if name.endswith("_web"):
                config.tools = {
                    "Intermediate Answer": {"func": search_text,
                                            "description": "Use Duck Duck go to search the web systax <qustion>"},
                }

            if name.endswith("_url"):
                config.tools = {
                    "Intermediate Answer": {"func": browse_url,
                                            "description": "browse web page via URL syntax <url>|<qustion>"},
                }
            if name.endswith("_memory"):
                config.tools = {
                    "Intermediate Answer": {"func": memory_search,
                                            "description": "Serch for simmilar memory imput <context>"},
                }

            config.task_list = ["Complete the task in as few steps and as carefully as possible."]

        path = self.observation_term_mem_file
        if not self.agent_collective_senses:
            path += name + ".mem"
        else:
            path += 'CollectiveObservationMemory.mem'

        try:
            if not os.path.exists(path):
                self.print("Crating Mem File")
                if not os.path.exists(self.observation_term_mem_file):
                    os.makedirs(self.observation_term_mem_file)
                with open(path, "a") as f:
                    f.write("[]")

            with open(path, "r") as f:
                mem = f.read()
                if mem:
                    config.observe_mem.text = str(mem)
                else:
                    with open(path, "a") as f:
                        f.write("[]")
        except FileNotFoundError and ValueError:
            print("File not found | mem not saved")

        mem = self.get_context_memory()

        def get_relevant_informations(x):
            ress = mem.get_context_for(x)

            task = f"Act as an summary expert your specialties are writing summary. you are known to think in small and " \
                   f"detailed steps to get the right result. Your task : write a summary reladet to {x}\n\n{ress}"
            res = self.run_agent(self.get_default_agent_config('think').set_model_name('gpt-3.5-turbo-0613'), task)

            if res:
                return res

            return ress

        def ad_data(x):
            mem.add_data('main', x)

            return 'added to memory'

        def crate_task_wrapper(task):
            if task:
                self.print(Style.GREEN("Crating Task"))
                chan_name = self.create_task(task)
                dis = self.agent_chain.get_discr(name)
                return (f"Crating Chain was a Success. The chain can be run with run_task_chain\nName: {chan_name}\n"
                        f"Description: {dis}\n")
            self.print(Style.YELLOW("Not Task specified"))

        self.add_tool("memory", get_relevant_informations, "a tool to get similar information from your memories."
                                                           " useful to get similar data. ",
                      "memory(<related_information>)",
                      config)

        self.add_tool("save_data_to_memory", ad_data, "tool to save data to memory,"
                                                      " write the data as specific"
                                                      " and accurate as possible.",
                      "save_data_to_memory(<store_information>)",
                      config)

        self.add_tool("crate_task_chain", crate_task_wrapper,
                      "tool to crate a task for a subject input subject:str",
                      "crate_task_chain(<subject>)",
                      config)

        # self.add_tool("optimise_task_chain", self.optimise_task, "tool to optimise a task enter task name",
        #               "optimise_task_chain(<subject>)",
        #               config)

        self.add_tool("run_task_chain", lambda name: str(self.run_chain_on_name(name, "$user-input")),
                      "tool to run a crated chain"
                      " with an task as objective",
                      "run_task_chain(<chain_name>,<objective>)",
                      config)

        def create_agent(Name: str or None = None,
                         # Mode: str or None = None,
                         Personal: str or None = None,
                         Goals: str or None = None,
                         Capabilities: str or None = None
                         ) -> str:

            """
    The create_agent These pairs specify various attributes of an agent that is to be created and run.

    The function parses the input string x and extracts the values associated with the following keys:

        Name: The name of the agent to be created. This key is required and must be present in the input string.
        Mode: The mode in which the agent is to be run. This is an optional key. available ar [free, execution]
        Personal: The personality of the agent. This is an optional key.
        Goals: The goals of the agent. This is an optional key.
        Capabilities: The capabilities of the agent. This is an optional key.

    The function then creates an AgentConfig object with the specified name and sets its personality, goals, and capabilities attributes to the values associated with the corresponding keys, if those keys were present in the input string."""
            # personal: Optional[str] = None
            # goals: Optional[str] = None
            # name: Optional[str] = None
            # capabilities: Optional[str] = None
            # mode: Optional[str] = None

            if not Name:
                return "ValueError('Agent name must be specified.')"

            agent_config: AgentConfig = self.get_agent_config_class(Name)
            agent_config.tools = agent_config.tools

            if Personal is not None:
                agent_config.personality = Personal

            if Goals is not None:
                agent_config.goals = Goals

            if Capabilities is not None:
                agent_config.capabilities = Capabilities

            #if Mode is not None:
                #if Mode not in agent_config.available_modes:
                #    return f"Unknown mode : avalabel ar : {agent_config.available_modes}"
            agent_config.set_mode("tools")

            # if Task is not None:
            #     return self.run_agent(agent_config, Task, mode_over_lode=Mode)

            return f"Agent {Name} created."

        self.add_tool("spawn_agent", create_agent,
                      "The create_agent These pairs specify various attributes of an agent that is to be created and run."
                      , f""" The function parses the input string x and extracts the values associated with the following keys:

        Name: The name of the agent to be created. This key is required and must be present in the input string.
        Personal: The personality of the agent. This is an optional key.
        Goals: The goals of the agent. This is an optional key.
        Capabilities: The capabilities of the agent. This is an optional key.

    The function then creates an AgentConfig object with the specified name and sets its personality, goals, and capabilities attributes to the values associated with the corresponding keys, if those keys were present in the input string.
    """,
                      config)
        self.add_tool("run_agent", lambda agent_name, task: self.run_agent(agent_name, task),
                      "The run_agent function takes a 2 arguments agent_name, task"
                      , """The function parses the input string x and extracts the values associated with the following keys:

                   Name: The name of the agent to be created. This key is required and must be present in the input string.
                   Task: The task that the agent is to perform. This is an optional key.

               The function then runs the Agent with the specified name.""",
                      config)

        self.add_tool("get_avalabel_agents", lambda: self.config['agents-name-list'],
                      "Use to get list of all agents avalabel"
                      , """get_avalabel_agents()""",
                      config)

        toolkit = FileManagementToolkit(
            root_dir=str(self.working_directory)
        )  # If you don't provide a root_dir, operations will default to the current working directory
        for file_tool in toolkit.get_tools():
            self.lang_chain_tools_dict[file_tool.name] = file_tool

        self.add_lang_chain_tools_to_agent(config, config.tools)

        return config

    def remove_agent_config(self, name):
        del self.config[f'agent-config-{name}']
        self.config["agents-name-list"].remove(name)

    def get_agent_config_class(self, agent_name="Normal") -> AgentConfig:

        if "agents-name-list" not in self.config.keys():
            self.config["agents-name-list"] = []

        if agent_name in self.config["agents-name-list"]:
            config = self.config[f'agent-config-{agent_name}']
            self.print(f"collecting AGENT: {config.name} {config.mode}")
        else:
            self.config["agents-name-list"].append(agent_name)
            config = self.get_default_agent_config(agent_name)
            self.config[f'agent-config-{agent_name}'] = config
            self.print(f"Init:Agent::{agent_name}:{config.name} {config.mode}\n")

        return config

    def mini_task_completion(self, mini_task):
        agent = self.get_agent_config_class("TaskCompletion")
        return self.stream_read_llm(mini_task, agent)

    def create_task(self, task: str, agent_execution=None):
        agent = self.get_agent_config_class("self")
        if agent_execution is None:
            agent_execution = self.get_agent_config_class("self")
        agent_execution.get_messages(create=True)

        task_generator = self.agent_chain.get("Task Generator")
        if not task_generator:
            task_generator = [
                {
                    "use": "tool",
                    "name": "memory",
                    "args": "$user-input",
                    "return": "$D-Memory"
                },
                {
                    "use": "agent",
                    "mode": "generate",
                    "name": "self",
                    "args": "Erstelle Eine Prompt für den Nächsten Agent."
                            "Der Agent soll eine auf das Subject angepasste task erhalten."
                            "Die prompt soll auf das Subject und die informationen"
                            " angepasst sein"
                            "Subject : $user-input"
                            "informationen die das system zum Subject hat: $D-Memory.",
                    "return": "$task"
                },
                {
                    "use": "agent",
                    "mode": "conversation",
                    "name": "self",
                    "args": "Finde einene Lösungs ansatz und gebe weiter information zu subject zurück"
                            "Subject : $user-input"
                            "informationen die das system zum Subject hat: $D-Memory.",
                    "return": "$infos"
                },
            ]
            self.agent_chain.add("Task Generator", task_generator)
        chain_data = {}
        """
        "Informationen : $infos \nNur die Aufgabenliste zurückgeben, sonst nichts!!!\nDetails für das Format:\n"
                            "die benutzereingabevariable ist (var zeichen)[$](var name)[benutzereingabe] immer die variabel in die erste eingabe einfügen"
                            "Aufgabenformat:\n"
                            "Schlüssel, die enthalten sein müssen [use,name,args,mode,return]\n"
                            f "Werte für use ['agent', 'tool']\n"
                            f "Werte für name wenn use='agent' {self.config['agents-name-list']}\n"
                            f "Werte für name if use='tool' {tools_list}\n"
                            "args: str = Befehl für den Agenten oder das Werkzeug"
                            "return = optionaler Rückgabewert, speichert den Rückgabewert in einer Variablen zur späteren Verwendung expel"
                            " $return-from-task1 -> args für nächste Aufgabe 'validate $return-from-task1'"
                            f "if use='agent' mode = {agent.available_modes}"
                            "return format : [task0,-*optional*-taskN... ]"
                            "Beispiel Aufgabe:dict = {'use':'agent','name':'self','args':'Bitte stell dich vor',"
                            "'mode':'free','return':'$return'}"
                            "versucht, nur eine Aufgabe zurückzugeben, wenn der Betreff mehrere größere Schritte enthält"
                            "mehrere Aufgaben in einer Liste zurückgeben."
                            "Tipp: Variablen werden mit $ angekündigt, also ist $a eine Variable a. Variablen werden durch den "
                            "Rückgabewert, so dass Informationen über Variablen versteckt werden können. die erste Variable, die "
                            " "immer verfügbar ist, ist die Benutzereingabe. Imparativ"
                            " "return_val:List[dict] = "
        """
        res, chain_data, _ = self.execute_thought_chain(task, task_generator + [{
            "use": "agent",
            "name": "think",
            "args": "$task "
                    "Informationen : $infos \nOnly return the task list nothing else!!\nDetails für das format:\n"
                    "the user input variabel is (var sign)[$](var name)[user-input] allways include the variabel in the first input.\n"
                    "Task format:\n"
                    "Keys that must be included [use,name,args,mode,return]\n"
                    "values for use ['agent', 'chain']\n"
                    f"values for name if use='agent' {self.config['agents-name-list']}\n"
                    f"values for name if use='chain' {str(self.agent_chain)}\n"
                    "args: str = task for the agent or chain"
                    "return = optional return value, stor return value in an variabel for later use expel"
                    " $return-from-task1 -> args for next task 'validate $return-from-task1'"
                    f"if use='agent' mode = {agent.available_modes}"
                    "return format : [task0,-*optional*-taskN... ]"
                    "try to return only one task if the subject includes mutabel bigger steppes"
                    "return multiple tasks in a list."
                    "tip: variables are announced with $ so $a is a variable. variables are decarated by the "
                    "return value. the first variable is "
                    "always user-input use it! Must use variables in the args keyword!!"
                    "!!!Note that the chain must be written in the imperative.!!!\n"
                    "return_val:List[dict] = "
            ,
            "return": "$taskDict",
        }], agent_execution, chain_data=chain_data,
                                                        chain_data_infos=True)
        task_list = []
        try:
            if '$taskDict' in chain_data.keys():
                task_list = chain_data['$taskDict']
            else:
                task_list = res[-1][-1]
            task_list = anything_from_str_to_dict(task_list)
            if isinstance(task_list, dict):
                task_list = [task_list]
        except ValueError as e:
            self.print_stream(Style.RED("Error parsing auto task builder"))
            self.logger.error(Style.RED(f"Error in auto task builder {e}"))

        if not isinstance(task_list, list):
            if isinstance(task_list, str):
                if task_list.startswith("{") and task_list.endswith("}"):
                    task_list = anything_from_str_to_dict(task_list)
                if task_list.startswith("[") and task_list.endswith("]"):
                    task_list = eval(task_list)
            if isinstance(task_list, dict):
                task_list = [task_list]

        task_name = self.mini_task_completion(f"Crate a name for this task {task_list} subject {task}\nTaskName:")
        if not task_name:
            task_name = self.process_completion(f"Crate a name for this task {task_list} subject {task}\nTaskName:",
                                                self.get_agent_config_class("think"))
        if not task_name:
            task_name = str(uuid.uuid4())
        task_name = task_name.strip()
        self.print(Style.Bold(Style.CYAN(f"TASK:{task_name}:{task_list}:{type(task_list)}####")))
        self.agent_chain.add(task_name, task_list)
        self.agent_chain.init_chain(task_name)
        self.describe_chain(task_name)
        return task_name

    def optimise_task(self, task_name):
        task_dict = []

        agent = self.get_agent_config_class("self")
        agent_execution = self.get_agent_config_class("optimise_task")
        agent_execution.get_messages(create=True)
        task = self.agent_chain.get(task_name)
        optimise_genrator = self.agent_chain.get("Task O Generator")
        if not optimise_genrator:
            optimise_genrator = [
                {
                    "use": "tool",
                    "name": "memory",
                    "args": "$user-input",
                    "return": "$D-Memory"
                },
                {
                    "use": "agent",
                    "name": "self",
                    "args": "Brainstorm about the users requesst $user-input find ways to improve it"
                            " consider all avalabel information"
                            "informationen die das system zum Subject hat: $D-Memory."
                    ,
                    "return": "$infos"
                },
                {
                    "use": "agent",
                    "mode": "generate",
                    "name": "self",
                    "args": "Erstelle Eine Prompt für den Nächsten Agent."
                            "Der Agent soll eine auf das Subject angepasste task Optimireren"
                            "Der Agent soll beachten das die Task Im korrekten json format ist. und das alle attribute"
                            " richtig ausgewählt werden sollen. Die prompt soll auf das Subject und die informationen"
                            " angepasst sein"
                            "Subject : $user-input"
                            "informationen die das system zum Subject hat: $D-Memory. $infos.",
                    "return": "$task"
                },
            ]
            self.agent_chain.add("Task O Generator", optimise_genrator)
        _, data, _ = self.execute_thought_chain(str(task), optimise_genrator + [{
            "use": "agent",
            "mode": "free",
            "name": "execution",
            "args": "$task Details für das format:\n"
                    "Task format:\n"
                    "Keys that must be included [use,name,args,mode,return]\n"
                    "values for use ['agent', 'chain']\n"
                    f"values for name if use='agent' {self.config['agents-name-list']}\n"
                    f"values for name if use='chain' {str(self.agent_chain)}\n"
                    "args: str = user-input for the agent or chain"
                    "return = optional return value, stor return value in an variabel for later use expel"
                    " $return-from-task1 -> args for next task 'validate $return-from-task1'"
                    f"if use='agent' mode = {agent.available_modes}"
                    "return format : [task0,-*optional*-taskN... ]"
                    "example task = {'use':'agent','name':'self','args':'Bitte stell dich vor',"
                    "'mode':'free','return':'$return'}"
                    "try to return only one task if the subject includes mutabel bigger steppes"
                    "return multiple tasks in a list."
            ,
            "return": "$taskDict",
        }], agent_execution, chain_data_infos=True)
        try:
            if '$taskDict' in data.keys():
                task_list = data['$taskDict']
            else:
                task_list = data[-1][-1]
            task_dict = anything_from_str_to_dict(task_list)
            if isinstance(task_list, dict):
                task_dict = [task_list]
        except ValueError as e:
            self.print_stream(Style.RED("Error parsing auto task builder"))
            self.logger.error(Style.RED(f"Error in auto task builder {e}"))
        return task_dict

    def test_use_tools(self, agent_text: str, config: AgentConfig) -> tuple[bool, Any, Any]:
        if not agent_text:
            return False, "", ""

        self.logger.info("Start testing tools")

        # print(f"_extract_from_json, {agent_text}")

        action, inputs = _extract_from_json(agent_text.replace("'", '"'), config)
        # print(f"{action=}| {inputs=} {action in config.tools.keys()=}")
        if action and action in config.tools.keys():
            return True, action, inputs

        # print("_extract_from_string")
        action, inputs = _extract_from_string(agent_text, config)
        # print(f"{action=}| {inputs=} {action in config.tools.keys()=}")
        if action and action in config.tools.keys():
            return True, action, inputs

        # print("_extract_from_string")
        action, inputs = _extract_from_string_de(agent_text, config)
        # print(f"{action=}| {inputs=} {action in config.tools.keys()=}")
        if action and action in config.tools.keys():
            return True, action, inputs

        self.logger.info("Use AI function to determine the action")
        action = self.mini_task_completion(
            f"Is one of the tools called in this line '''{agent_text}''' Avalabel tools: {list(config.tools.keys())}? If yes, answer with the tool name, if no, then with NONE. Answer:\n")
        action = action.strip()
        self.logger.info(f"Use AI : {action}")
        self.print(f"Use AI : {action}")
        # print(action in list(config.tools.keys()), list(config.tools.keys()))
        if action in list(config.tools.keys()):
            inputs = agent_text.split(action, 1)
            if len(inputs):
                inputs = inputs[1].strip().replace("Inputs: ", "")
            print(f"{action=}| {inputs=} ")
            return True, action, inputs

        return False, "", ""

    @staticmethod
    def test_task_done(agent_text):

        done = False

        if not ":" in agent_text:
            done = True

        for line in agent_text.split("\n"):

            if line.startswith("Answer:"):
                done = True

            if line.startswith("Thought: I now know the final answer"):
                done = True

            if "Final Answer:" in line:
                done = True

        return done

    @staticmethod
    def parse_arguments(command: str, sig) -> (list, dict):
        # Initialisierung der Ausgabeliste und des Wörterbuchs
        out_list = []
        out_dict = {}
        param_keys = list(sig.parameters)

        # Überprüfung, ob der Befehl ein Wörterbuch enthält
        if isinstance(command, dict):
            command = json.dumps(command)

        if "{" in command and "}" in command:
            s = {}
            for x in param_keys:
                s[x] = None
            arg_dict = anything_from_str_to_dict(command, expected_keys=s)

            # Überprüfung, ob es nur einen falschen Schlüssel und einen fehlenden gültigen Schlüssel gibt

            missing_keys = [key for key in param_keys if key not in arg_dict]
            extra_keys = [key for key in arg_dict if key not in param_keys]

            if len(missing_keys) == 1 and len(extra_keys) == 1:
                correct_key = missing_keys[0]
                wrong_key = extra_keys[0]
                arg_dict[correct_key] = arg_dict.pop(wrong_key)
            out_dict = arg_dict
        else:
            # Aufteilung des Befehls durch Komma
            if len(param_keys) == 0:
                pass
            elif len(param_keys) == 1:
                out_list.append(command)
            elif len(param_keys) >= 2:

                comma_cont = command.count(',')
                saces_cont = command.count(' ')
                newline_cont = command.count('\n')
                split_key = "-"
                if comma_cont == len(param_keys)-1:
                    split_key = ","
                elif newline_cont == len(param_keys)-1:
                    split_key = "\n"
                elif saces_cont == len(param_keys)-1:
                    split_key = " "

                print(f"{len(param_keys)=}\n{comma_cont}\n{saces_cont}\n{newline_cont}")

                if len(param_keys) == 2:
                    if split_key == "-":
                        split_key = ","
                        pos_space = command.find(" ")
                        pos_comma = command.find(",")
                        if pos_space < pos_comma:
                            split_key = " "
                    args = [arg.strip() for arg in command.split(split_key)]
                    args = [args[0], split_key.join(args[1:])]
                else:
                    args = [arg.strip() for arg in command.split(split_key)]

                # Befüllen des Wörterbuchs und der Liste basierend auf der Signatur
                print(f"{args=}\n{split_key=}\n")
                for i, arg in enumerate(args):
                    if i < len(param_keys):
                        out_dict[param_keys[i]] = arg
                    else:
                        out_list.append(arg)

        return out_list, out_dict

    def run_tool(self, command: str, function_name: str, config: AgentConfig):

        for func in config.tools.keys():
            if function_name.lower().strip() == func.lower().strip():
                function_name = func
                break
        else:
            self.logger.error(f"Unknown Function {function_name}. Valid functions are: {config.tools.keys()}")
            return f"Unknown Function {function_name}. Valid functions are: {config.tools.keys()}"

        tool = config.tools[function_name]
        sig = signature(tool['func'])
        len_para = len(list(sig.parameters))
        self.logger.info(f"Running: {function_name} with signature: {sig}")
        positional_args, keyword_args = self.parse_arguments(command, sig)

        print(f"{positional_args=}\n{keyword_args=}\n")

        try:
            if sig.parameters:
                # Prüfen, ob die Anzahl der Parameter mit den aus parse_arguments übereinstimmt
                if len(sig.parameters) == len(positional_args) + len(keyword_args.keys()):
                    # Prüfen, ob die Funktion nur einen Parameter erwartet und mehrere Positional Arguments vorhanden sind
                    if len(sig.parameters) == len(positional_args):
                        observation = tool['func'](*positional_args)
                    elif len(sig.parameters) == len(keyword_args.keys()):
                        observation = tool['func'](**keyword_args)
                    else:
                        observation = tool['func'](*positional_args, **keyword_args)
                else:
                    # Wenn die Anzahl der Parameter nicht übereinstimmt, versuchen wir, die Keyword Arguments in Positional Arguments umzuwandeln
                    for key in keyword_args.keys():
                        if key not in sig.parameters:
                            positional_args.append(keyword_args[key])
                    observation = tool['func'](*positional_args)
            else:
                observation = tool['func']()
        except Exception as e:
            self.logger.error(f"Fatal error in tool {function_name}: {str(e)}")
            observation = f"Fatal error in tool {function_name}: {str(e)}"

        self.logger.info(f"Observation: {observation}")

        path = self.observation_term_mem_file
        if not self.agent_collective_senses:
            path += config.name + ".mem"
        else:
            path += 'CollectiveObservationMemory.mem'

        with open(path, "w") as f:
            try:
                f.write(str(observation))
            except UnicodeEncodeError:
                self.logger.error("Memory not encoded properly")

        if isinstance(observation, dict):
            observation = self.summarize_dict(observation, config)

        if not observation:
            observation = "Problem running function, try running with more details"

        if not isinstance(observation, str):
            observation = str(observation)

        config.short_mem.text = observation

        return observation

    def run_agent(self, name: str or AgentConfig, text: str, mode_over_lode: str or None = None, r=2.0):
        config = None
        if isinstance(name, str):
            config = self.get_agent_config_class(name)

        if isinstance(name, AgentConfig):
            config = name
            name = config.name

        if mode_over_lode:
            mode_over_lode, config.mode = config.mode, mode_over_lode

        self.print(f"Running agent {name} {config.mode}")

        # print("AGENT:CONFIG"+str(config)+"ENDE\n\n\n")

        out = "Invalid configuration\n"
        stream = config.stream
        self.logger.info(f"stream mode: {stream} mode : {config.mode}")
        if config.mode == "talk":
            sto, config.add_system_information = config.add_system_information, False
            prompt = PromptTemplate(
                input_variables=["input"],
                template=config.prompt,
            )
            out = LLMChain(prompt=prompt, llm=self.get_llm_models(config.model_name)).run(text)
            config.add_system_information = sto
        elif config.mode == "tools":
            sto, config.add_system_information = config.add_system_information, False

            try:
                prompt = PromptTemplate(
                    input_variables=["input"],
                    template=config.prompt,
                )
            except Exception as e:
                return f"ERROR: Could not parse prompt {e}"
            tools = []

            for tool_name in config.tools.keys():

                return_direct = False
                if 'return_direct' in config.tools[tool_name].keys():
                    return_direct = True
                if 'langchain-tool' in list(config.tools[tool_name].keys()):
                    tools.append(config.tools[tool_name]["func"])
                else:
                    from langchain.tools import StructuredTool
                    tools.append(StructuredTool.from_function(func=config.tools[tool_name]["func"],
                                                              name=tool_name,
                                                              description=config.tools[tool_name]["description"],
                                                              return_direct=return_direct
                                                              ))
            agent_type = config.agent_type
            # if agent_type in ["structured-chat-zero-shot-react-description"]:
            #    if text:
            #        config.step_between = text
            #    out = initialize_agent(tools, prompt=prompt,
            #                           llm=self.get_llm_models(config.model_name),
            #                           agent=agent_type, verbose=config.verbose,
            #                           max_iterations=config.max_iterations).run(text)
            #    print(out)
            # else:
            try:
                out = initialize_agent(tools, prompt=prompt,
                                       llm=self.get_llm_models(config.model_name),
                                       agent=agent_type, verbose=config.verbose,
                                       return_intermediate_steps=True,
                                       max_iterations=config.max_iterations)(text)
            except Exception as e:
                out = e
                if r > 0:
                    return self.run_agent(name,
                                          text +str(e)+ "\nFocus on using the rigt input for the actions som take just a string as input",
                                          r=r - 1)
            if agent_type not in ["structured-chat-zero-shot-react-description"]:
                out = self.summarize_dict(out, config)
            config.add_system_information = sto
        elif config.mode == "conversation":
            sto, config.add_system_information = config.add_system_information, False
            prompt = PromptTemplate(
                input_variables=["input", "history"],
                template='{history}' + config.prompt,

            )
            out = ConversationChain(prompt=prompt, llm=self.get_llm_models(config.model_name)).predict(input=text)
            config.add_system_information = sto
        elif config.mode == "live":
            self.logger.info(f"stream mode: {stream}")

            all_description = ""

            for key in self.agent_chain.chains.keys():
                if "Task Generator" in key or "Task-Generator" in key:
                    continue
                des = self.agent_chain.get_discr(key)
                if des is None:
                    des = key
                all_description += f"NAME:{key} \nUse case:{des}"

            config.capabilities = all_description
            config.onLiveMode = ""
            config.action_called = False
            config.task_list = []
            config.task_index = 0

            def olivemode(line: str):
                modes = ["ACTION:", "TALK:", "SPEAK:", "THINK:", "PLAN:"]
                if not line:
                    config.onLiveMode = ""
                    return False
                for mode in modes:
                    if mode in line.upper() or line.startswith('{'):
                        config.onLiveMode = mode
                        return mode
                if config.onLiveMode:
                    return config.onLiveMode
                return False

            def online(line):
                mode = olivemode(line)
                if not mode:
                    return

                self.print("Mode: " + mode)

                if mode == "ACTION:":
                    use_tool, func_name, command_ = self.test_use_tools(line, config)
                    self.logger.info(f"analysing test_task_done")
                    task_done = self.test_task_done(line)
                    if use_tool:
                        self.print(f"Using-tools: {func_name} {command_}")
                        ob = self.run_tool(command_, func_name, config)
                        # config.observe_mem.text = ob
                        config.add_message("system", f"The tool : {func_name} result : {ob}")
                        self.print(f"Observation: {ob}")
                        config.short_mem.text = ob
                        config.onLiveMode = ""
                        config.action_called = True
                        config.next_task()
                    if task_done:  # new task
                        self.print(f"Task done")
                        # self.speek("Ist die Aufgabe abgeschlossen?")
                        if config.short_mem.tokens > 50:
                            config.short_mem.clear_to_collective()
                        config.onLiveMode = ""
                        config.action_called = False
                if mode == "PLAN:":
                    config.task_list.append(line)

                if mode == "SPEAK:":
                    self.speak(line.replace("SPEAK:", ""))
                    config.onLiveMode = ""

            last_call = False

            for turn in range(config.max_iterations):
                self.print(f"\n=================== Enter Turn : {turn} of {config.max_iterations} =================\n")

                if config.stream:
                    out = self.stream_read_llm(text, config, line_interpret=True, interpret=online)
                    if not "\n" in out:
                        online(out)
                else:
                    out = self.stream_read_llm(text, config)
                    online(out)
                if not stream:
                    self.print_stream("execution-free : " + out)

                config.add_message("assistant", self.mas_text_summaries(out))

                if out.endswith("EVAL") and r:
                    continue
                if config.action_called and r:
                    continue
                else:
                    if last_call:
                        break
                    text = "This is the last Call Report to the user!"
                    last_call = True

        else:
            out = self.stream_read_llm(text, config)
            if not stream:
                self.print_stream(out)
            else:
                print("\n------stream-end------")

            config.observe_mem.text = out

        # except NameError and Exception as e:
        #    print(f"ERROR runnig AGENT: {name} retrys {retrys} errormessage: {e}")
        #    res = ""
        #    if retrys:
        #        res = self.run_agent(name, text, config, mode_over_lode, retrys-1)
        #        if not isinstance(res, str):
        #            print(res)
        #            res = res['output']
        #    else:
        #        return f"\nERROR runnig agent named: {name} retrys {str(retrys)} errormessage: {str(e)}\n{str(res)}"

        if config.mode not in ["free", 'conversation'] and isinstance(out, str):
            py_code, type_ = extract_code(out)
            if type_.lower() == 'python':
                self.print("Executing Python code")
                py_res = self.config[f'agent-config-{name}'].python_env.run_and_display(py_code)
                out += '\n\nPython\n' + py_res
                self.print(f"Result : {py_res}\n")

        config.short_mem.text = f"\n\n{config.name} RESPONSE:\n{out}\n\n"

        if mode_over_lode:
            mode_over_lode, config.mode = config.mode, mode_over_lode

        return out

    def execute_thought_chain(self, user_text: str, agent_tasks, config: AgentConfig, speak=lambda x: x, start=0,
                              end=None, chain_ret=None, chain_data=None, uesd_mem=None, chain_data_infos=False):
        if uesd_mem is None:
            uesd_mem = {}
        if chain_data is None:
            chain_data = {}
        if chain_ret is None:
            chain_ret = []
        if end is None:
            end = len(agent_tasks) + 1
        ret = ""

        default_mode_ = config.mode
        default_completion_mode_ = config.completion_mode
        config.completion_mode = "chat"
        config.get_messages(create=True)
        sto_name = config.name
        sto_config = None
        chain_mem = self.get_context_memory()
        self.logger.info(Style.GREY(f"Starting Chain {agent_tasks}"))
        config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

        invalid = False
        error = ""
        if not isinstance(agent_tasks, list):
            self.print(Style.RED(f"tasks must be list ist: {type(agent_tasks)}:{agent_tasks}"))
            error = "tasks must be a list"
            invalid = True
        if len(agent_tasks) == 0:
            self.print(Style.RED("no tasks specified"))
            error = "no tasks specified"
            invalid = True

        if invalid:
            if chain_data_infos:
                return chain_ret, chain_data, uesd_mem
            else:
                return error, chain_ret

        work_pointer = start
        running = True
        while running:

            task = agent_tasks[work_pointer]

            self.logger.info(Style.GREY(f"{type(task)}, {task}"))
            chain_ret_ = []
            config.mode = "free"
            config.completion_mode = "chat"

            sum_sto = ""

            keys = list(task.keys())

            task_name = task["name"]
            use = task["use"]
            args = task["args"].replace("$user-input", user_text)

            if use == 'agent':
                sto_config, config = config, self.get_agent_config_class(task_name)
            else:
                config = self.get_agent_config_class('self')

            default_mode = config.mode
            default_completion_mode = config.completion_mode

            if 'mode' in keys:
                config.mode = task['mode']
                self.logger.info(Style.GREY(f"In Task {work_pointer} detected 'mode' {config.mode}"))
            if 'completion-mode' in keys:
                config.completion_mode = task['completion-mode']
                self.logger.info(
                    Style.GREY(f"In Task {work_pointer} detected 'completion-mode' {config.completion_mode}"))
            if "infos" in keys:
                config.short_mem.text += task['infos']
                self.logger.info(Style.GREY(f"In Task {work_pointer} detected 'info' {task['infos'][:15]}..."))

            chain_data['$edit-text-mem'] = config.edit_text.text

            for c_key in chain_data.keys():
                if c_key in args:
                    args = args.replace(c_key, str(chain_data[c_key]))

            if use == 'chain':
                for c_key in chain_data.keys():
                    if c_key in task_name:
                        task_name = task_name.replace(c_key, str(chain_data[c_key]))

            self.print(f"Running task: {args}")

            speak(f"Chain running {task_name} at step {work_pointer} with the input : {args}")

            if 'chuck-run-all' in keys:
                self.logger.info(Style.GREY(f"In Task {work_pointer} detected 'chuck-run-all'"))
                chunk_num = -1
                for chunk in chain_data[task['chuck-run-all']]:
                    chunk_num += 1
                    self.logger.info(Style.GREY(f"In chunk {chunk_num}"))
                    if not chunk:
                        self.logger.warning(Style.YELLOW(f"In chunk {chunk_num} no detected 'chunk' detected"))
                        continue

                    self.logger.info(Style.GREY(f"detected 'chunk' {str(chunk)[:15]}..."))

                    args_ = args.replace(task['chuck-run-all'], str(chunk))

                    ret, sum_sto, chain_ret_ = self.chain_cor_runner(use, task_name, args_, config, sto_name, task,
                                                                     work_pointer, keys,
                                                                     chain_ret, sum_sto)

            elif 'chuck-run' in keys:
                self.logger.info(Style.GREY(f"In Task {work_pointer} detected 'chuck-run'"))
                rep = chain_mem.vector_store[uesd_mem[task['chuck-run']]]['represent']
                if len(rep) == 0:
                    self.get_context_memory().crate_live_context(uesd_mem[task['chuck-run']])
                    rep = chain_mem.vector_store[uesd_mem[task['chuck-run']]]['represent']
                if len(rep) == 0:
                    final = chain_mem.search(uesd_mem[task['chuck-run']], args)
                    if len(final) == 0:
                        final = chain_mem.get_context_for(args)

                    action = f"Act as an summary expert your specialties are writing summary. you are known to " \
                             f"think in small and " \
                             f"detailed steps to get the right result. Your task : write a summary reladet to {args}\n\n{final}"
                    t = self.get_agent_config_class('thinkm')
                    ret = self.run_agent(t, action)
                ret_chunk = []
                chunk_num = -1
                for chunk_vec in rep:
                    chunk_num += 1
                    self.logger.info(Style.GREY(f"In chunk {chunk_num}"))
                    if not chunk_vec:
                        self.logger.warning(Style.YELLOW(f"In chunk {chunk_num} no detected 'chunk' detected"))
                        continue

                    chunk = chain_mem.hydrate_vectors(uesd_mem[task['chuck-run']], chunk_vec)

                    args_ = args.replace(task['chuck-run'], str(chunk[0].page_content))

                    ret, sum_sto, chain_ret_ = self.chain_cor_runner(use, task_name, args_, config, sto_name, task,
                                                                     work_pointer,
                                                                     keys,
                                                                     chain_ret, sum_sto)
                    ret_chunk.append(ret)
                ret = ret_chunk

            else:

                ret, sum_sto, chain_ret_ = self.chain_cor_runner(use, task_name, args, config, sto_name, task,
                                                                 work_pointer,
                                                                 keys,
                                                                 chain_ret, sum_sto)

            # if 'validate' in keys:
            #     self.print("Validate task")
            #     try:
            #         pipe_res = self.text_classification(ret)
            #         self.print(f"Validation :  {pipe_res[0]}")
            #         if pipe_res[0]['score'] > 0.8:
            #             if pipe_res[0]['label'] == "NEGATIVE":
            #                 print('🟡')
            #                 if 'on-error' in keys:
            #                     if task['validate'] == 'inject':
            #                         task['inject'](ret)
            #                     if task['validate'] == 'return':
            #                         task['inject'](ret)
            #                         chain_ret.append([task, ret])
            #                         return "an error occurred", chain_ret
            #             else:
            #                 print(f'🟢')
            #     except Exception as e:
            #         print(f"Error in validation : {e}")

            if 'to-edit-text' in keys:
                config.edit_text.text = ret

            chain_data, chain_ret, uesd_mem = self.chain_return(keys, chain_ret_, task, task_name, ret,
                                                                chain_data, uesd_mem, chain_ret)

            self.print(Style.ITALIC(Style.GREY(f'Chain at {work_pointer}\nreturned : {str(ret)[:150]}...')))

            if sto_config:
                config = sto_config
                sto_config = None

            config.mode = default_mode
            config.completion_mode = default_completion_mode

            if 'brakeOn' in keys:
                do_brake = False
                if isinstance(task['brakeOn'], list):
                    for b in task['brakeOn']:
                        if b in ret:
                            do_brake = True

                if isinstance(task['brakeOn'], str):

                    if task['brakeOn'] in ret:
                        do_brake = True

                if isinstance(task['brakeOn'], bool):

                    if task['brakeOn']:
                        do_brake = True

                running = not do_brake

            work_pointer += 1
            if work_pointer >= end or work_pointer >= len(agent_tasks):
                running = False

        config.mode = default_mode_
        config.completion_mode = default_completion_mode_

        if chain_data_infos:
            return chain_ret, chain_data, uesd_mem

        chain_sum_data = self.summarize_ret_list(chain_ret)
        sum_a = self.get_agent_config_class("think")
        sum_a.get_messages(create=True)
        return self.run_agent(sum_a,
                              f"Develop a concise and relevant response for the user. This response should be brief"
                              f", pertinent, and clear. Summarize the information and present the progress."
                              f" This reply will be transcribed into speech for the user."
                              f"\nInformation:{chain_sum_data}"
                              f"User Input:{user_text}\n", mode_over_lode="conversation"), chain_ret

    def chain_cor_runner(self, use, task_name, args, config, sto_name, task, steps, keys, chain_ret, sum_sto):
        ret = ''
        ret_data = []
        task_name = task_name.strip()
        self.logger.info(Style.GREY(f"using {steps} {use} {task_name} {args[:15]}..."))
        if use == "tool":
            if 'agent' in task_name.lower():
                ret = self.run_agent(config, args, mode_over_lode="tools")
            else:
                ret = self.run_tool(args, task_name, config)

        elif use == "agent":
            if config.mode == 'free':
                config.task_list.append(args)
            ret = self.run_agent(config, args, mode_over_lode=config.mode)
        elif use == 'function':
            if 'function' in keys:
                if callable(task['function']) and chain_ret:
                    task['function'](chain_ret[-1][1])

        elif use == 'expyd' or use == 'chain':
            ret, ret_data = self.execute_thought_chain(args, self.agent_chain.get(task_name.strip()), config,
                                                       speak=self.speak)
        else:
            self.print(Style.YELLOW(f"use is not available {use} avalabel ar [tool, agent, function, chain]"))

        self.logger.info(Style.GREY(f"Don : {str(ret)[:15]}..."))

        if 'short-mem' in keys:
            self.logger.warning(Style.GREY(f"In chunk {steps} no detected 'short-mem' {task['short-mem']}"))
            if task['short-mem'] == "summary":
                short_mem = config.short_mem.text
                if short_mem != sum_sto:
                    config.short_mem.clear_to_collective()
                    config.short_mem.text = self.mas_text_summaries(short_mem)
                else:
                    sum_sto = short_mem
            if task['short-mem'] == "full":
                pass
            if task['short-mem'] == "clear":
                config.short_mem.clear_to_collective()

        return ret, sum_sto, ret_data

    def chain_return(self, keys, chain_ret_, task, task_name, ret, chain_data, uesd_mem, chain_ret):

        if "return" in keys:
            if chain_ret_:
                ret = chain_ret_
            if 'text-splitter' in keys:
                mem = self.get_context_memory()
                sep = ''
                al = 'KMeans'
                if 'separators' in keys:
                    sep = task['separators']
                    if task['separators'].endswith('code'):
                        al = 'AgglomerativeClustering'
                        sep = sep.replace('code', '')
                self.print(f"task_name:{task_name} al:{al} sep:{sep}")
                ret = mem.split_text(task_name, ret, separators=sep, chunk_size=task['text-splitter'])
                mem.add_data(task_name)

                mem.crate_live_context(task_name, al)
                uesd_mem[task['return']] = task_name

            chain_data[task['return']] = ret
            chain_ret.append([task['name'], ret])

        return chain_data, chain_ret, uesd_mem

    def execute_2tree(self, user_text, tree, config: AgentConfig):
        config.binary_tree = tree
        config.stop_sequence = "\n\n\n\n"
        config.set_completion_mode('chat')
        res_um = 'Plan for The Task:'
        res = ''
        tree_depth_ = config.binary_tree.get_depth(config.binary_tree.root)
        for _ in range(tree_depth_):
            self.print(f"NEXT chain {config.binary_tree.get_depth(config.binary_tree.root)}"
                       f"\n{config.binary_tree.get_left_side(0)}")
            res = self.run_agent(config, user_text, mode_over_lode='q2tree')
            tree_depth = config.binary_tree.get_depth(config.binary_tree.root)
            don, next_on, speak = False, 0, res
            str_ints_list_to = list(range(tree_depth + 1))
            for line in res.split("\n"):
                if line.startswith("Answer"):
                    print(F"LINE:{line[:10]}")
                    for char in line[6:12]:
                        char_ = char.strip()
                        if char_ in [str(x) for x in str_ints_list_to]:
                            next_on = int(char_)
                            break

                if line.startswith("+1"):
                    print(F"detected +1")
                    line = line.replace("+1", '')
                    exit_on = -1
                    if "N/A" in line:
                        alive = False
                        res_um = "Task is not fitting isaa's capabilities"
                        break
                    for char in line[0:6]:
                        char_ = char.strip()
                        if char_ in [str(x) for x in str_ints_list_to]:
                            exit_on = int(char_)
                            break
                    if exit_on != -1:
                        next_on = exit_on

            if next_on == 0:
                if len(res) < 1000:
                    for char in res:
                        char_ = char.strip()
                        if char_ in [str(x) for x in str_ints_list_to]:
                            next_on = int(char_)
                            break

            if next_on == tree_depth:
                alive = False
                break

            elif next_on == 0:
                alive = False
                res_um = 'Task is to complicated'
                break
            else:
                new_tree = config.binary_tree.cut_tree('L' * (next_on - 1) + 'R')
                config.binary_tree = new_tree

        return res, res_um

    def stream_read_llm(self, text, config, r=2.0, line_interpret=False, interpret=lambda x: ''):

        p_token_num = config.get_tokens(text)
        config.token_left = config.max_tokens - p_token_num
        # self.print(f"TOKENS: {p_token_num}:{len(text)} | left = {config.token_left if config.token_left > 0 else '-'} |"
        #            f" max : {config.max_tokens}")
        llm_output = None

        if config.token_left < 0:
            text = self.mas_text_summaries(text)
            p_token_num = config.get_tokens(text)
            config.token_left = config.max_tokens - p_token_num
            self.print(f"TOKENS: {p_token_num} | left = {config.token_left if config.token_left > 0 else '-'}")

        if p_token_num == 0 and len(text) <= 9:
            self.print(f"No context")
            return "No context"

        if '/' in config.model_name:
            # if text:
            #     config.step_between = text
            template = config.prompt.replace('{', '}}').replace('}', '{{') + '{xVx}'
            prompt = PromptTemplate(
                input_variables=['xVx'],
                template=template,
            )
            try:
                llm_output = LLMChain(prompt=prompt, llm=self.get_llm_models(config.model_name)).run(text)
            except ValueError:
                llm_output = "ValueError: on generation"

            return self.add_price_data(prompt=template,
                                       config=config,
                                       llm_output=llm_output)

        elif config.model_name.startswith('gpt4all#'):
            if text:
                config.step_between = text

            prompt = config.prompt
            if f'LLM-model-{config.model_name}' not in self.config.keys():
                self.load_llm_models(config.model_name)

            llm_output = self.config[f'LLM-model-{config.model_name}'].generate(
                prompt=prompt,
                streaming=config.stream,

                temp=config.temperature,
                top_k=34,
                top_p=0.4,
                repeat_penalty=1.18,
                repeat_last_n=64,
                n_batch=8,
            )

            if not config.stream:
                return self.add_price_data(prompt=prompt,
                                           config=config,
                                           llm_output=llm_output)

        try:
            if not config.stream and llm_output is None:
                with Spinner(
                    f"Generating response {config.name} {config.model_name} {config.mode} {config.completion_mode}"):
                    res = self.process_completion(text, config)
                if config.completion_mode == 'chat':
                    config.add_message('assistant', res)
                self.add_price_data(prompt=config.last_prompt, config=config, llm_output=res)
                return res

            if llm_output is None:
                llm_output = self.process_completion(text, config)

            # print(f"Generating response (/) stream (\\) {config.name} {config.model_name} {config.mode} "
            #       f"{config.completion_mode}")
            min_typing_speed, max_typing_speed, res = 0.01, 0.005, ""
            try:
                line_content = ""
                results = []
                for line in llm_output:  # openai.error.InvalidRequestError if error try  is_text_completion = False
                    ai_text = ""

                    if len(line) == 0 or not line:
                        continue

                    if isinstance(line, dict):
                        data = line['choices'][0]

                        if "text" in data.keys():
                            ai_text = line['choices'][0]['text']
                        elif "content" in data['delta'].keys():
                            ai_text = line['choices'][0]['delta']['content']

                    if isinstance(line, str):
                        ai_text = line
                    line_content += ai_text
                    if line_interpret and "\n" in line_content:
                        interpret(line_content)
                        line_content = ""

                    for i, word in enumerate(ai_text):
                        if not word:
                            continue
                        print(word, end="", flush=True)
                        if self.print_stream != print:
                            self.print_stream({'isaa-text': word})
                        typing_speed = random.uniform(min_typing_speed, max_typing_speed)
                        time.sleep(typing_speed)
                        # type faster after each word
                        min_typing_speed = min_typing_speed * 0.04
                        max_typing_speed = max_typing_speed * 0.03
                    res += str(ai_text)
                if line_interpret and line_content:
                    interpret(line_content)
            except requests.exceptions.ChunkedEncodingError as ex:
                print(f"Invalid chunk encoding {str(ex)}")
                self.print(f"{' ' * 30} | Retry level: {r} ", end="\r")
                with Spinner("ChunkedEncodingError", symbols='c'):
                    time.sleep(2 * (3 - r))
                if r > 0:
                    print('\n\n')
                    return self.stream_read_llm(text + '\n' + res, config, r - 1)
            if config.completion_mode == 'chat':
                config.add_message('assistant', res)
            self.add_price_data(prompt=config.last_prompt, config=config, llm_output=res)
            return res
        except openai.error.RateLimitError:
            self.print(f"{' ' * 30}  | Retry level: {r} ", end="\r")
            if r > 0:
                self.logger.info(f"Waiting {5 * (8 - r)} seconds")
                with Spinner("Waiting RateLimitError", symbols='+'):
                    time.sleep(5 * (8 - r))
                self.print(f"\n Retrying {r} ", end="\r")
                return self.stream_read_llm(text, config, r - 1)
            else:
                self.logger.error("The server is currently overloaded with other requests. Sorry about that!")
                return "The server is currently overloaded with other requests. Sorry about that! ist als possible that" \
                       " we hit the billing limit consider updating it."

        except openai.error.InvalidRequestError:
            self.print(f"{' ' * 30} | Retry level: {r} ", end="\r")
            with Spinner("Waiting InvalidRequestError", symbols='b'):
                time.sleep(1.5)
            if r > 1.25:
                config.short_mem.max_length -= 300
                if config.short_mem.max_length < 0:
                    config.short_mem.max_length = 2000
                config.edit_text.max_length -= 300
                if config.edit_text.max_length < 0:
                    config.edit_text.max_length = 2000
                config.observe_mem.max_length -= 300
                if config.observe_mem.max_length < 0:
                    config.observe_mem.max_length = 1000
                config.short_mem.cut()
                config.edit_text.cut()
                config.observe_mem.cut()
                return self.stream_read_llm(text, config, r - 0.25)
            elif r > 1:
                config.shorten_prompt()
                return self.stream_read_llm(text, config, r - 0.25)
            elif r > .75:
                config.get_messages(create=True)
                return self.stream_read_llm(self.mas_text_summaries(text), config, r - 0.25)
            elif r > 0.5:
                config.stream = False
                res = self.stream_read_llm(self.mas_text_summaries(text), config, r - 0.25)
                config.stream = True
                return res
            elif r > 0.25:
                config.short_mem.clear_to_collective()
                config.edit_text.clear_to_collective()
                config.observe_mem.cut()
                return self.stream_read_llm(text, config, r - 0.25)
            else:
                self.logger.error("The server is currently overloaded with other requests. Sorry about that!")
                return "The System cannot correct the text input for the agent."

        # except Exception as e:
        #    self.logger.error(str(e))
        #    return "*Error*"

    @staticmethod
    def process_completion(text, config: AgentConfig):

        if not isinstance(config, AgentConfig):
            raise TypeError("Invalid config")

        if len(config.task_list) == 0 and len(text) != 0:
            config.step_between = text

        model_name = config.model_name
        ret = ""
        if config.stream:
            ret = {'choices': [{'text': "", 'delta': {'content': ''}}]}

        if '/' in model_name:
            return "only supported for open ai."

        if config.completion_mode == 'text':

            if model_name.startswith('gpt-'):
                model_name = "text-davinci-003"

            ret = openai.Completion.create(
                model=model_name,
                prompt=config.prompt,
                # max_tokens=int(config.token_left * 0.9),
                temperature=config.temperature,
                n=1,
                stream=config.stream,
                stop=config.stop_sequence,
            )

            if not config.stream:
                ret = ret.choices[0].text

        elif config.completion_mode == 'chat':
            messages = config.get_messages(create=False)
            if not messages:
                messages = config.get_messages(create=True)
            if text:
                config.add_message("user", text)
            # print(f"Chat Info\n{model_name=}\n{messages=}\n{config.token_left=}\n{config.temperature=}\n
            # {config.stream=}\n{config.stop_sequence=}\n")
            ret = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                # max_tokens=config.token_left,
                temperature=config.temperature,
                n=1,
                stream=config.stream,
                stop=config.stop_sequence,
            )
            if not config.stream:
                ret = ret.choices[0].message.content

        elif config.completion_mode == 'edit':
            if text:
                config.edit_text.text = text
            else:
                config.edit_text.text = config.short_mem.text

            ret = openai.Edit.create(
                model=model_name,
                input=config.edit_text.text,
                instruction=text,
            )
            ret = ret.choices[0].text
        else:
            raise ValueError(f"Invalid mode : {config.completion_mode} valid ar 'text' 'chat' 'edit'")

        return ret

    def mas_text_summaries(self, text, min_length=1600):

        len_text = len(text)
        if len_text < min_length:
            return text

        if text in self.mas_text_summaries_dict[0]:
            self.print("summ return vom chash")
            return self.mas_text_summaries_dict[1][self.mas_text_summaries_dict[0].index(text)]

        cap = 800
        max_length = 45
        summary_chucks = ""

        if 'text-splitter0-init' not in self.config.keys():
            self.config['text-splitter0-init'] = False
        if not self.config['text-splitter0-init'] or not isinstance(self.config['text-splitter0-init'],
                                                                    CharacterTextSplitter):
            self.config['text-splitter0-init'] = CharacterTextSplitter(chunk_size=cap, chunk_overlap=cap / 6)

        splitter = self.config['text-splitter0-init']

        if len(text) >= 6200:
            cap = 1200
            max_length = 80
            if 'text-splitter1-init' not in self.config.keys():
                self.config['text-splitter1-init'] = False
            if not self.config['text-splitter1-init'] or not isinstance(self.config['text-splitter1-init'],
                                                                        CharacterTextSplitter):
                self.config['text-splitter1-init'] = CharacterTextSplitter(chunk_size=cap, chunk_overlap=cap / 6)

            splitter = self.config['text-splitter1-init']

        if len(text) >= 10200:
            cap = 1800
            max_length = 160
            if 'text-splitter2-init' not in self.config.keys():
                self.config['text-splitter2-init'] = False
            if not self.config['text-splitter2-init'] or not isinstance(self.config['text-splitter2-init'],
                                                                        CharacterTextSplitter):
                self.config['text-splitter2-init'] = CharacterTextSplitter(chunk_size=cap, chunk_overlap=cap / 6)

            splitter = self.config['text-splitter2-init']

        if len(text) >= 70200:
            cap = 1900
            max_length = 412
            if 'text-splitter3-init' not in self.config.keys():
                self.config['text-splitter3-init'] = False
            if not self.config['text-splitter3-init'] or not isinstance(self.config['text-splitter3-init'],
                                                                        CharacterTextSplitter):
                self.config['text-splitter3-init'] = CharacterTextSplitter(chunk_size=cap, chunk_overlap=cap / 6)

            splitter = self.config['text-splitter3-init']

        summarization_mode_sto = 0
        if len(text) > self.summarization_limiter and self.summarization_mode:
            self.summarization_mode, summarization_mode_sto = 0, self.summarization_mode

        def summary_func(x):
            return self.summarization(x, max_length=max_length)

        def summary_func2(x, babage=None):
            agent = self.get_agent_config_class('summary')
            if babage:
                agent.set_model_name("text-babbage-001")
            else:
                agent.set_model_name("text-curie-001")
            if isinstance(x, list):
                end = []
                for i in x:
                    text_sum = self.stream_read_llm(i + "\nSummary :", agent, r=0)
                    end.append({'summary_text': text_sum})
            else:
                text_sum = self.stream_read_llm(x + "\nSummary :", agent, r=0)
                end = [{'summary_text': text_sum}]
            return end

        def summary_func3(x):
            if isinstance(x, list):
                end = []
                for i in x:
                    end.append({'summary_text': self.stream_read_llm(i + "\nTask: Write a Summary.",
                                                                     self.get_agent_config_class('thinkm'), r=0)})
            else:
                end = [{'summary_text': self.stream_read_llm(x + "\nTask: Write a Summary.",
                                                             self.get_agent_config_class('thinkm'), r=0)}]
            return end

        # while len(text) > cap:
        #     chucks.append(text[:cap])
        #     text = text[cap:]
        # if text:
        #     chucks.append(text)

        chunks: list[str] = splitter.split_text(text)
        i = 0
        max_iter = int(len(chunks) * 1.2)
        while i < len(chunks) and max_iter > 0:
            max_iter -= 1
            chunk = chunks[i]
            if len(chunk) > cap * 1.5:
                chunks = chunks[:i] + [chunk[:len(chunk) // 2], chunk[len(chunk) // 2:]] + chunks[i + 1:]
            else:
                i += 1

        self.print(f"SYSTEM: chucks to summary: {len(chunks)} cap : {cap}")
        with Spinner("Generating summary", symbols='d'):
            if self.summarization_mode == 0:
                summaries = summary_func(chunks)
            elif self.summarization_mode == 1:
                summaries = summary_func2(chunks, 'babbage')
            elif self.summarization_mode == 2:
                summaries = summary_func2(chunks)
            elif self.summarization_mode == 3:
                summaries = summary_func3(chunks)
            else:
                summaries = summary_func(chunks)

        for i, chuck_summary in enumerate(summaries):
            summary_chucks += chuck_summary['summary_text'] + "\n"

        self.print(f"SYSTEM: all summary_chucks : {len(summary_chucks)}")

        if len(summaries) > 8:
            if len(summary_chucks) < 20000:
                summary = summary_chucks
            elif len(summary_chucks) > 20000:
                if self.summarization_mode == 0:
                    summary = summary_func2(summary_chucks)[0]['summary_text']
                else:
                    summary = summary_func3(summary_chucks)[0]['summary_text']
            else:
                summary = self.mas_text_summaries(summary_chucks)
        else:
            summary = summary_chucks

        self.print(
            f"SYSTEM: final summary from {len_text}:{len(summaries)} ->"
            f" {len(summary)} compressed {len_text / len(summary):.2f}X\n")

        if summarization_mode_sto:
            self.summarization_mode = summarization_mode_sto

        self.mas_text_summaries_dict[0].append(text)
        self.mas_text_summaries_dict[1].append(summary)

        return summary

    def summarize_dict(self, input_dict, config: AgentConfig):
        input_str = input_dict['input']
        output_str = input_dict['output']
        intermediate_steps = input_dict['intermediate_steps']
        chucs = []
        i = 0
        for step in intermediate_steps:
            if isinstance(step, tuple):
                step_content = f"\naction {i}" + str(step[0].tool)
                step_content += f"\ntool_input {i} " + str(step[0].tool_input)
                step_content += f"\nlog {i} " + str(step[1])
                chucs.append(self.mas_text_summaries(step_content))
            i += 1

        if chucs:
            config.observe_mem.text = '\n'.join(chucs)
        return output_str

    def summarize_ret_list(self, ret_list):
        chucs = []
        print("ret_list:", ret_list)
        for i, step in enumerate(ret_list):
            print("i, step:", i, step)
            if isinstance(step, list):
                step_content = ""
                if len(step) == 2:
                    if isinstance(step[1], str):
                        step_content += f"\nlog {i}  input : {str(step[0])} output :  {str(step[1])}"
                    if isinstance(step[1], list):
                        step_content += f"\nlog {i}  input : {str(step[0][0])} output :  {str(step[0][1])}"
                    if isinstance(step[1], dict):
                        if 'input' in step[1].keys():
                            step_content += f"\ninput {i} " + str(step[1]['input'])
                        if 'output' in step[1].keys():
                            step_content += f"\noutput {i} " + str(step[1]['output'])
                if len(step) > 1600:
                    step_content = self.mas_text_summaries(step_content)
                chucs.append(step_content)
        text = 'NoContent'
        if chucs:
            text = '\n'.join(chucs)
        return text

    def init_db_questions(self, db_name, config: AgentConfig):
        retriever = self.get_context_memory().get_retriever(db_name)
        if retriever is not None:
            retriever.search_kwargs['distance_metric'] = 'cos'
            retriever.search_kwargs['fetch_k'] = 20
            retriever.search_kwargs['maximal_marginal_relevance'] = True
            retriever.search_kwargs['k'] = 20
            return ConversationalRetrievalChain.from_llm(self.get_llm_models(config.model_name), retriever=retriever)
        return None

    def get_chain(self, hydrate=None, f_hydrate=None) -> AgentChain:
        logger = get_logger()
        logger.info(Style.GREYBG(f"AgentChain requested"))
        agent_chain = self.agent_chain
        if hydrate is not None or f_hydrate is not None:
            self.agent_chain.add_hydrate(hydrate, f_hydrate)
        logger.info(Style.Bold(f"AgentChain instance, returned"))
        return agent_chain

    def get_context_memory(self) -> AIContextMemory:
        logger = get_logger()
        logger.info(Style.GREYBG(f"AIContextMemory requested"))
        cm = self.agent_memory
        logger.info(Style.Bold(f"AIContextMemory instance, returned"))
        return cm

    def add_price_data(self, prompt: str, llm_output: str, config: AgentConfig):
        input_price, output_price = config.calc_price(prompt, llm_output)
        self.price['input'] += input_price
        self.price['output'] += output_price
        self.price['all'] += output_price + input_price
        self.price['price_consumption'] += config.consumption
        self.price['consumption'].append([config.name, config.consumption])
        if config.model_name not in self.price['model_consumption'].keys():
            self.price['model_consumption'][config.model_name] = {
                'i': 0,
                'o': 1,
                'c': 0,
            }
        self.price['model_consumption'][config.model_name]['i'] += input_price
        self.price['model_consumption'][config.model_name]['o'] += output_price
        self.price['model_consumption'][config.model_name]['c'] += 1
        config.consumption = 0
        return llm_output

    def clear_price(self):
        self.price = {
            'all': 0,
            'input': 0,
            'output': 1,
            'consumption': [],
            'price_consumption': 0,
            'model_consumption': {},
            'history': {},
        }

    def show_usage(self, print=print):

        self.print("Consumption Overview\n")
        doller_euro = 0.91
        prosessing_factor = 0.2

        def show_eruo_doller(price, t='€'):
            return f"{(price / 100) * doller_euro:.4f}€" if t == '€' else f"{(price / 100):.4f}$"

        print(f"{Style.WHITE(Style.Bold(' === Usage === '))}\n")
        print(f"{Style.WHITE(' --- Agents --- ')}\n")

        agents = {}

        for agent_usage in self.price['consumption']:
            if not agent_usage[0] in agents.keys():
                agents[agent_usage[0]] = []
            agents[agent_usage[0]].append(agent_usage[1])

        if self.price['price_consumption'] == 0:
            self.price['price_consumption'] = 1
        if self.price['output'] == 0:
            self.price['output'] = 1

        for name, agent_usage in agents.items():
            print(
                f"\t- {show_eruo_doller(sum(agent_usage) * prosessing_factor)} by {name}")

        if agents == {}:
            print("\tNo Agents were used")
        print(f"\n{Style.WHITE(' --- Models --- ')}\n")

        for name, value in self.price['model_consumption'].items():
            input_con = value['i']
            out_con = value['o']
            sum_con = input_con + out_con
            print(
                f"\t- {show_eruo_doller(sum_con * prosessing_factor)} by {name}  calld {value['c']}X")
        if self.price['model_consumption'] == {}:
            print("\tNo Models were used")
        print(f"\n{Style.WHITE(f' --- History : {self.app_.id} --- ')}\n")

        for name, value in self.price['history'].items():
            input_con = value['i']
            out_con = value['o']
            sum_con = input_con + out_con
            print(
                f"\t- {show_eruo_doller(sum_con)} intern {show_eruo_doller(value['c'] * prosessing_factor)} at {name}")
        if self.price['history'] == {}:
            print("\tNo history found")

        print(f"\n{Style.WHITE(' --- Summary --- ')}\n")
        print(
            f"\t{Style.BLUE(Style.Bold('I/O:'))} {show_eruo_doller(self.price['input'])} / {show_eruo_doller(self.price['output'])} sum : {show_eruo_doller(self.price['all'])}")
        print(Style.GREY(f"\t- i/o balance {self.price['input'] // self.price['output']} times mor input then Output"))
        print(Style.ITALIC(Style.GREEN(
            f"\tInternal consumption: {show_eruo_doller(self.price['price_consumption'] * prosessing_factor)}")))

        print(Style.ITALIC(Style.Bold(Style.CYAN("\tCold Price : ") + show_eruo_doller(self.price['all']))))
        print(Style.ITALIC(Style.Bold(Style.MAGENTA("\tHot  Price : ") + show_eruo_doller(
            self.price['all'] + (self.price['price_consumption'] * prosessing_factor)))))

    def init_cli(self, command, app):
        app.SUPER_SET += list(self.agent_chain.chains.keys())
        self.load_keys_from_env()
        if "augment" in self.config.keys():
            self.init_from_augment(self.config['augment'],
                                   exclude=['task_list', 'task_list_done', 'step_between', 'pre_task', 'task_index'])
            self.print("Initialized from config augment")
            if 'tools' in self.config['augment'].keys():
                if self.config['augment']['tools']:
                    return
        self.init_from_augment({'tools':
                                    {'lagChinTools': ['python_repl', 'requests_all', 'terminal', 'sleep',
                                                      'google-search',
                                                      'ddg-search', 'wikipedia', 'llm-math', 'requests_get',
                                                      'requests_post',
                                                      'requests_patch', 'requests_put', 'requests_delete', 'human'],
                                     'huggingTools': [],
                                     'Plugins': [], 'Custom': []}}, 'tools')
        self.init_from_augment({'tools':
                                    {'lagChinTools': ['python_repl', 'requests_all', 'terminal', 'sleep',
                                                      'google-search',
                                                      'ddg-search', 'wikipedia', 'llm-math', 'requests_get',
                                                      'requests_post',
                                                      'requests_patch', 'requests_put', 'requests_delete', 'human'],
                                     'huggingTools': [],
                                     'Plugins': [], 'Custom': []}}, 'self')
        self.init_from_augment({'tools':
                                    {'lagChinTools': ['python_repl', 'requests_all', 'terminal', 'sleep',
                                                      'google-search',
                                                      'ddg-search', 'wikipedia', 'llm-math', 'requests_get',
                                                      'requests_post',
                                                      'requests_patch', 'requests_put', 'requests_delete', 'human'],
                                     'huggingTools': [],
                                     'Plugins': [], 'Custom': []}}, 'liveInterpretation')

    def create_task_cli(self):
        self.print("Enter your text (empty line to finish):")
        task = get_multiline_input()
        self.create_task(task)

    def optimise_task_cli(self):
        all_chains = list(self.agent_chain.chains.keys())
        chain_name = choiceList(all_chains, self.print)
        new_chain = self.optimise_task(chain_name)
        self.print(new_chain)

    def describe_chain(self, name):
        run_chain = self.agent_chain.get(name)
        if not len(run_chain):
            return "invalid Chain Namen"

        task = (f"Bitte analysieren und interpretieren Sie das gegebene JSON-Objekt, das eine Aufgabenkette "
                "repräsentiert. Identifizieren Sie das übergeordnete Ziel, die Anwendungsfälle und die Strategie, "
                "die durch diese Aufgabenkette dargestellt werden. Stellen Sie sicher,"
                " dass Ihre Analyse detailliert und präzise ist. Ihre Antwort sollte klar und präzise sein,"
                " um ein vollständiges Verständnis der Aufgabenkette und ihrer "
                "möglichen Einschränkungen zu ermöglichen. Deine Antwort soll kurtz und pregnant sein Maximal 2 sätze"
                f"zu analysierende Aufgabenkette: {run_chain}")

        discription = self.stream_read_llm(task, self.get_agent_config_class("think"))

        if len(discription) > 1000:
            discription = self.mas_text_summaries(discription, min_length=1000)

        self.print(f"Infos : {discription}")
        self.agent_chain.add_discr(name, discription)
        return discription

    def save_to_mem(self):
        for name in self.config['agents-name-list']:
            self.get_agent_config_class(agent_name=name).save_to_permanent_mem()

    def describe_all_chains(self):

        for chain_name in self.agent_chain.chains.keys():
            if self.agent_chain.get_discr(chain_name):
                continue
            self.describe_chain(chain_name)

    def run_describe_chains(self, command):

        if len(command) == 2:
            self.describe_chain(command[1])

        else:
            self.describe_all_chains()

    def get_best_fitting(self, subject):

        all_description = ""

        for key in self.agent_chain.chains.keys():
            if "Task Generator" in key or "Task-Generator" in key:
                continue
            des = self.agent_chain.get_discr(key)
            if des is None:
                des = key
            all_description += f"NAME:{key} \nUse case:{des}"

        mini_task0 = f"""Bitte durchsuchen Sie eine Liste von Aufgabenketten oder Lösungsansätzen und identifizieren
        Sie die beste Option für ein bestimmtes Thema oder Problem. Berücksichtigen Sie dabei die spezifischen
        Anforderungen und Ziele des Themas. Ihre Analyse sollte gründlich und detailliert sein, um die Stärken und
        Schwächen jeder Option zu beleuchten und zu begründen, warum die von Ihnen gewählte Option die beste ist.
        Stellen Sie sicher, dass Ihre Antwort klar, präzise und gut begründet ist.
        geben sie den Namen des Lösungsansatz mit an!
        Problem :
        {subject}
        Lösungsansätze:
        {all_description}
        """
        mini_task0_res = self.stream_read_llm(mini_task0, self.get_agent_config_class("think"))

        mini_task1 = f""" "{mini_task0_res}"\n welcher Lösungsansatz wurde ausgewählt von diesen ausgewählt {list(self.agent_chain.chains.keys())}
        gebe nur den namen zurück
        wenn keiner der ansätze passt das gebe None zurück.
        name:"""

        mini_task1_res = self.mini_task_completion(mini_task1)

        chain_name = mini_task1_res
        for task_name in list(self.agent_chain.chains.keys()):
            if mini_task1_res.lower() in task_name.lower():
                chain_name = task_name

        self.print(f"Das system schlägt {chain_name} vor")
        self.print(f"mit der beründung : {mini_task0_res}")

        return chain_name, mini_task0_res

    def run_create_task_cli(self):
        self.print("Enter your text (empty line to finish):")
        task = get_multiline_input()
        name = self.create_task(task=task)
        self.print(f"New Task Crated name : {name}")

    def remove_chain_cli(self):
        all_chains = list(self.agent_chain.chains.keys())
        if not all_chains:
            return "No Cains Installed or loaded"

        chain_name = choiceList(all_chains, self.print)
        self.agent_chain.remove(chain_name)

    def run_chain_cli(self):
        all_chains = list(self.agent_chain.chains.keys())
        if not all_chains:
            return "No Cains Installed or loaded"

        chain_name = choiceList(all_chains, self.print)
        self.print("Enter your text (empty line to finish):")
        task = get_multiline_input()
        self.print(f"Starting Chin : {chain_name}")
        run_chain = self.agent_chain.get(chain_name)
        self.print(f"Chin len : {len(run_chain)}")
        if run_chain:
            res = self.execute_thought_chain(task, run_chain, self.get_agent_config_class("self"))
            self.print(f"Chain return \n{self.app.pretty_print(list(res))}")
        else:
            res = ["No chain found"]

        return res

    def run_auto_chain_cli(self):
        all_chains = list(self.agent_chain.chains.keys())
        if not all_chains:
            return "No Cains Installed or loaded"

        self.print("Enter your text (empty line to finish):")
        task = get_multiline_input()

        return self.run_auto_chain(task)

    def run_auto_chain(self, task):

        chain_name, begründung = self.get_best_fitting(task)

        if "y" not in input("Validate (y/n)"):
            return "Presses Stopped"

        self.print(f"Starting Chin : {chain_name}")
        return self.run_chain_on_name(chain_name, task)

    def run_chain_on_name(self, name, task):
        run_chain = self.agent_chain.get(name)
        self.print(f"Chin len : {len(run_chain)}")

        if run_chain:
            res = self.execute_thought_chain(task, run_chain, self.get_agent_config_class("self"))
            self.print(f"Chain return \n{self.app.pretty_print(list(res))}")
        else:
            res = "No chain found", []

        return res


def get_multiline_input():
    lines = []

    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    return "\n".join(lines)


def choiceList(all_chains, print_=print, input_=input, do_INQUIRER=True):
    if INQUIRER and do_INQUIRER:

        questions = [
            inquirer.List('chain',
                          message="Choose a chain?",
                          choices=all_chains,
                          ),
        ]
        choice = inquirer.prompt(questions)['chain']

    else:
        choice = input_(f"{all_chains} select one (q) to quit:")
        while choice not in all_chains:
            if choice.lower() == 'q':
                return "Quiet"
            print_("Invalid Chain name")
            choice = input_(f"{all_chains} select one (q) to quit:")
    return choice


def get_tool(app: App):
    if not app:
        return Tools(App('isaa'))

    if app.AC_MOD:
        if app.AC_MOD.name == 'isaa':
            return app.AC_MOD

    app.logger.error('Unknown - app isaa module is not the active mod')
    if isinstance(app.new_ac_mod('isaa'), bool):
        return app.AC_MOD

    app.logger.error('activation failed try loading module')
    if app.save_load('isaa'):
        app.new_ac_mod('isaa')
        return app.AC_MOD
    app.logger.critical('cant load isaa module')

    return Tools()


def initialize_gi(app: App or Tools, model_name):
    app.logger.info(f'initializing gi {model_name}')
    if isinstance(app, App):
        mod = get_tool(app)
    elif isinstance(app, Tools):
        mod = app
    else:
        raise ValueError(f"Unknown app or mod type {type(app)}")
    mod.config['genrate_image-in'] = model_name

    if not mod.config['genrate_image-init']:
        if 'REPLICATE_API_TOKEN' not in mod.config.keys():
            raise ValueError("No REPLICATE_API_TOKEN Specified pleas set the key in the config")
        mod.config[f'replicate'] = replicate.Client(api_token=mod.config[f'REPLICATE_API_TOKEN'])

    mod.config['genrate_image-init'] = True

    # mod.config[f'genrate_image{model_name}'] = StableDiffusionPipeline.from_pretrained(model_name, revision="fp16",
    #                                                                                    torch_dtype=torch.float16)
    # mod.config[f'genrate_image{model_name}'].scheduler = DPMSolverMultistepScheduler.from_config(
    #     mod.config[f'genrate_image{model_name}'].scheduler.config)

    model = mod.config[f'replicate'].models.get(model_name)
    mod.config[f'genrate_image-{model_name}'] = model.versions.get(
        "db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf")
    print(f"Initializing Model : {model_name}")


def genrate_image(inputs, app: App, model="stability-ai/stable-diffusion"):
    mod = get_tool(app)
    if 'genrate_image-init' not in mod.config.keys():
        mod.config['genrate_image-init'] = False
    if not mod.config['genrate_image-init']:
        initialize_gi(mod, model)
    if 'genrate_image-in' not in mod.config.keys():
        mod.config['genrate_image-in'] = model

    if mod.config['genrate_image-in'] != model:
        initialize_gi(mod, model)

    if f'genrate_image-{model}' not in mod.config.keys():
        initialize_gi(mod, model)

    return mod.config[f'genrate_image-{model}'].predict(**inputs)  # (text).images


from toolboxv2.mods import BROWSER


def show_image_in_internet(images_url, browser=BROWSER):
    if isinstance(images_url, str):
        images_url = [images_url]
    for image_url in images_url:
        os.system(f'start {browser} {image_url}')


def image_genrating_tool(prompt, app):
    app.logger.info("Extracting data from prompt")
    app.logger.info("Splitting data")
    if '|' in prompt:
        prompt = prompt.split('|')[1]
    if isinstance(prompt, str):
        inputs = {
            # Input prompt
            'prompt': prompt,

            # pixel dimensions of output image
            'image_dimensions': "512x512",

            # Specify things to not see in the output
            # 'negative_prompt': ...,

            # Number of images to output.
            # Range: 1 to 4
            'num_outputs': 1,

            # Number of denoising steps
            # Range: 1 to 500
            'num_inference_steps': 50,

            # Scale for classifier-free guidance
            # Range: 1 to 20
            'guidance_scale': 7.5,

            # Choose a scheduler.
            'scheduler': "DPMSolverMultistep",

        }
    else:
        inputs = prompt

    print(f"Generating Image")
    images = genrate_image(inputs, app)

    print(f"Showing Images")

    show_image_in_internet(images)


# @ gitHub Auto GPT
def browse_website(url, question, summ):
    summary = get_text_summary(url, question, summ)
    links = get_hyperlinks(url)

    # Limit links to 5
    if len(links) > 5:
        links = links[:5]

    result = f"""Website Content Summary: {summary}\n\nLinks: {links}"""

    return result


def get_text_summary(url, question, summarize):
    text = scrape_text(url)
    print(text)
    summary = summarize(f"Context ###{text}### Question ###{question}###")
    if isinstance(summary, list):
        summary = '\n'.join(summary)

    return """Result: """ + summary


def get_hyperlinks(url):
    link_list = scrape_links(url)
    return link_list


def scrape_text(url):
    response = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"})

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        return "Error: HTTP " + str(response.status_code) + " error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def extract_hyperlinks(soup):
    hyperlinks = []
    for link in soup.find_all('a', href=True):
        hyperlinks.append((link.text, link['href']))
    return hyperlinks


def format_hyperlinks(hyperlinks):
    formatted_links = []
    for link_text, link_url in hyperlinks:
        formatted_links.append(f"{link_text} ({link_url})")
    return formatted_links


def scrape_links(url):
    response = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"})

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        return "error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup)

    return format_hyperlinks(hyperlinks)


def _extract_from_json(agent_text, config):
    try:
        json_obj = anything_from_str_to_dict(agent_text, {"Action": None, "Inputs": None})
        # print("OBJ:::::::::", json_obj)
        if json_obj:
            json_obj = json_obj[0]
            if not isinstance(json_obj, dict):
                return None, ''
            if "Action" in json_obj.keys() and "Inputs" in json_obj.keys():
                action = json_obj["Action"]
                inputs = json_obj["Inputs"]
                if action in config.tools.keys():
                    return action, inputs
    except json.JSONDecodeError:
        pass
    return None, ''


def _extract_from_string(agent_text, config):
    action_match = re.search(r"Action:\s*(\w+)", agent_text)
    inputs_match = re.search(r"Inputs:\s*({.*})", agent_text)
    inputs_matchs = re.search(r"Inputs:\s*(.*)", agent_text)
    if action_match is not None and inputs_match is not None:
        action = action_match.group(1)
        inputs = inputs_match.group(1)
        if action in config.tools.keys():
            return action.strip(), inputs

    if action_match is not None and inputs_matchs is not None:
        action = action_match.group(1)
        inputs = inputs_matchs.group(1)
        print(f"action: {action=}\n{action in config.tools.keys()=}\n {config.tools.keys()=}")
        if action in config.tools.keys():
            return action.strip(), inputs

    if action_match is not None:
        action = action_match.group(1)
        if action in config.tools.keys():
            return action.strip(), ''

    return None, ''


def _extract_from_string_de(agent_text, config):
    action_match = re.search(r"Aktion:\s*(\w+)", agent_text)
    inputs_match = re.search(r"Eingaben:\s*({.*})", agent_text)
    inputs_matchs = re.search(r"Eingaben:\s*(.*)", agent_text)

    if action_match is not None and inputs_match is not None:
        action = action_match.group(1)
        inputs = inputs_match.group(1)
        if action in config.tools.keys():
            return action.strip(), inputs

    if action_match is not None and inputs_matchs is not None:
        action = action_match.group(1)
        inputs = inputs_matchs.group(1)
        print(f"action: {action=}\n{action in config.tools.keys()=}\n {config.tools.keys()=}")
        if action in config.tools.keys():
            return action.strip(), inputs

    if action_match is not None:
        action = action_match.group(1)
        if action in config.tools.keys():
            return action.strip(), ''

    return None, ''


# print(get_tool(get_app('debug')).get_context_memory().get_context_for("Hallo das ist ein Test")) Fridrich


def add_shell_tool(isaa: Tools, config: AgentConfig):
    def run_command(command: str) -> str:
        """
        Runs a command in the user's shell.
        It is aware of the current user's $SHELL.
        :param command: A shell command to run.
        :return: A JSON string with information about the command execution.
        """
        if platform.system() == "Windows":
            is_powershell = len(os.getenv("PSModulePath", "").split(os.pathsep)) >= 3
            full_command = (
                f'powershell.exe -Command "{command}"'
                if is_powershell
                else f'cmd.exe /c "{command}"'
            )
        else:
            shell = os.environ.get("SHELL", "/bin/sh")
            full_command = f"{shell} -c {shlex.quote(command)}"

        try:
            output = subprocess.check_output(full_command, shell=True, stderr=subprocess.STDOUT)
            return json.dumps({"success": True, "output": output.decode()})
        except subprocess.CalledProcessError as e:
            return json.dumps({"success": False, "error": str(e), "output": e.output.decode()})

    isaa.add_tool("shell", run_command, "run shell command only input command",
                  "run_command(command: str) -> str json dict", config)
