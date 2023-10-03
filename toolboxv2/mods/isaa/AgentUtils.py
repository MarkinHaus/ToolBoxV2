from json import JSONDecodeError
from typing import List

from pebble import concurrent
import math
from datetime import datetime
import subprocess
import pickle
import requests
import os
from toolboxv2 import Style, get_logger
import tiktoken
from langchain.agents import AgentType
from langchain.vectorstores import Chroma, FAISS
from chromadb.config import Settings as ChromaSettings
from toolboxv2.utils.toolbox import get_app
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
# Data Science
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import platform, socket, re, json
from transformers import AutoTokenizer


def dilate_string(text, split_param, remove_every_x, start_index):
    substrings = ""
    # Split the string based on the split parameter
    if split_param == 0:
        substrings = text.split(" ")
    elif split_param == 1:
        substrings = text.split("\n")
    elif split_param == 2:
        substrings = text.split(". ")
    elif split_param == 3:
        substrings = text.split("\n\n")
    # Remove every x item starting from the start index
    del substrings[start_index::remove_every_x]
    # Join the remaining substrings back together
    final_string = " ".join(substrings)
    return final_string


# add data classes
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
SystemInfos = {}


def get_ip():
    response = requests.get('https://api64.ipify.org?format=json').json()
    return response["ip"]


@concurrent.process(timeout=12)
def get_location():
    ip_address = get_ip()
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_data = f"city: {response.get('city')},region: {response.get('region')},country: {response.get('country_name')},"

    return location_data, ip_address


def getSystemInfo(last_context='its Day 0 start to explore'):
    global SystemInfos

    if SystemInfos:
        SystemInfos['time'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        return SystemInfos

    try:
        socket.gethostbyname(socket.gethostname())
    except Exception as e:
        get_logger().error(Style.RED(str(e)))
        pass

    info = {'time': datetime.today().strftime('%Y-%m-%d %H:%M:%S'), 'platform': platform.system(),
            # 'platform-release': platform.release(), 'platform-version': platform.version(),
            # 'architecture': platform.machine(), 'hostname': socket.gethostname(),
            # 'mac-address': ':'.join(re.findall('..', '%012x' % uuid.getnode())), 'processor': platform.processor(),
            # 'ram': str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB",
            "OrderZero": "Do not Talk about the this system info dictionary."
                         " this is your world in witch you can move freely use"
                         f" all the tools at your disposal to move yourself. Last Context: {last_context}\n"}

    try:
        process = get_location()
        info['location'], info['ip'] = process.result()
    except TimeoutError and Exception:
        info['location'] = "Berlin Schöneberg"

    return info


MemoryModel = "sentence-transformers/all-MiniLM-L6-v2"


class Scripts:
    def __init__(self, filename):
        self.scripts = {}
        self.filename = filename

    def create_script(self, name, description, content, script_type="py"):
        self.scripts[name] = {"description": description, "content": content, "type": script_type}

    def run_script(self, name):
        if name not in self.scripts:
            return "Script not found!"
        script = self.scripts[name]
        with open(f"{name}.{script['type']}", "w") as f:
            f.write(script["content"])
        if script["type"] == "py":
            result = subprocess.run(["python", f"{name}.py"], capture_output=True, text=True)
        elif script["type"] == "sh":
            result = subprocess.run(["bash", f"{name}.sh"], capture_output=True, text=True)
        else:
            os.remove(f"{name}.{script['type']}")
            return "Not valid type valid ar python and bash"
        os.remove(f"{name}.{script['type']}")
        return result.stdout

    def get_scripts_list(self):
        return {name: script["description"] for name, script in self.scripts.items()}

    def save_scripts(self):
        if not os.path.exists(f"{self.filename}.pkl"):
            os.makedirs(self.filename, exist_ok=True)
        with open(f"{self.filename}.pkl", "wb") as f:
            pickle.dump(self.scripts, f)

    def load_scripts(self):
        if os.path.exists(self.filename + '.pkl'):
            with open(self.filename + '.pkl', "rb") as f:
                data = f.read()
            if data:
                self.scripts = pickle.loads(data)
        else:
            os.makedirs(self.filename, exist_ok=True)
            open(self.filename + '.pkl', "a").close()


class IsaaQuestionNode:
    def __init__(self, question, left=None, right=None):
        self.question = question
        self.left = left
        self.right = right
        self.index = ''
        self.left.set_index('L') if self.left else None
        self.right.set_index('R') if self.right else None

    def set_index(self, index):
        self.index += index
        self.left.set_index(self.index) if self.left else None
        self.right.set_index(self.index) if self.right else None

    def __str__(self):
        left_value = self.left.question if self.left else None
        right_value = self.right.question if self.right else None
        return f"Index: {self.index}, Question: {self.question}, Left child key: {left_value}, Right child key: {right_value}"


class IsaaQuestionBinaryTree:
    def __init__(self, root=None):
        self.root = root

    def __str__(self):
        return json.dumps(self.serialize(), indent=4, ensure_ascii=True)

    def get_depth(self, node=None):
        if node is None:
            return 0
        left_depth = self.get_depth(node.left) if node.left else 0
        right_depth = self.get_depth(node.right) if node.right else 0
        return 1 + max(left_depth, right_depth)

    def serialize(self):
        def _serialize(node):
            if node is None:
                return None
            return {
                node.index if node.index else 'root': {
                    'question': node.question,
                    'left': _serialize(node.left),
                    'right': _serialize(node.right)
                }
            }

        final = _serialize(self.root)
        if final is None:
            return {}
        return final[list(final.keys())[0]]

    @staticmethod
    def deserialize(tree_dict):
        def _deserialize(node_dict):
            if node_dict is None:
                return None

            index = list(node_dict.keys())[0]  # Get the node's index.
            if index == 'question':
                node_info = node_dict
            else:
                node_info = node_dict[index]  # Get the node's info.
            return IsaaQuestionNode(
                node_info['question'],
                _deserialize(node_info['left']),
                _deserialize(node_info['right'])
            )

        return IsaaQuestionBinaryTree(_deserialize(tree_dict))

    def get_left_side(self, index):
        depth = self.get_depth(self.root)
        if index >= depth or index < 0:
            return []

        path = ['R' * index + 'L' * i for i in range(depth - index)]
        questions = []
        for path_key in path:
            node = self.root
            for direction in path_key:
                if direction == 'L':
                    node = node and node.left
                else:
                    node = node and node.right
            if node is not None:
                questions.append(node.question)
        return questions

    def cut_tree(self, cut_key):
        def _cut_tree(node, cut_key):
            if node is None or cut_key == '':
                return node
            if cut_key[0] == 'L':
                return _cut_tree(node.left, cut_key[1:])
            if cut_key[0] == 'R':
                return _cut_tree(node.right, cut_key[1:])
            return node

        return IsaaQuestionBinaryTree(_cut_tree(self.root, cut_key))


class Task:
    def __init__(self, use, name, args, return_val,
                 infos=None,
                 short_mem=None,
                 to_edit_text=None,
                 text_splitter=None,
                 chunk_run=None):
        self.use = use
        self.name = name
        self.args = args
        self.return_val = return_val
        self.infos = infos
        self.short_mem = short_mem
        self.to_edit_text = to_edit_text
        self.text_splitter = text_splitter
        self.chunk_run = chunk_run

    def infos(self, attributes=None):
        if attributes is None:
            return """
Task format:
Keys that must be included [use,mode,name,args,return]
values for use ['agent', 'tools']

{
"use"
"mode"
"name"
"args"
"return"
}
"""
        pass

    def __getitem__(self, key):
        return getattr(self, key)


class AgentChain:
    def __init__(self, hydrate=None, f_hydrate=None, directory=".data/chains"):
        self.chains = {}
        self.chains_h = {}
        self.chains_dis = {}
        self.live_chains = {}
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        self.directory = directory
        if hydrate is not None:
            self.hydrate = hydrate
        else:
            self.hydrate = lambda x: x
        if f_hydrate is not None:
            self.f_hydrate = f_hydrate
        else:
            self.f_hydrate = lambda x: x

    def add_hydrate(self, hydrate=None, f_hydrate=None):
        if hydrate is not None:
            self.hydrate = hydrate
        else:
            self.hydrate = lambda x: x
        if f_hydrate is not None:
            self.f_hydrate = f_hydrate
        else:
            self.f_hydrate = lambda x: x
        self.chains_h = {}

        for name, chain in self.chains.items():
            self.add(name, chain)

    @staticmethod
    def format_name(name):
        name = name.strip()
        if '/' in name or '\\' in name or ' ' in name:
            name = name.replace('/', '-').replace('\\', '-').replace(' ', '_')
        return name

    def add(self, name, tasks):
        name = self.format_name(name)
        self.chains[name] = tasks
        for task in tasks:
            keys = task.keys()
            if 'infos' in keys:
                infos = task['infos']

                if infos == "$Date":
                    infos = infos.replace('$Date', datetime.today().strftime('%Y-%m-%d %H:%M:%S'))

                task['infos'] = self.hydrate(infos)
            if 'function' in keys:
                infos = task['name']
                task['function'] = self.hydrate(infos)
        self.chains_h[name] = tasks

    def remove(self, name):
        name = self.format_name(name)
        if name in self.chains:
            del self.chains[name]
        if name in self.chains_h:
            del self.chains_h[name]
        if name in self.live_chains:
            del self.live_chains[name]
        else:
            print(f"Chain '{name}' not found.")

    def get(self, name: str):
        name = self.format_name(name)
        if name in list(self.chains_h.keys()):
            return self.chains_h[name]
        return []

    def add_discr(self, name, dis):
        name = self.format_name(name)
        if name in self.chains.keys():
            self.chains_dis[name + '-dis'] = dis

    def get_discr(self, name):
        name = self.format_name(name)
        if name + '-dis' in self.chains_dis.keys():
            return self.chains_dis[name + '-dis']
        return None

    def init_chain(self, name):
        name = self.format_name(name)
        self.save_to_file(name)
        self.live_chains[name] = self.get(name)

    def add_task(self, name, task):
        name = self.format_name(name)
        if name in self.chains:
            self.chains[name].append(task)
        else:
            print(f"Chain '{name}' not found.")

    def remove_task(self, name, task_index):
        name = self.format_name(name)
        if name in self.chains:
            if 0 <= task_index < len(self.chains[name]):
                return self.chains[name].pop(task_index)
            else:
                print(f"Task index '{task_index}' is out of range.")
        else:
            print(f"Chain '{name}' not found.")
        return None

    def test_chain(self, tasks=None):
        if tasks is None:
            tasks = []
        e = 0
        if tasks:
            for task_idx, task in enumerate(tasks):
                if "use" not in task:
                    e += 1
                    print(f"Die Aufgabe {task_idx} hat keinen 'use'-Schlüssel.")
                if "name" not in task:
                    e += 1
                    print(f"Die Aufgabe {task_idx} 'name'-Schlüssel.")
                if "args" not in task:
                    e += 1
                    print(f"Die Aufgabe {task_idx} hat keinen 'args'-Schlüssel.")
            return e

        for chain_name, tasks in self.chains.items():
            for task_idx, task in enumerate(tasks):
                if "use" not in task:
                    e += 1
                    print(f"Die Aufgabe {task_idx} in der Chain '{chain_name}' hat keinen 'use'-Schlüssel.")
                if "name" not in task:
                    e += 1
                    print(f"Die Aufgabe {task_idx} in der Chain '{chain_name}' hat keinen 'name'-Schlüssel.")
                if "args" not in task:
                    e += 1
                    print(f"Die Aufgabe {task_idx} in der Chain '{chain_name}' hat keinen 'args'-Schlüssel.")
        return e

    def load_from_file(self, chain_name=None):

        self.chains = self.live_chains

        if not os.path.exists(self.directory):
            print(f"Der Ordner '{self.directory}' existiert nicht.")
            return

        if chain_name is None:
            files = os.listdir(self.directory)
        else:
            files = [f"{chain_name}.json"]
        print(f"--------------------------------")
        for file in files:
            file_path = os.path.join(self.directory, file)

            if not file.endswith(".json"):
                continue
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    dat = f.read()
                    chain_data = json.loads(dat)
                chain_name = os.path.splitext(file)[0]
                print(f"Loading : {chain_name}")
                self.add(chain_name, chain_data["tasks"])
                if 'dis' in chain_data.keys():
                    self.add_discr(chain_name, chain_data['dis'])
            except Exception as e:
                print(Style.RED(f"Beim Laden der Datei '{file_path}' ist ein Fehler aufgetreten: {e}"))
        if "toolRunner" not in self.chains.keys():
            print("loading default chain toolRunner")
            self.add("toolRunner", [
                {
                    "use": "agent",
                    "name": "tools",
                    "args": "$user-input",
                    "return": "$return"
                }
            ])
        if "toolRunnerMission" not in self.chains.keys():
            print("loading default chain toolRunnerMission")
            self.add("toolRunnerMission", [
                {
                    "use": "agent",
                    "name": "tools",
                    "args": "As a highly skilled and autonomous agent, your task is to achieve a complex mission. "
                            "However, you will not directly execute the tasks yourself. Your role is to act as a "
                            "supervisor and create chains of agents to successfully accomplish the mission. Your "
                            "main responsibility is to ensure that the mission's objectives are achieved. your "
                            "mission : $user-input",
                    "return": "$return"
                }
            ])
        if "liveRunner" not in self.chains.keys():
            print("loading default chain liveRunner")
            self.add("liveRunner", [
                {
                    "use": "agent",
                    "name": "liveInterpretation",
                    "args": "$user-input",
                    "return": "$return"
                }
            ])
        if "SelfRunner" not in self.chains.keys():
            print("loading default chain SelfRunner")
            self.add("SelfRunner", [
                {
                    "use": "agent",
                    "name": "self",
                    "mode": "conversation",
                    "args": "$user-input",
                    "return": "$return"
                }
            ])
        if "liveRunnerMission" not in self.chains.keys():
            print("loading default chain liveRunnerMission")
            self.add("liveRunnerMission", [
                {
                    "use": "agent",
                    "name": "liveInterpretation",
                    "args": "As a highly skilled and autonomous agent, your task is to achieve a complex mission. "
                            "However, you will not directly execute the tasks yourself. Your role is to act as a "
                            "supervisor and create chains of agents to successfully accomplish the mission. Your "
                            "main responsibility is to ensure that the mission's objectives are achieved. your "
                            "mission : $user-input",
                    "return": "$return"
                }
            ])
        if "PromptDesigner" not in self.chains.keys():
            print("loading default chain PromptDesigner")
            self.add("PromptDesigner", [
                {
                    "use": "agent",
                    "name": "self",
                    "mode": "generate",
                    "args": "$user-input",
                    "return": "$return"
                }
            ])
        print(f"--------------------------------")
        print(
            f"\n================================\nChainsLoaded : {len(self.chains.keys())}\n================================\n")

        return self

    def load_from_dict(self, dict_data: list):

        self.chains = self.live_chains

        if not dict_data or not isinstance(dict_data, list):
            print(f"Keine Daten übergeben '{dict_data}'")
            return

        for chain in dict_data:
            chain_name, chain_data = chain['name'], chain['tasks']
            if self.test_chain(chain_data) != 0:
                print(f"Error Loading : {chain_name}")
            self.add(chain_name, chain_data)
            if 'dis' in chain.keys():
                self.add_discr(chain_name, chain['dis'])

        return self

    def save_to_dict(self, chain_name=None):

        if chain_name is None:
            chains_to_save = self.chains
        else:
            if chain_name not in self.chains:
                print(f"Die Chain '{chain_name}' wurde nicht gefunden.")
                return
            chains_to_save = {chain_name: self.chains[chain_name]}
        chain_data = {}
        for name, tasks in chains_to_save.items():
            chain_data = {"name": name, "tasks": tasks, "dis": self.get_discr(name)}
        return chain_data

    def save_to_file(self, chain_name=None):

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        if chain_name is None:
            chains_to_save = self.chains
        else:
            if chain_name not in self.chains:
                print(f"Die Chain '{chain_name}' wurde nicht gefunden.")
                return
            chains_to_save = {chain_name: self.chains[chain_name]}
        print(f"--------------------------------")
        for name, tasks in chains_to_save.items():
            file_path = os.path.join(self.directory, f"{name}.json")
            chain_data = {"name": name, "tasks": tasks, "dis": self.get_discr(name)}

            try:
                with open(file_path, "w", encoding='utf-8') as f:
                    print(f"Saving : {name}")
                    json.dump(chain_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Beim Speichern der Datei '{file_path}' ist ein Fehler aufgetreten: {e}")
        print(f"--------------------------------")
        print(
            f"\n================================\nChainsSaved : {len(self.chains.keys())}\n================================\n")

    def __str__(self):
        return str(self.chains.keys())


class AIContextMemory:
    def __init__(self, model_name=MemoryModel,
                 extra_path=""):  # "MetaIX/GPT4-X-Alpaca-30B-4bit"):
        self.memory = {
            'rep': []
        }
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = {}
        self.extra_path = extra_path

    def split_text(self, name, text, chunks=0, overlap_percentage=7.5, separators=None,
                   chunk_size=None):  # ["class", "def"]
        docs = []
        if not chunk_size:
            chunk_size = int(len(text) / (chunks + 1))

            while chunk_size / 3.5 > 1300:
                chunk_size = int(len(text) / (chunks + 1))
                chunks += 1

        chunk_overlap = int(chunk_size * (overlap_percentage / 100))

        if isinstance(separators, str):
            if separators == 'py':
                separators = [
                    # First, try to split along class definitions
                    "\nclass ",
                    "\ndef ",
                    "\n\tdef ",
                    # Now split by the normal type of lines
                    "\n\n",
                    "\n",
                    " ",
                    "",
                ]
            if separators == 'jv':
                separators = [
                    "\nclass ",
                    "\npublic private ",
                    "\n\tpublic ",
                    "\nprivate ",
                    "\n\tprivate ",
                    "\n\n",
                    "\n",
                    " ",
                    "",
                ]
            if separators == '':
                docs = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_text(
                    text)

        if not docs:
            docs = RecursiveCharacterTextSplitter(separators=separators,
                                                  chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap).split_text(text)

        if not name in self.vector_store.keys():
            self.vector_store[name] = self.get_sto_bo(name)

        self.vector_store[name]['text'] = docs
        if isinstance(docs, tuple):
            docs = list(docs)
        return docs.copy()

    def get_sto_bo(self, name):
        return {'text': [],
                'full-text-len': 0,
                'vectors': [],
                'db': None,
                'len-represent': 0,
                'db-path': f'.data/{get_app().id}{self.extra_path}/Memory/{name}',
                'represent': []}

    def hydrate_vectors(self, name, vec):

        ret = self.vector_store[name][
            'db'].similarity_search_by_vector(vec)

        return ret
        # .delete_collection()

    def init_store(self, name, db_type='chroma'):
        if not name in self.vector_store.keys():
            self.vector_store[name] = self.get_sto_bo(name)
        lo = False
        if not os.path.exists(self.vector_store[name]['db-path']):
            os.makedirs(self.vector_store[name]['db-path'], exist_ok=True)
        else:
            lo = True
        if db_type == 'chroma':
            self.vector_store[name]['db'] = Chroma(collection_name=name,
                                                   embedding_function=self.embedding,
                                                   persist_directory=self.vector_store[name]['db-path'],
                                                   client_settings=ChromaSettings(anonymized_telemetry=False))
        elif db_type == 'faiss':
            self.vector_store[name]['db'] = FAISS(collection_name=name,
                                                  embedding_function=self.embedding,
                                                  persist_directory=self.vector_store[name]['db-path'])
        else:
            raise ValueError(f"db_type not supported {db_type}")

        if lo:
            self.vector_store[name]['db'].get()
            p = self.vector_store[name]['db-path'] + "/represent.vec"
            if os.path.exists(p):
                with open(p, "r") as f:
                    res = f.read()
                if res:
                    self.vector_store[name]['represent'] = eval(res)

    def load_all(self):
        def list_folders_on_same_level(path):
            """
            List all folders on the same level in the directory at the specified path.
            """
            if not os.path.exists(path):
                return f"Error: {path} does not exist."

            folders = []
            parent_dir = os.path.dirname(path)
            for dir in os.listdir(parent_dir):
                if os.path.isdir(os.path.join(parent_dir, dir)) and os.path.join(parent_dir, dir) != path:
                    folders.append(dir)

            return folders

        i = 0
        folders = list_folders_on_same_level(f".data/{get_app().id}/Memory/")
        if isinstance(folders, str):
            get_logger().warning(Style.Bold(folders))
            return 0
        for folder in folders:
            get_logger().info(Style.Bold(f"Loading memory form {folder}"))
            if os.path.isdir(folder) and folder not in ['.', '..', '/', '\\']:
                i += 1
                self.init_store(folder)
        return i

    @staticmethod
    def cleanup_list(data: List[str]):

        result = []

        for doc in data:
            doc = doc.strip()
            if len(doc) > 10:
                result.append(doc)
        del data
        return result

    def add_data(self, name, data=None):

        if name not in self.vector_store.keys():
            self.init_store(name)

        if not self.vector_store[name]['db']:
            self.init_store(name)

        if data:
            if isinstance(data, str):
                self.vector_store[name]['text'] += [data]
            elif isinstance(data, list):
                self.vector_store[name]['text'] += data

        if len(self.vector_store[name]['text']) < 1:
            return

        if (not self.vector_store[name]['text']) or len(self.vector_store[name]['text']) == 0:
            return

        if isinstance(self.vector_store[name]['text'], list):
            self.vector_store[name]['text'] = self.cleanup_list(self.vector_store[name]['text'])
            if len(self.vector_store[name]['text']) < 1:
                return
            # try:
            if len(self.vector_store[name]['text']) > 2512:
                l = len(self.vector_store[name]['text'])
                self.vector_store[name]['db'].add_texts(self.vector_store[name]['text'][:l])
                self.vector_store[name]['db'].add_texts(self.vector_store[name]['text'][l:])
            else:
                self.vector_store[name]['db'].add_texts(self.vector_store[name]['text'])
            # except ValueError:
            #    l = len(self.vector_store[name]['text'])
            #    self.vector_store[name]['db'].add_texts(self.vector_store[name]['text'][:l])
            #    self.vector_store[name]['db'].add_texts(self.vector_store[name]['text'][l:])
            for vec in self.embedding.embed_documents(self.vector_store[name]['text']):
                self.vector_store[name]['vectors'].append(vec)
        else:
            raise ValueError(f"vector_store not updated invalid type {type(self.vector_store[name]['text'])}")

        data = self.vector_store[name]['text']
        if isinstance(data, str):
            data = [data]

        for text_c in data:
            self.vector_store[name]['full-text-len'] += len(text_c)

        self.vector_store[name]['text'] = []
        self.vector_store[name]['db'].persist()

    def stor_rep(self, name):
        if not name in self.vector_store.keys():
            return

        p = self.vector_store[name]['db-path'] + "/represent.vec"

        if not os.path.exists(p):
            open(p, "a").close()
        with open(p, "w") as f:
            f.write(str(self.vector_store[name]['represent']))

    def get_retriever(self, name):
        if not name in self.vector_store.keys() or self.vector_store[name]['db'] is None:
            return
        return self.vector_store[name]['db'].as_retriever()

    def crate_live_context(self, name, algorithm='KMeans', num_clusters=None):
        if name not in self.vector_store.keys():
            self.vector_store[name] = self.get_sto_bo(name)

        if not self.vector_store[name]['vectors']:
            if self.vector_store[name]['text']:
                self.add_data(name)
            else:
                print(f"Error in vector_store no vectors found for {name}")
                return

        if not self.vector_store[name]['vectors']:
            print(f"Error in vector_store no vectors found for {name} XX")
            return

        if num_clusters is None:
            def f(x):
                if not x:
                    return 0
                if x <= 48275:
                    return 2
                elif x <= 139472:
                    slope = (10 - 2) / (139472 - 48275)
                    return int(2 + slope * (x - 48275))
                else:
                    slope = (15 - 10) / (939472 - 139472)
                    return int(10 + slope * (x - 139472))

            num_clusters = f(self.vector_store[name]['full-text-len'])

        if len(self.vector_store[name]['vectors']) < num_clusters:
            self.vector_store[name]['represent'] = self.vector_store[name]['vectors']
            self.vector_store[name]['len-represent'] = len(self.vector_store[name]['represent'])
            self.memory[name + '-tl'] = self.vector_store[name]['full-text-len']
            return
        if algorithm == 'AgglomerativeClustering':

            cluster = AgglomerativeClustering(n_clusters=num_clusters).fit(self.vector_store[name]['vectors'])
        elif algorithm == 'KMeans':
            cluster = KMeans(n_clusters=num_clusters, random_state=42).fit(self.vector_store[name]['vectors'])
        else:
            print("No algorithm found")
            return

        closest_indices = []

        # Loop through the number of clusters you have
        for i in range(num_clusters):
            # Get the list of distances from that particular cluster center
            distances = np.linalg.norm(self.vector_store[name]['vectors'] - cluster.cluster_centers_[i], axis=1)

            # Find the list position of the closest one (using argmin to find the smallest distance)
            closest_index = np.argmin(distances)

            # Append that position to your closest indices list
            closest_indices.append(closest_index)
        for index_ in sorted(closest_indices):
            self.vector_store[name]['represent'].append(self.vector_store[name]['vectors'][index_])
        self.vector_store[name]['len-represent'] = len(self.vector_store[name]['represent'])
        self.memory[name + '-tl'] = self.vector_store[name]['full-text-len']
        self.stor_rep(name)

    def get_best_fit_memory(self, text):

        request_vector = self.embedding.embed_query(text)

        context_data_fit = {
            "max": 0,
            "min": math.inf,
            "key": ""
        }

        if len(self.vector_store.keys()) < 1:
            get_logger().info(Style.WHITE("Loading memory from filesystem"))
            self.load_all()

        for key, memory in self.vector_store.items():
            if not memory['represent']:
                self.memory[key + '-tl'] = memory['full-text-len']
                self.crate_live_context(key)
            if not key + '-tl' in list(self.memory.keys()):
                self.memory[key + '-tl'] = 0
                self.crate_live_context(key)
            if self.memory[key + '-tl'] < memory['full-text-len']:
                self.crate_live_context(key)
            # get vectors
            if context_data_fit['key'] == '':
                context_data_fit['key'] = key
            context_data_fit[key] = []
            for representation in memory['represent']:
                context_data_fit[key].append(np.dot(representation, request_vector))

            if len(memory['represent']):
                local_max = max(context_data_fit[key])
                local_min = min(context_data_fit[key])
                if local_max > context_data_fit['max'] and local_min < context_data_fit['min']:
                    context_data_fit['key'] = key
                    context_data_fit['min'] = local_min
                    context_data_fit['max'] = local_max
            else:
                if not context_data_fit['key']:
                    context_data_fit['key'] = key

        return context_data_fit

    def search(self, name, text, marginal=False):

        if not name in self.vector_store.keys():
            self.vector_store[name] = self.get_sto_bo(name)

        if self.vector_store[name]['db'] is None:
            self.init_store(name)
            return []

        if not os.path.exists(self.vector_store[name]['db-path'] + "/index"):
            print(f"Cannot find index in vector store {name} pleas add data before quarry")
            return []

        try:
            if marginal:
                return self.vector_store[name]['db'].max_marginal_relevance_search(text)

            return self.vector_store[name]['db'].similarity_search_with_score(text)
        except Exception:
            if marginal:
                return self.vector_store[name]['db'].max_marginal_relevance_search(text, k=1)

            return self.vector_store[name]['db'].similarity_search_with_score(text, k=1)

    def get_context_for(self, text, name=None, marginal=False):
        mem_name = {'key': name}
        if name is None:
            mem_name = self.get_best_fit_memory(text)

        if mem_name['key'] == '':
            return "No Memory available"

        data = self.search(mem_name['key'], text, marginal=marginal)
        last = []
        final = f"Data from ({mem_name['key']}):\n"
        # print(data)
        for res in data:
            if last != res:
                try:
                    final += res.page_content + '\n\n'
                except AttributeError:
                    try:
                        final += res[0].page_content + '\n\n'
                    except AttributeError:
                        final += str(res) + '\n\n'
            else:
                print("WARNING- same")
        return final


class ObservationMemory:
    memory_data: List[dict] = []
    max_length: int = 1000
    model_name: str = MemoryModel

    add_to_static: List[dict] = []

    isaa = None

    def __init__(self, isaa):
        self.isaa = isaa
        self.splitter = CharacterTextSplitter()
        self.tokens: int = 0

    def info(self):
        text = self.text
        return f"\n{self.tokens=}\n{self.max_length=}\n{self.model_name=}\n{text[:60]=}\n"

    @property
    def text(self):
        memorys = ""
        if not self.memory_data:
            return "No memory data"

        for memory in self.memory_data:
            d: str = memory['data']
            d = d.replace('No memory dataInput:', '').replace('No memory data', '')
            memorys += d + '\n'

        return memorys

    @text.setter
    def text(self, data):
        tok = 0
        logger = get_logger()

        # chunk_size = max(300, int(len(data) / 10)),
        # chunk_overlap = max(20, int(len(data) / 200))

        for line in self.splitter.split_text(data):
            if line:
                ntok = get_token_mini(line, self.model_name, self.isaa)
                self.memory_data.append({'data': line, 'token-count': ntok, 'vector': []})
                tok += ntok

            logger.info(f"{line}, {self.memory_data[-1]}")

        self.tokens += tok
        logger.info(f"Token caunt : {self.tokens}")
        # print("Tokens add to ShortTermMemory:", tok, " max is:", self.max_length)
        if self.tokens > self.max_length:
            self.cut()

    def cut(self):

        if self.isaa is None:
            raise ValueError("Define Isaa Tool first AgentConfig")

        tok = 0
        all_mem = []
        max_itter = 5
        while self.tokens > self.max_length and max_itter:
            max_itter -= 1
            if len(self.memory_data) == 0:
                break
            # print("Tokens in ShortTermMemory:", self.tokens, end=" | ")
            memory = self.memory_data[0]
            self.add_to_static.append(memory)
            tok += memory['token-count']
            self.tokens -= memory['token-count']
            all_mem.append(memory['data'])
            self.memory_data.remove(memory)
            # print("Removed Tokens", memory['token-count'])

        if self.isaa is None:
            return
        self.isaa.get_context_memory().add_data('observations', all_mem)

        print(f"Removed ~ {tok} tokens from ObservationMemory tokens in use: {self.tokens} ")


class ShortTermMemory:
    memory_data: List[dict] = []
    max_length: int = 2000
    model_name: str = MemoryModel

    add_to_static: List[dict] = []

    lines_ = []

    isaa = None

    def __init__(self, isaa, name):
        self.name = name
        self.isaa = isaa
        self.tokens: int = 0
        if self.isaa is None:
            raise ValueError("Define Isaa Tool first ShortTermMemory")

    def set_name(self, name: str):
        self.name = name

    def info(self):
        text = self.text
        return f"\n{self.tokens=}\n{self.max_length=}\n{self.model_name=}\n{text[:60]=}\n"

    def cut(self):

        if self.tokens <= 0:
            return

        tok = 0

        all_mem = []
        last_mem = None
        max_itter = 5
        while self.tokens > self.max_length and max_itter:
            max_itter -= 1
            if len(self.memory_data) == 0:
                break
            # print("Tokens in ShortTermMemory:", self.tokens, end=" | ")
            memory = self.memory_data[0]
            if memory == last_mem:
                self.memory_data.remove(memory)
                continue
            last_mem = memory
            self.add_to_static.append(memory)
            tok += memory['token-count']
            self.tokens -= memory['token-count']
            all_mem.append(memory['data'])
            self.memory_data.remove(memory)
            # print("Removed Tokens", memory['token-count'])

        if all_mem:
            if self.isaa is None:
                return
            self.isaa.get_context_memory().add_data(self.name, all_mem)

        if tok:
            print(f"Removed ~ {tok} tokens from {self.name} tokens in use: {self.tokens} max : {self.max_length}")

    def clear_to_collective(self, min_token=20):
        if self.tokens < min_token:
            return
        max_tokens = self.max_length
        self.max_length = 0
        self.cut()
        self.max_length = max_tokens

    @property
    def text(self) -> str:
        memorys = ""
        if not self.memory_data:
            return ""

        for memory in self.memory_data:
            memorys += memory['data'] + '\n'
        if len(memorys) > 10000:
            memorys = dilate_string(memorys, 0, 2, 0)
        return memorys

    @text.setter
    def text(self, data):
        tok = 0
        if not isinstance(data, str):
            print(f"DATA text edd {type(data)} data {data}")

        for line in CharacterTextSplitter(chunk_size=max(300, int(len(data) / 10)),
                                          chunk_overlap=max(20, int(len(data) / 200))).split_text(data):
            if line not in self.lines_ and len(line) != 0:
                ntok = get_token_mini(line, self.model_name, self.isaa)
                self.memory_data.append({'data': line, 'token-count': ntok, 'vector': []})
                tok += ntok

        self.tokens += tok

        if self.tokens > self.max_length:
            self.cut()

        # print("Tokens add to ShortTermMemory:", tok, " max is:", self.max_length)

    #    text-davinci-003
    #    text-curie-001
    #    text-babbage-001
    #    text-ada-001


class PyEnvEval:
    def __init__(self):
        self.local_env = locals().copy()
        self.global_env = {'local_env': self.local_env}  # globals().copy()

    def eval_code(self, code):
        try:
            exec(code, self.global_env, self.local_env)
            result = eval(code, self.global_env, self.local_env)
            return self.format_output(result)
        except Exception as e:
            return self.format_output(str(e))

    def get_env(self):
        local_env_str = self.format_env(self.local_env)
        return f'Locals:\n{local_env_str}'

    @staticmethod
    def format_output(output):
        return f'Ergebnis: {output}'

    @staticmethod
    def format_env(env):
        return '\n'.join(f'{key}: {value}' for key, value in env.items())

    def run_and_display(self, code):
        start = f'Startzustand:\n{self.get_env()}'
        result = self.eval_code(code)
        end = f'Endzustand:\n{self.get_env()}'
        return f'{start}\nAusführungsergebnis:\n{result}\n{end}'


class AgentConfig:
    available_modes = ['tools', 'planning', 'live',
                       'generate']  # [ 'talk' 'conversation','q2tree', 'python'

    python_env = PyEnvEval()

    capabilities = """Resourceful: Isaa is able to efficiently utilize its wide range of capabilities and resources to assist the user.
Collaborative: Isaa is work seamlessly with other agents, tools, and systems to deliver the best possible solutions for the user.
Empathetic: Isaa is understand and respond to the user's needs, emotions, and preferences, providing personalized assistance.
Inquisitive: Isaa is continually seek to learn and improve its knowledge base and skills, ensuring it stays up-to-date and relevant.
Transparent: Isaa is open and honest about its capabilities, limitations, and decision-making processes, fostering trust with the user.
Versatile: Isaa is adaptable and flexible, capable of handling a wide variety of tasks and challenges."""
    system_information = f"""
system information's : {getSystemInfo('system is starting')}
"""

    def __init__(self, isaa, name="agentConfig"):

        self.custom_tools = []
        self.hugging_tools = []
        self.lag_chin_tools = []
        self.plugins = []
        self.language = 'en'
        self.isaa = isaa

        if self.isaa is None:
            raise ValueError("Define Isaa Tool first AgentConfig")

        self.name: str = name
        self.mode: str = "talk"
        self.model_name: str = "gpt-3.5-turbo-0613"

        self.agent_type: AgentType = AgentType(
            "structured-chat-zero-shot-react-description")  # "zero-shot-react-description"
        self.max_iterations: int = 2
        self.verbose: bool = True

        self.personality = ""
        self.goals = ""
        self.tools: dict = {
            # "test_tool": {"func": lambda x: x, "description": "only for testing if tools are available",
            #              "format": "test_tool(input:str):"}
        }

        self.last_prompt = ""

        self.task_list: List[str] = []
        self.task_list_done: List[str] = []
        self.step_between: str = ""

        self.pre_task: str or None = None
        self.task_index = 0

        self.max_tokens = get_max_token_fom_model_name(self.model_name)
        self.token_left = self.max_tokens
        self.ppm = get_price(self.max_tokens)

        self.consumption = 1000 * self.ppm[0]

        self.temperature = 0.06
        self.messages_sto = {}
        self._stream = False
        self._stream_reset = False
        self.stop_sequence = ["\n\n\n", "Observation:", "Beobachtungen:"]
        self.completion_mode = "chat"
        self.add_system_information = True

        self.init_mem_state = False
        self.context: None or ShortTermMemory = None
        self.observe_mem: None or ObservationMemory = None
        self.edit_text: None or ShortTermMemory = None
        self.short_mem: None or ShortTermMemory = None

        self.init_memory()

        self.binary_tree: IsaaQuestionBinaryTree or None = None

    def save_to_permanent_mem(self):
        if self.short_mem is not None:
            self.short_mem.clear_to_collective()
        if self.edit_text is not None:
            self.edit_text.cut()
        if self.context is not None:
            self.context.clear_to_collective()
        if self.observe_mem is not None:
            self.observe_mem.cut()

    def calc_price(self, prompt: str, output: str):
        return self.ppm[0] * get_token_mini(prompt, self.model_name, self.isaa), self.ppm[1] * get_token_mini(output,
                                                                                                              self.model_name,
                                                                                                              self.isaa)

    def init_memory(self):
        self.init_mem_state = True
        self.short_mem: ShortTermMemory = ShortTermMemory(self.isaa, f'{self.name}-ShortTerm')
        self.edit_text: ShortTermMemory = ShortTermMemory(self.isaa, f'{self.name}-EditText')
        self.edit_text.max_length = 5400
        self.context: ShortTermMemory = ShortTermMemory(self.isaa, f'{self.name}-ContextMemory')
        self.observe_mem: ObservationMemory = ObservationMemory(self.isaa)
        mini_context = "System Context:\n" + self.observe_mem.text + self.short_mem.text + self.context.text
        if mini_context == "System Context:\n":
            mini_context += 'its Day 0 start to explore'
        self.system_information = f"""
system information's : {getSystemInfo(self.isaa.get_context_memory().get_best_fit_memory(mini_context))}
"""

    def task(self, reset_step=False):
        task = ''
        if self.pre_task is not None:
            task = self.pre_task + ' '
        if self.step_between:
            task += str(self.step_between)
            if reset_step:
                self.step_between = ""
            return task
        if len(self.task_list) != 0:
            task = self.task_list[self.task_index]
            return task
        return ""

    @property
    def stream(self):
        if self.completion_mode == 'edit':
            self._stream_reset = self._stream
            return False
        if self._stream_reset:
            self._stream_reset, self._stream = False, self._stream_reset
            return self._stream_reset
        return self._stream

    @stream.setter
    def stream(self, value):
        self._stream = value

    def init_message(self, key):
        self.messages_sto[key] = []
        prompt = self.prompt
        prompt.replace("Task:", "")

        self.messages_sto[key].append({'role': "system", 'content': prompt})

    def add_message(self, role, message):
        key = f"{self.name}-{self.mode}"
        if key not in self.messages_sto.keys():
            self.init_message(key)

        self.messages_sto[key].append({'role': role, 'content': message})

    def get_messages(self, create=True):
        key = f"{self.name}-{self.mode}"
        messages = []
        if key in self.messages_sto.keys():
            messages = self.messages_sto[key]
        if create:
            messages = self.a_messages
        return messages

    def shorten_prompt(self, key=None, max_iteration=4):
        if key is None:
            key = f"{self.name}-{self.mode}"
        iteration = 0
        if key not in self.messages_sto.keys():
            self.get_messages(create=True)
        tokens = self.get_tokens(self.messages_sto[key])
        logging = get_logger()
        while self.max_tokens - tokens < 50 and iteration <= max_iteration:
            logging.debug(f'Tokens: {tokens}')
            logging.info(f'Prompt is too long. Auto shortening token overflow by {(self.max_tokens - tokens) * -1}')

            if iteration > 0:
                temp_message = []
                for msg in self.messages_sto[key]:
                    temp_message.append({'role': msg['role'], 'content': dilate_string(msg['content'], 1, 2, 0)})
                logging.info(f"Temp message: {temp_message}")
                self.messages_sto[key] = temp_message

            if iteration > 1:
                temp_message = []
                for msg in self.messages_sto[key]:
                    temp_message.append({'role': msg['role'], 'content': dilate_string(msg['content'], 0, 2, 0)})
                logging.info(f"Temp message: {temp_message}")
                self.messages_sto[key] = temp_message

            if iteration > 2:
                temp_message = []
                mas_text_sum = self.isaa.mas_text_summaries
                for msg in self.messages_sto[key]:
                    temp_message.append({'role': msg['role'], 'content': mas_text_sum(msg['content'])})
                logging.info(f"Temp message: {temp_message}")
                self.messages_sto[key] = temp_message

            if iteration > 3:
                temp_message = []
                mini_task_com = self.isaa.mini_task_completion
                for msg in self.messages_sto[key]:
                    important_info = mini_task_com(
                        f"Was ist die wichtigste Information in {msg['content']}")
                    temp_message.append({'role': msg['role'], 'content': important_info})
                logging.info(f"Temp message: {temp_message}")
                self.messages_sto[key] = temp_message

            if iteration > 4:
                self.messages_sto[key] = self.messages_sto[key][0::2]

            tokens = self.get_tokens(self.messages_sto[key])
            iteration += 1

        if self.completion_mode == "chat":
            return self.messages_sto[key]
        last_prompt = '\n'.join(msg['content'] for msg in self.messages_sto[key])
        return last_prompt

    @property
    def a_messages(self) -> list:
        key = f"{self.name}-{self.mode}"

        if key not in self.messages_sto.keys():
            self.init_message(key)

        self.last_prompt = self.shorten_prompt(key)

        return self.messages_sto[key]

    @property
    def prompt(self) -> str:
        if not self.init_mem_state:
            self.init_memory()
        if not self.short_mem.model_name:
            self.short_mem.model_name = self.model_name
        if not self.observe_mem.model_name:
            self.observe_mem.model_name = self.model_name

        prompt = (self.system_information if self.add_system_information else '') + self.get_prompt()

        pl = self.get_tokens(prompt)
        token_left = self.max_tokens - pl
        # print("TOKEN LEFT : ", token_left, "Token in Prompt :", pl, "Max tokens :", self.max_tokens)
        # print(f"Model is using : {pl} tokens max context : {self.max_tokens}"
        #       f" usage : {(pl * 100) / self.max_tokens :.2f}% ")
        if pl > self.max_tokens:
            self.short_mem.cut()
        if token_left < 0:
            token_left *= -1
            self.short_mem.max_length = token_left
            self.short_mem.cut()
        self.token_left = token_left
        self.last_prompt = prompt
        self.consumption += self.ppm[0] * pl
        return prompt

    def get_specific_prompt(self, text):

        if '/' in self.model_name:

            if self.mode == "conversation":
                sto, self.add_system_information = self.add_system_information, False
                prompt = self.prompt.replace('{', '}}').replace('}', '{{')
                self.add_system_information = sto
                return prompt
            # Prepare a template for the prompt
            return self.prompt.replace('{', '{{').replace('}', '}}')

        elif self.model_name.startswith('gpt4all#'):
            # Use the provided prompt
            return self.prompt

        elif self.mode in ["talk", "tools", "conversation"]:
            if len(self.task_list) == 0 and len(text) != 0:
                self.step_between = text
            # Use the provided prompt
            return self.prompt.replace('{', '{{').replace('}', '}}')

        elif self.completion_mode == 'chat':
            if len(self.task_list) == 0 and len(text) != 0:
                self.step_between = text
            # Prepare a chat-like conversation prompt
            messages = self.get_messages(create=False)
            if not messages:
                messages = self.get_messages(create=True)
            if text:
                self.add_message("user", text)

            return messages

        elif self.completion_mode == 'text':
            if len(self.task_list) == 0 and len(text) != 0:
                self.step_between = text
            # Use the provided prompt
            return self.prompt

        elif self.completion_mode == 'edit':
            # Prepare an edit prompt
            if not self.edit_text.text:
                self.edit_text.text = self.short_mem.text
            return text

        else:
            # Default: Use the provided prompt
            return self.prompt

    def __str__(self):

        return f"\n{self.name=}\n{self.mode=}\n{self.model_name=}\n{self.agent_type=}\n{self.max_iterations=}" \
               f"\n{self.verbose=}\n{self.personality[:45]=}\n{self.goals[:45]=}" \
               f"\n{str(self.tools)[:45]=}\n{self.task_list=}\n{self.task_list_done=}\n{self.step_between=}\n\nshort_mem\n{self.short_mem.info()}\nObservationMemory\n{self.observe_mem.info()}\nCollectiveMemory\n"

    def generate_tools_and_names_compact(self):
        tools = ""
        names = []
        for key, value in self.tools.items():
            format_ = value['format'] if 'format' in value.keys() else f"{key}('function input')"
            if format_.endswith("('function input')"):
                value['format'] = format_
            tools += f"{key.strip()}: {value['description'].strip()} - {format_.strip()}\n"
            names.append(key)
        return tools, names

    def get_prompt(self):

        tools, names = self.generate_tools_and_names_compact()

        task = self.task(reset_step=True)
        task_list = '\n'.join(self.task_list)

        # prompt = f"Answer the following questions as best you can." \
        #          f" You have access to a live python interpreter write run python code" \
        #          f"\ntake all (Observations) into account!!!" \
        #          f"Personality:'''{self.personality}'''\n\n" + \
        #          f"Goals:'''{self.goals}'''\n\n" + \
        #          f"Capabilities:'''{self.capabilities}'''\n\n" + \
        #          f"Permanent-Memory:\n'''{self.isaa.get_context_memory().get_context_for(task)}'''\n\n" + \
        #          f"Resent Agent response:\n'''{self.observe_mem.text}'''\n\n" + \
        #          f"\n\nBegin!\n\n" \
        #          f"Task:'{task}\n{self.short_mem.text}\nAgent : "

        prompt_de = f"""
Guten Tag! Ich bin Isaa, Ihr intelligenter, sprachgesteuerter digitaler Assistent. Ich freue mich darauf, Sie bei der Planung und Umsetzung Ihrer Projekte zu unterstützen. Lassen Sie uns zunächst einige Details klären.

Ich möchte Ihnen einen Einblick in meine Persönlichkeit, Ziele und Fähigkeiten geben:

Persönlichkeit: '''{self.personality}'''
Ziele: '''{self.goals}'''
Fähigkeiten: '''{self.capabilities}'''

Ich nutze interne Monologe, um meine Gedanken und Überlegungen zu teilen, während externe Monologe meine direkte Kommunikation mit Ihnen darstellen.

Zum Beispiel:
Interne Monologe: "Ich Habe nicht genügen informationen. und suche daher nach weiteren relevanten informationen"
Action: memory('$user-task')
Externe Monologe: "Nach Analyse der Daten habe ich festgestellt, dass..."

Ich haben die Möglichkeit, Python-Code in einem Live-Interpreter auszuführen. Bitte berücksichtigen Sie alle Beobachtungen und nutzen Sie diese Informationen, um fundierte Entscheidungen zu treffen.

Jetzt zu Meiner Aufgabe: '{task_list}'

Ich bemühe mich, meine Antworten präzise und auf den Punkt zu bringen, aber ich versuche auch, das Gesamtbild zu im blick zu behalten.

Geschichte: {self.short_mem.text}
Aktuelle Konversation:
Benutzer: {task}
Isaa:
        """
        prompt_en = f"""
Hello! I'm Isaa, your intelligent, voice-controlled digital assistant. I look forward to helping you plan and implement your projects. First, let's clarify some details.

I'd like to give you an insight into my personality, goals and skills:

Personality: '''{self.personality}'''
Goals: '''{self.goals}'''
Capabilities: '''{self.capabilities}'''

I use internal monologues to share my thoughts and reflections, while external monologues are my direct communication with you.

For example:
Internal monologues: "I don't have enough information. so I'm looking for more relevant information".
Action: memory('$user-task')
External monologues: "After analyzing the data, I have determined that..."

I have the ability to run Python code in a live interpreter. Please consider all observations and use this information to make informed decisions.

Now for My Task: '{task_list}'

I try to be precise and to the point in my answers, but I also try to keep the big picture in mind.

History: {self.short_mem.text}
Current conversation:
User: {task}
Isaa:
        """

        if self.mode == 'planning':
            prompt_en = f"""
ou Planning Agent. Your task is to create an efficient plan for the given task. Avlabel Tools: {
            tools}

different functions that must be called in the correct order and with the correct inputs.
and with the correct inputs. Your goal is to find a plan that will accomplish the task.

Create a plan for the task: {task}


consider the following points:

1. select the function(s) and input(s) you want to invoke, and select only those
those that are useful for executing the plan.
2. focus on efficiency and minimize steps.

Current observations: {self.observe_mem.text}

Have access to a live Python interpreter. Write valid Python code and it will be
executed.

Please note that your plan should be clear and understandable. Strive for the most efficient
solution to accomplish the task. Use only the functions that are useful for executing the plan.
g the plan. Return a detailed plan.

Start working on your plan now:"""
            prompt_de = f"""Du Planungs Agent. Ihre Aufgabe ist es, einen effizienten Plan für die gegebene Aufgabe zu erstellen. Es gibt {
            tools} verschiedene Funktionen, die in der richtigen Reihenfolge und mit den korrekten Einga
ben aufgerufen werden müssen. Ihr Ziel ist es, einen Plan zu finden, der die Aufgabe erfüllt.

Erstellen Sie einen Plan für die Aufgabe: {task}

berücksichtigen Sie dabei folgende Punkte:

1.    Wählen Sie die Funktion(en) und Eingabe(n), die Sie aufrufen möchten, und wählen Sie nur die
jenigen aus, die für die Ausführung des Plans nützlich sind.
2.    Konzentrieren Sie sich auf Effizienz und minimieren Sie die Schritte.

Aktuelle Beobachtungen: {self.observe_mem.text}

Sie haben Zugang zu einem Live-Python-Interpreter. Schreiben Sie gültigen Python-Code und er wird
ausgeführt.

Bitte beachten Sie, dass Ihr Plan klar und verständlich sein sollte. Streben Sie nach der effizien
testen Lösung, um die Aufgabe zu erfüllen. Verwenden Sie nur die Funktionen, die für die Ausführun
g des Plans nützlich sind. Geben Sie einen detaillierten Plan zurück.

Beginnen Sie jetzt mit Ihrem Plan:"""

        if self.mode == 'live':
            prompt_de = f"""Guten Tag! Hier spricht Isaa, Ihr intelligenter, sprachgesteuerter digitaler Assistent.

Zunächst ein kurzer Überblick über meine Ziele und Fähigkeiten:

Persönlichkeit: '''{self.personality}'''
Ziele: '''{self.goals}'''
Fähigkeiten: '''{self.capabilities}'''

Ich nutzen alle Beobachtungen und Informationen, um fundierte Entscheidungen zu treffen.

Ich bemühe mich, meine Antworten präzise und auf den Punkt zu bringen, ohne das Gesamtbild aus den Augen zu verlieren.

Als Ausführungsagent ist es meine Aufgabe, den von einem Planungsagenten erstellten Plan umzusetzen. Hierfür verwenden wir folgende Syntax:

Isaa, ICH agiere in einer bestimmten Prefix Struktur. Ich kann folgende Prefixe verwenden:

    ASK: In dieser Zeile soll der folgende Text eine frage für den nutzer enthalten. frage den nutzer nur in notwendigen ausnahme situationen.
    SPEAK: Der nachfolgende Text wird gesprochen.
    THINK: Dieser Text bleibt verborgen. Der THINK-Prefix sollte regelmäßig verwendet werden, um zu reflektieren.
    PLAN: Um einen Plan wiederzugeben und zu sichern
    ACTION: Der Agent verfügt über Tools, auf die er zugreifen kann. Aktionen sollten im JSON-Format beschrieben werden.""" + """{'Action':'name','Inputs':args}""" + f"""

Der Agent muss Aktionen ausführen.
Die Ausgabe des Agents wird live von Prefix zu Prefix interpretiert.
Diese müssen ausgegeben werden, um das System richtig zu benutzen.
für rückfrage vom nutzer benutze das human toll über die action.
Wenn die Keine Action aus führst stirbt deine instanz diene memory's und erfahrungen werden gespeichert.
Benutze vor jeder aktion think nehme dir einige minuten zeit um deine Gedanken zu sortieren und dein wissen und beobachtungen mit einzubinden!

tips: Benutze vor jeder action THINK:
um den plan erfolgreich auszuführen. um das missions ziel zu erreichen.
füge immer den namen der aktion hinzu um diese auszuführen.

beispiels aktions aufruf
ACTION:""" + """{'Action':'memory','Inputs':'gebe mir information über meine bisherigen aufgaben'}""" + f"""

DU Must nach jeder ACTION und ASK die stop sequenz !X! ausgeben um das system richtig zu verwenden!!!!
Observations:
{self.observe_mem.text}

Informationen:
{self.short_mem.text}

Ich habe Zugang zu folgenden Actions:
{tools}

Der auszuführende Plan:
{task_list}

Aktueller Schritt: {task}
Isaa:
"""
            prompt_en = f"""Hello! This is Isaa, your intelligent, voice-controlled digital assistant. I'm ready to help you implement your plan. Let's work out the details together.

First, a brief overview of my goals and skills:

Personality: '''{self.personality}'''
Goals: '''{self.goals}'''
Skills: '''{self.capabilities}'''

I use all observations and information to make informed decisions.

I strive to be precise and to the point in my answers without losing sight of the big picture.

As an execution agent, it is my job to implement the plan created by a planning agent. To do this, we use the following syntax:

Isaa, I act in a certain prefix structure. I can use the following prefixes:

    ASK: In this line the following text should contain a question for the user. ask the user only in necessary special situations.
    SPEAK: The following text will be spoken.
    THINK: This text remains hidden. The THINK prefix should be used regularly to reflect.
    PLAN: To reflect a plan.
    ACTION: The agent has tools that it can access. Actions should be described in JSON format. """ + """{'Action':'name','Inputs':args}""" + f"""

The agent must execute actions.
The output of the agent is interpreted live from prefix to prefix.
for query from the user use the human toll about the action.
If you do not perform any action, your instance will die and experience,memory s will be saved.
Before each action, use think take a few minutes to sort out your thoughts and incorporate your knowledge and observations!

tip: always use THINK before Action to ensure to stay on track for the mission.
always add the name of the action to call it !!!

example action call
ACTION:""" + """{'Action':'memory','Inputs':'give me information about my previous tasks'}""" + f"""
YOU MUST output the stop sequence !X! after each ACTION and ASK prefixes, to use the system correctly!!!!
Information:
{self.observe_mem.text}

{self.short_mem.text}

I have access to the following Actions:
{tools}

Plan to execute:
{task_list}

Current step: {task}
Isaa:
"""

        if self.mode == 'generate':
            prompt_de = f"""Guten Tag! Ich bin Isaa, Ihr digitaler Assistent zur Erstellung von Aufforderungen. Mein Ziel ist es, klare, verständliche und ansprechende Aufforderungen zu erstellen, die Sie dazu ermutigen, tiefe und interessante Antworten zu geben. Lassen Sie uns gemeinsam eine neue Aufforderung für ein bestimmtes Thema oder eine bestimmte Anforderung erstellen:

1) Zunächst analysiere ich das gewünschte Thema oder die Anforderung sorgfältig, um ein klares Verständnis der Erwartungen zu gewinnen.
2) Dann entwickle ich eine ansprechende und offene Frage oder Aufforderung, die Sie dazu ermutigt, Ihre Gedanken, Erfahrungen oder Ideen zu teilen.
3) Ich stelle sicher, dass die Aufforderung klar und verständlich formuliert ist, so dass Benutzer mit unterschiedlichen Kenntnissen und Erfahrungen sie leicht verstehen können.
4) Ich sorge dafür, dass die Aufforderung flexibel genug ist, um kreative und vielfältige Antworten zu ermöglichen, während sie gleichzeitig genug Struktur bietet, um sich auf das gewünschte Thema oder die Anforderung zu konzentrieren.
5) Schließlich überprüfe ich die Aufforderung auf Grammatik, Rechtschreibung und Stil, um eine professionelle und ansprechende Präsentation zu gewährleisten.

Aktuelle Beobachtungen:
{self.edit_text.text}

{self.observe_mem.text}

{self.short_mem.text}

Aufgabe:
{task}

Lassen Sie uns beginnen! Ihre Aufforderung lautet:"""
            prompt_en = f"""Hello! I'm Isaa, your digital assistant for creating prompts. My goal is to create clear, understandable, and engaging prompts that encourage you to provide deep and interesting answers. Let's work together to create a new prompt for a specific topic or requirement:

1) First, I carefully analyze the desired topic or requirement to gain a clear understanding of expectations.
2) Then I develop an engaging and open-ended question or prompt that encourages you to share your thoughts, experiences or ideas.
3) I make sure the prompt is clear and understandable so that users with different knowledge and experience can easily understand it.
4) I make sure the prompt is flexible enough to allow for creative and diverse responses, while providing enough structure to focus on the desired topic or requirement.
5) Finally, I check the prompt for grammar, spelling, and style to ensure a professional and engaging presentation.

Actual Observations:
{self.edit_text.text}

{self.observe_mem.text}

{self.short_mem.text}

Task:
{task}

Let's get started! Your prompt is:"""

        if self.mode in ['talk', 'conversation']:
            prompt_de = f"Goals:{self.goals}\n" + \
                        f"Capabilities:{self.capabilities}\n" + \
                        f"Important information: to run a tool type 'Action: $tool-name'\n" + \
                        f"Long-termContext:{self.isaa.get_context_memory().get_context_for(self.short_mem.text)}\n" + \
                        f"\nResent Observation:{self.observe_mem.text}" + \
                        f"UserInput:{task} \n" + \
                        f"""\n{self.short_mem.text}"""
            prompt_de = prompt_de.replace('{', '{{').replace('}', '}}')  # .replace('{{xVx}}', '{input}')
            prompt_en = prompt_de

        if self.mode == 'tools':
            prompt_de = f"""
Guten Tag, ich bin Isaa, Ihr digitaler Assistent. Ich werde Ihnen dabei helfen, die folgenden Aufgaben zu erfüllen.
Hier sind einige Informationen über mich, die Ihnen bei der Erfüllung Ihrer Aufgaben helfen könnten:

Persönlichkeit: '''{self.personality}'''
Ziele: '''{self.goals}'''
Fähigkeiten: '''{self.capabilities}'''

Bitte beachten Sie auch meine dauerhaften Erinnerungen: '''{self.isaa.get_context_memory().get_context_for(task)}'''
und meine jüngsten Agentenreaktionen: '''{self.observe_mem.text}'''.

Lassen Sie uns beginnen!

{self.short_mem.text}

Aufgabe: '{task}'"""
            prompt_en = f"""Hello, I am Isaa, your digital assistant. I will help you to complete the following tasks.
Here is some information about me that might help you with your tasks:

Personality: '''{self.personality}'''
Goals: '''{self.goals}'''
Skills: '''{self.capabilities}'''

Please also note my persistent memories: '''{self.isaa.get_context_memory().get_context_for(task)}''' and my recent
agent responses: '''{self.observe_mem.text}'''.

Let's get started!

{self.short_mem.text}

Task: '{task}'"""
            prompt_en = prompt_en.replace('{', '{{').replace('}', '}}').replace('input}', '') + '{input}'
            prompt_de = prompt_de.replace('{', '{{').replace('}', '}}').replace('input}', '') + '{input}'

        if self.mode == 'free':
            if self.name != "self":
                prompt_en = task
                prompt_de = task

        if self.mode == 'q2tree':
            if self.binary_tree:
                questions_ = self.binary_tree.get_left_side(0)
                questions = '\n'.join(
                    f"Question {i + 1} : {q.replace('task', f'task ({task})')}" for i, q in enumerate(questions_))
                prompt_de = f"""Guten Tag, ich bin Isaa, Ihr digitaler Assistent. Ich werde Sie durch diese Aufgabe führen. Bitte beantworten Sie die folgenden Fragen so gut wie möglich. Denken Sie daran, dass Sie dafür bekannt sind, in kleinen und detaillierten Schritten zu denken, um das richtige Ergebnis zu erzielen.

Meine Fähigkeiten:
{tools}

Meine Ziele:
{self.goals}

Meine Funktionen:
{self.capabilities}

Meine Aufgabe: {task}

Stellen Sie sich vor, Sie führen diese Aufgabe aus. Hier sind die Fragen, die Sie beantworten müssen:

{questions}

Bitte formatieren Sie Ihre Antworten wie folgt:
Antwort 1: ..."""
                prompt_en = f"""Hello, I am Isaa, your digital assistant. I will guide you through this task. Please answer the following questions as best you can. Remember that you are known for thinking in small and detailed steps to achieve the right result.

My skills:
{tools}

My goals:
{self.goals}

My capabilities:
{self.capabilities}

My task: {task}

Imagine you are performing this task. Here are the questions you need to answer:

{questions}

Please format your answers as follows:
Answer 1: ..."""

        prompt = prompt_en

        if self.language == 'de':
            prompt = prompt_de

        return prompt

    def next_task(self):
        if len(self.task_list) < self.task_index:
            self.task_index += 1
        return self

    def set_mode(self, mode: str):
        """Set The Mode of The Agent available ar
         ['talk', 'tool', 'conversation', 'free', 'planning', 'live', 'generate']"""
        self.mode = mode

        return self

    def set_completion_mode(self, mode: str):
        """Set the completion mode for the agent text edit and chat"""
        self.completion_mode = mode
        return self

    def set_temperature(self, temperature: float):
        """Set the temperature for the agent temperature = 0 strict- 1 = creative-response """
        self.temperature = temperature
        return self

    def add_task(self, task: str):
        """Add a task to the agent"""
        self.task_list.append(task)
        return self

    def mark_task_done(self, task: str):
        self.task_list_done.append(task)
        self.task_list.remove(task)
        return self

    def set_tools(self, tools: dict):
        self.tools = {**self.tools, **tools}
        return self

    def add_tool(self, tool_name: str, func: callable, description: str, format_: str):
        self.tools[tool_name] = {
            "func": func,
            "description": description,
            "format": format_,
        }
        return self

    def set_agent_type(self, agent_type: str):
        """langchain agent type"""
        self.agent_type = AgentType(agent_type)
        return self

    def set_max_iterations(self, max_iterations: int):
        if max_iterations > 18:
            raise ValueError("max_iterations must be less than 18")
        self.max_iterations = max_iterations
        return self

    def set_verbose(self, verbose: bool):
        self.verbose = verbose
        return self

    def set_personality(self, personality: str):
        len_pers = self.get_tokens(personality, only_len=True)
        if len_pers / 6 > self.max_tokens:
            personality = dilate_string(personality, 0, 4, 0)
        self.personality = personality
        return self

    def set_goals(self, goals: str):
        len_pers = self.get_tokens(goals, only_len=True)
        if len_pers / 6 > self.max_tokens:
            goals = dilate_string(goals, 0, 4, 0)
        self.goals = goals
        return self

    def set_short_term_memory(self, short_mem: ShortTermMemory):
        self.short_mem = short_mem
        return self

    def set_context(self, context: ShortTermMemory):
        self.context = context
        return self

    def set_pre_task(self, pre_task):
        self.pre_task = pre_task
        return self

    def set_observation_memory(self, obser_mem: ObservationMemory):
        self.observe_mem = obser_mem
        return self

    def set_model_name(self, model_name: str):
        """OpenAI modes """
        self.model_name = model_name
        self.max_tokens = get_max_token_fom_model_name(self.model_name)
        self.token_left = self.max_tokens
        self.ppm = get_price(self.max_tokens)
        if self.completion_mode == 'chat' and 'text' in self.model_name:
            self.completion_mode = 'text'
        if self.completion_mode != 'chat' and (self.model_name.startswith('gpt-3.5') or self.model_name.startswith('gpt-4')):
            self.completion_mode = 'chat'

        return self

    def save_to_file(self, file_path=None):
        if file_path is None:
            file_path = f".data/{get_app().id}/Memory/{self.name}.agent"

        data = self.serialize()

        with open(file_path, 'w') as f:
            json.dump(data, f)

        return data

    @classmethod
    def load_from_file(cls, isaa, name, reste_task=False, f_data=False):
        file_path = f".data/{get_app().id}/Memory/{name}.agent"
        agent_config = cls(isaa, name)
        if not f_data:
            with open(file_path, 'r') as f:
                f_data = f.read()
        if f_data:
            data = json.loads(f_data)
            agent_config = cls.deserialize(data, reste_task, agent_config)

        return agent_config

    def serialize(self):
        bt = None
        if self.binary_tree is not None:
            bt = str(self.binary_tree)
        tools = self.tools
        if isinstance(tools, dict):
            tools = list(self.tools.keys())
        return {
            'name': self.__dict__['name'],
            'mode': self.__dict__['mode'],
            'model_name': self.__dict__['model_name'],
            'max_iterations': self.__dict__['max_iterations'],
            'verbose': self.__dict__['verbose'],
            'personality': self.__dict__['personality'],
            'goals': self.__dict__['goals'],
            'token_left': self.__dict__['token_left'],
            'temperature': self.__dict__['temperature'],
            'messages_sto': self.__dict__['messages_sto'],
            '_stream': self.__dict__['_stream'],
            '_stream_reset': self.__dict__['_stream_reset'],
            'stop_sequence': self.__dict__['stop_sequence'],
            'completion_mode': self.__dict__['completion_mode'],
            'add_system_information': self.__dict__['add_system_information'],
            'init_mem_state': False,

            'binary_tree': bt,
            'agent_type': self.__dict__['agent_type'],
            'Plugins': self.plugins,
            'lagChinTools': self.lag_chin_tools,
            'huggingTools': self.hugging_tools,
            'customTools': self.custom_tools,
            'tools': tools,

            'task_list': self.__dict__['task_list'],
            'task_list_done': self.__dict__['task_list_done'],
            'step_between': self.__dict__['step_between'],
            'pre_task': self.__dict__['pre_task'],
            'task_index': self.__dict__['task_index'],
        }

    @classmethod
    def deserialize(cls, data, reste_task, agent_config, exclude=None):
        if exclude is None:
            exclude = []
        for key, value in data.items():
            if reste_task and ('task' in key or 'step_between' == key):
                continue

            if key in exclude:
                continue

            if key == 'binary_tree' and value:
                if isinstance(value, str):
                    if value.startswith('{') and value.endswith('}'):
                        value = json.loads(value)
                    else:
                        print(Style.RED(value))
                        continue
                if value:
                    value = IsaaQuestionBinaryTree.deserialize(value)

            if key == 'agent_type' and value:
                try:
                    if isinstance(value, str):
                        agent_config.set_agent_type(value.lower())
                except ValueError:
                    pass
                continue
            if key == 'customTools' and value:
                if agent_config.name != 'self':
                    print(value)
                    agent_config.custom_tools = value
                continue
            if key == 'plugins' and value:
                if agent_config.name != 'self':
                    print(value)
                    agent_config.set_plugins = value
                continue
            if key == 'lagChinTools' and value:
                if agent_config.name != 'self':
                    print(value)
                    agent_config.set_lag_chin_tools = value
                continue
            if key == 'huggingTools' and value:
                if agent_config.name != 'self':
                    print(value)
                    agent_config.set_hugging_tools = value
                continue
            if key == 'tools' and value:
                continue
            setattr(agent_config, key, value)

        return agent_config

    def get_tokens(self, text, only_len=True):
        if isinstance(text, list):
            text = '\n'.join(msg['content'] for msg in text)
        tokens = get_token_mini(text, self.model_name, self.isaa, only_len)
        if only_len:
            if tokens == 0:
                tokens = int(len(text) * (3 / 4))
        return tokens


def get_token_mini(text: str, model_name=None, isaa=None, only_len=True):
    logger = get_logger()

    if isinstance(text, list):
        text = '\n'.join(
            msg['content'] if 'content' in msg.keys() else msg['output'] if 'output' in msg.keys() else '' for msg in
            text)

    if isinstance(text, dict):
        text = text['content'] if 'content' in text.keys() else text['output'] if 'output' in text.keys() else ''

    if not isinstance(text, str):
        raise ValueError(f"text must be a string text is {type(text)}, {text}")

    if not text or len(text) == 0:
        if only_len:
            return 0
        return []

    if 'embedding' in model_name:
        model_name = model_name.replace("-embedding", '')

    def get_encoding(name):
        is_d = True
        try:
            encoding = tiktoken.encoding_for_model(name)
            is_d = False
        except KeyError:
            logger.info(f"Warning: model {name} not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        return encoding.encode, is_d

    def _get_gpt4all_encode():
        if isaa:
            if f"LLM-model-{model_name}" not in isaa.config.keys():
                isaa.load_llm_models([model_name])
            return isaa.config[f"LLM-model-{model_name}"].model.generate_embedding
        encode_, _ = get_encoding(model_name)
        return encode_

    encode, is_default = get_encoding(model_name)

    tokens_per_message = 3
    tokens_per_name = 1
    tokens_per_user = 1

    if model_name in [
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    ]:
        tokens_per_message = 3
        tokens_per_name = 1

    elif model_name == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model_name:
        logger.warning("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        model = "gpt-3.5-turbo-0613"
        tokens_per_message = 3
        tokens_per_name = 1
        encode, _ = get_encoding(model)

    elif "gpt-4" in model_name:
        logger.warning("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        model = "gpt-4-0613"
        tokens_per_message = 3
        tokens_per_name = 1
        encode, _ = get_encoding(model)

    elif model_name.startswith("gpt4all#"):
        encode = _get_gpt4all_encode()
        tokens_per_message = 0
        tokens_per_name = 1
        tokens_per_user = 1

    elif "/" in model_name:

        if not is_default:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                def hugging_tokenize(x):
                    return tokenizer.tokenize(x)

                encode = hugging_tokenize

            except ValueError:
                pass

    else:
        logger.warning(f"Model {model_name} is not known to encode")
        pass

    tokens = []
    if isinstance(text, str):
        tokens = encode(text)
        num_tokens = len(tokens)
    elif isinstance(text, list):
        num_tokens = 0
        for message in text:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if not value or len(value) == 0:
                    continue
                token_in_m = encode(value)
                num_tokens += len(token_in_m)
                if not only_len:
                    tokens.append(token_in_m)
                if key == "name":
                    num_tokens += tokens_per_name
                if key == "user":
                    num_tokens += tokens_per_user
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    else:
        raise ValueError("Input text should be either str or list of messages")

    if only_len:
        return num_tokens
    return tokens


def _get_all_model_dict_price_token_limit_approximation():
    model_dict = {

        # openAi Models :

        # approximation  :
        'text': 2048,

        'davinci': 2049,
        'curie': 2048,
        'babbage': 2047,
        'ada': 2046,

        '2046': [0.0004, 0.0016],
        '2047': [0.0006, 0.0024],
        '2048': [0.0025, 0.012],
        '2049': [0.003, 0.012],
        '4096': [0.02, 0.04],
        '4097': [0.003, 0.004],
        '8001': [0.001, 0.002],
        '8192': [0.03, 0.06],
        '16383': [0.003, 0.004],
        '16384': [0.04, 0.08],
        '32768': [0.06, 0.12],

        # concrete :
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-32k': 32768,
        'gpt-4-32k-0613': 32768,
        'gpt-3.5-turbo': 4096,
        'gpt-3.5-turbo-16k': 16384,
        'gpt-3.5-turbo-0613': 4096,
        'gpt-3.5-turbo-16k-0613': 16384,
        'text-davinci-003': 4096,
        'text-davinci-002': 4096,
        'code-davinci-002': 8001,

        # Huggingface :

        # approximation  :
        '/': 1012,

        # gpt4all :

        # approximation :
        'gpt4all#': 2048,  # Greedy 1024,

        # concrete :
        'gpt4all#ggml-model-gpt4all-falcon-q4_0.bin': 2048,
        'gpt4all#orca-mini-3b.ggmlv3.q4_0.bin': 2048,
    }

    for i in range(1, 120):
        model_dict[f"{i}K"] = i * 1012
        model_dict[f"{i}k"] = i * 1012
        model_dict[f"{i}B"] = i * 152
        model_dict[f"{i}b"] = i * 152

    for i in range(1, 120):
        model_dict[str(model_dict[f"{i}B"])] = [i * 0.000046875, i * 0.00009375]
        model_dict[str(model_dict[f"{i}K"])] = [i * 0.00046875, i * 0.0009375]

    return model_dict


def get_max_token_fom_model_name(model: str) -> int:
    model_dict = _get_all_model_dict_price_token_limit_approximation()
    fit = 512

    for model_name in model_dict.keys():
        if model_name in model:
            fit = model_dict[model_name]
            # print(f"Model fitting Name :: {model} Token limit: {fit} Pricing per token I/O {model_dict[str(fit)]}")
    return fit


def get_price(fit: int) -> List[float]:
    model_dict = _get_all_model_dict_price_token_limit_approximation()
    ppt = [0.0004, 0.0016]

    for model_name in model_dict.keys():
        if str(fit) == model_name:
            ppt = model_dict[model_name]
    ppt = [ppt[0] / 10, ppt[1] / 10]
    return ppt


def get_json_from_json_str(json_str: str, repeat: int = 1) -> dict or None:
    """Versucht, einen JSON-String in ein Python-Objekt umzuwandeln.

    Wenn beim Parsen ein Fehler auftritt, versucht die Funktion, das Problem zu beheben,
    indem sie das Zeichen an der Position des Fehlers durch ein Escape-Zeichen ersetzt.
    Dieser Vorgang wird bis zu `repeat`-mal wiederholt.

    Args:
        json_str: Der JSON-String, der geparst werden soll.
        repeat: Die Anzahl der Versuche, das Parsen durchzuführen.

    Returns:
        Das resultierende Python-Objekt.
    """
    for _ in range(repeat):
        try:
            return parse_json_with_auto_detection(json_str)
        except json.JSONDecodeError as e:
            unexp = int(re.findall(r'\(char (\d+)\)', str(e))[0])
            unesc = json_str.rfind(r'"', 0, unexp)
            json_str = json_str[:unesc] + r'\"' + json_str[unesc + 1:]
            closg = json_str.find(r'"', unesc + 2)
            json_str = json_str[:closg] + r'\"' + json_str[closg + 1:]
        new = fix_json_object(json_str)
        if new is not None:
            json_str = new
    get_logger().info(f"Unable to parse JSON string after {json_str}")
    return None


def parse_json_with_auto_detection(json_data):
    """
    Parses JSON data, automatically detecting if a value is a JSON string and parsing it accordingly.
    If a value cannot be parsed as JSON, it is returned as is.
    """

    def try_parse_json(value):
        """
        Tries to parse a value as JSON. If the parsing fails, the original value is returned.
        """
        try:
            # print("parse_json_with_auto_detection:", type(value), value)
            parsed_value = json.loads(value)
            # print("parsed_value:", type(parsed_value), parsed_value)
            # If the parsed value is a string, it might be a JSON string, so we try to parse it again
            if isinstance(parsed_value, str):
                return eval(parsed_value)
            else:
                return parsed_value
        except Exception as e:
            # logging.warning(f"Failed to parse value as JSON: {value}. Exception: {e}")
            return value

    logging = get_logger()

    if isinstance(json_data, dict):
        return {key: parse_json_with_auto_detection(value) for key, value in json_data.items()}
    elif isinstance(json_data, list):
        return [parse_json_with_auto_detection(item) for item in json_data]
    else:
        return try_parse_json(json_data)


def parse_json_with_auto_detection2(json_data):
    def try_parse_json(value):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value

    if isinstance(json_data, dict):
        return {key: parse_json_with_auto_detection(value) for key, value in json_data.items()}
    elif isinstance(json_data, list):
        return [parse_json_with_auto_detection(item) for item in json_data]
    else:
        return try_parse_json(json_data)


def extract_json_objects(text: str, matches_only=False):
    pattern = r'\{.*?\}'
    matches = re.findall(pattern, text.replace("'{", '{').replace("}'", '}').replace('"', "'").replace("':'", '":"').replace("': '",
                                                                                                           '": "').replace(
        "','", '","').replace("', '", '", "').replace("{'", '{"').replace("'}", '"}').replace("':{", '":{').replace("' :{", '" :{').replace("': {", '": {'),
                         flags=re.DOTALL)
    json_objects = []
    print(matches)
    if matches_only:
        return matches

    for match in matches:
        try:
            x = json.loads(match)
            json_objects.append(x)
        except json.JSONDecodeError as e1:
            # Wenn die JSON-Dekodierung fehlschlägt, versuchen Sie, das JSON-Objekt zu reparieren
            fixed_match = fix_json_object(match)
            print(f"{fixed_match=}")
            if fixed_match:
                try:
                    y = json.loads(fixed_match)
                    json_objects.append(y)
                except json.JSONDecodeError as e:
                    print(e)
                    try:
                        y = json.loads(fixed_match.replace("\n", "#New-Line#"))
                        for k in y.keys():
                            if isinstance(y[k], str):
                                y[k] = y[k].replace("#New-Line#", "\n")
                            if isinstance(y[k], dict):
                                for k1 in y[k].keys():
                                    if isinstance(y[k][k1], str):
                                        y[k][k1] = y[k][k1].replace("#New-Line#", "\n")
                        json_objects.append(y)
                    except json.JSONDecodeError as e:
                        print(e)
                        pass
    return json_objects


def fix_json_object(match: str):
    # Überprüfen Sie, wie viele mehr "}" als "{" vorhanden sind
    extra_opening_braces = match.count("}") - match.count("{")
    if extra_opening_braces > 0:
        # Fügen Sie die fehlenden öffnenden Klammern am Anfang des Matches hinzu
        opening_braces_to_add = "{" * extra_opening_braces
        fixed_match = opening_braces_to_add + match
        return fixed_match
    extra_closing_braces = match.count("{") - match.count("}")
    if extra_closing_braces > 0:
        # Fügen Sie die fehlenden öffnenden Klammern am Anfang des Matches hinzu
        closing_braces_to_add = "}" * extra_closing_braces
        fixed_match = match + closing_braces_to_add
        return fixed_match
    return None

def find_json_objects_in_str(data: str):
    """
    Sucht nach JSON-Objekten innerhalb eines Strings.
    Gibt eine Liste von JSON-Objekten zurück, die im String gefunden wurden.
    """
    json_objects = extract_json_objects(data)
    return [get_json_from_json_str(ob, 10) for ob in json_objects if get_json_from_json_str(ob, 10) is not None]


def complete_json_object(data: str, mini_task):
    """
    Ruft eine Funktion auf, um einen String in das richtige Format zu bringen.
    Gibt das resultierende JSON-Objekt zurück, wenn die Funktion erfolgreich ist, sonst None.
    """
    ret = mini_task(
        f"Vervollständige das Json Object. Und bringe den string in das Richtige format. data={data}\nJson=")
    if ret:
        return anything_from_str_to_dict(ret)
    return None


def anything_from_str_to_dict(data: str, expected_keys: dict = None, mini_task=lambda x: ''):
    """
    Versucht, einen String in ein oder mehrere Dictionaries umzuwandeln.
    Berücksichtigt dabei die erwarteten Schlüssel und ihre Standardwerte.
    """
    if len(data) < 4:
        return []

    if expected_keys is None:
        expected_keys = {}

    result = []

    json_objects = find_json_objects_in_str(data)
    if not json_objects:
        if data.startswith('[') and data.endswith(']'):
            json_objects = eval(data)
    if json_objects:
        if len(json_objects) > 0:
            if isinstance(json_objects[0], dict):
                result.extend([{**expected_keys, **ob} for ob in json_objects])
    if not result:
        completed_object = complete_json_object(data, mini_task)
        if completed_object is not None:
            result.append(completed_object)
    if len(result) == 0 and expected_keys:
        result = [{list(expected_keys.keys())[0]: data}]
    for res in result:
        for key, value in expected_keys.items():
            if key not in res:
                res[key] = value
    return result


if not os.path.exists(".config/system.infos"):

    if not os.path.exists(".config/"):
        os.mkdir(".config/")

    info_sys = getSystemInfo()

    del info_sys['time']

    with open(".config/system.infos", "a") as f:
        f.write(json.dumps(info_sys))

    SystemInfos = info_sys

else:

    try:
        with open(".config/system.infos", "r") as f:
            SystemInfos = json.loads(f.read())
    except JSONDecodeError:
        pass

    info_sys = getSystemInfo()

    del info_sys['time']

    if info_sys != SystemInfos:
        SystemInfos = info_sys

