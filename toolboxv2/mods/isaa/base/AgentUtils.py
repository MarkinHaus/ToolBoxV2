import json
import os
import pickle
import platform
import re
import socket
import subprocess
import threading
from datetime import datetime
from json import JSONDecodeError

import requests
import tiktoken
from pebble import concurrent

from toolboxv2 import Singleton, Style, get_logger
from toolboxv2.mods.isaa.base.KnowledgeBase import KnowledgeBase


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
    elif isinstance(split_param, str):
        substrings = text.split(split_param)
    else:
        raise ValueError
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


def initialize_system_infos(info_sys):
    global SystemInfos
    if not os.path.exists(".config/system.infos"):

        if not os.path.exists(".config/"):
            os.mkdir(".config/")

        del info_sys['time']

        with open(".config/system.infos", "a", encoding="utf8") as f:
            f.write(json.dumps(info_sys))

        SystemInfos = info_sys

    else:

        try:
            with open(".config/system.infos", encoding="utf8") as f:
                SystemInfos = json.loads(f.read())
        except JSONDecodeError:
            pass

        del info_sys['time']

        if info_sys != SystemInfos:
            SystemInfos = info_sys


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
            "last_context": last_context}

    try:
        process = get_location()
        info['location'], info['ip'] = process.result()
    except TimeoutError and Exception:
        info['location'] = "Berlin Schöneberg"
    initialize_system_infos(info)
    return info


class Scripts:
    def __init__(self, filename):
        self.scripts = {}
        self.filename = filename

    def create_script(self, name, description, content, script_type="py"):
        self.scripts[name] = {"description": description, "content": content, "type": script_type}

    def remove_script(self, name):
        del self.scripts[name]

    def run_script(self, name):
        if name not in self.scripts:
            return "Script not found!"
        script = self.scripts[name]
        with open(f"{name}.{script['type']}", "w") as f:
            f.write(script["content"])
        if script["type"] == "py":
            result = subprocess.run(["python", f"{name}.py"], capture_output=True, text=True, encoding='cp850')
        elif script["type"] == "sh":
            result = subprocess.run(["bash", f"{name}.sh"], capture_output=True, text=True, encoding='cp850')
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
                node = node and node.left if direction == 'L' else node and node.right
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
    def format_name(name: str) -> str:
        if name is None:
            return ''
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
        if name in self.chains:
            self.chains_dis[name + '-dis'] = dis

    def get_discr(self, name):
        name = self.format_name(name)
        if name + '-dis' in self.chains_dis:
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
        print("--------------------------------")
        for file in files:
            file_path = os.path.join(self.directory, file)

            if not file.endswith(".json"):
                continue
            try:
                with open(file_path, encoding='utf-8') as f:
                    dat = f.read()
                    chain_data = json.loads(dat)
                chain_name = os.path.splitext(file)[0]
                print(f"Loading : {chain_name}")
                self.add(chain_name, chain_data["tasks"])
                if 'dis' in chain_data:
                    self.add_discr(chain_name, chain_data['dis'])
            except Exception as e:
                print(Style.RED(f"Beim Laden der Datei '{file_path}' ist ein Fehler aufgetreten: {e}"))
        if "toolRunner" not in self.chains:
            print("loading default chain toolRunner")
            self.add("toolRunner", [
                {
                    "use": "agent",
                    "name": "tools",
                    "args": "$user-input",
                    "return": "$return",

                    "running_mode": "invoke"
                }
            ])
        if "toolRunnerMission" not in self.chains:
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
                    "return": "$return",
                    "running_mode": "lineIs"
                }
            ])
        if "liveRunner" not in self.chains:
            print("loading default chain liveRunner")
            self.add("liveRunner", [
                {
                    "use": "agent",
                    "name": "liveInterpretation",
                    "args": "$user-input",
                    "return": "$return"
                }
            ])
        if "SelfRunner" not in self.chains:
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
        if "liveRunnerMission" not in self.chains:
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
            if 'dis' in chain:
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
        if chains_to_save:
            print("--------------------------------", end='\r')
        for name, tasks in chains_to_save.items():
            file_path = os.path.join(self.directory, f"{name}.json")
            chain_data = {"name": name, "tasks": tasks, "dis": self.get_discr(name)}

            try:
                with open(file_path, "w", encoding='utf-8') as f:
                    print(f"Saving : {name}")
                    json.dump(chain_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Beim Speichern der Datei '{file_path}' ist ein Fehler aufgetreten: {e}")
        if len(self.chains.keys()):
            print(
                f"================================\nChainsSaved : {len(self.chains.keys())}\n================================\n")

    def __str__(self):
        return str(self.chains.keys())



class AISemanticMemory(metaclass=Singleton):
    def __init__(self,
                 base_path: str = "/semantic_memory",
                 default_model: str = os.getenv("DEFAULTMODELSUMMERY"),
                 default_embedding_model: str = os.getenv("DEFAULTMODELEMBEDDING"),
                 default_similarity_threshold: float = 0.61,
                 default_batch_size: int = 64,
                 default_n_clusters: int = 2,
                 default_deduplication_threshold: float = 0.85):
        """
        Initialize AISemanticMemory with KnowledgeBase integration

        Args:
            base_path: Root directory for memory storage
            default_model: Default model for text generation
            default_embedding_model: Default embedding model
            default_similarity_threshold: Default similarity threshold for retrieval
            default_batch_size: Default batch size for processing
            default_n_clusters: Default number of clusters for FAISS
            default_deduplication_threshold: Default threshold for deduplication
        """
        self.base_path = os.path.join(os.getcwd(), ".data", base_path)
        self.memories: dict[str, KnowledgeBase] = {}

        # Map of embedding models to their dimensions
        self.embedding_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "nomic-embed-text": 768,
            "default": 768
        }

        self.default_config = {
            "embedding_model": default_embedding_model,
            "embedding_dim": self._get_embedding_dim(default_embedding_model),
            "similarity_threshold": default_similarity_threshold,
            "batch_size": default_batch_size,
            "n_clusters": default_n_clusters,
            "deduplication_threshold": default_deduplication_threshold,
            "model_name": default_model
        }

    def _get_embedding_dim(self, model_name: str) -> int:
        """Get embedding dimension for a model"""
        return self.embedding_dims.get(model_name, 768)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize memory name for filesystem safety"""
        name = re.sub(r'[^a-zA-Z0-9_-]', '-', name)[:63].strip('-')
        if not name:
            raise ValueError("Invalid memory name")
        if len(name) < 3:
            name += "Z" * (3 - len(name))
        return name

    def create_memory(self,
                      name: str,
                      model_config: dict | None = None,
                      storage_config: dict | None = None) -> KnowledgeBase:
        """
        Create new memory store with KnowledgeBase

        Args:
            name: Unique name for the memory store
            model_config: Configuration for embedding model
            storage_config: Configuration for KnowledgeBase parameters
        """
        sanitized = self._sanitize_name(name)
        if sanitized in self.memories:
            raise ValueError(f"Memory '{name}' already exists")

        # Determine embedding model and dimension
        embedding_model = self.default_config["embedding_model"]
        model_name = self.default_config["model_name"]
        if model_config:
            embedding_model = model_config.get("embedding_model", embedding_model)
            model_name = model_config.get("model_name", model_name)
        embedding_dim = self._get_embedding_dim(embedding_model)

        # Get KnowledgeBase parameters
        kb_params = {
            "embedding_dim": embedding_dim,
            "embedding_model": embedding_model,
            "similarity_threshold": self.default_config["similarity_threshold"],
            "batch_size": self.default_config["batch_size"],
            "n_clusters": self.default_config["n_clusters"],
            "deduplication_threshold": self.default_config["deduplication_threshold"],
            "model_name": model_name,
        }

        if storage_config:
            kb_params.update({
                "similarity_threshold": storage_config.get("similarity_threshold", kb_params["similarity_threshold"]),
                "batch_size": storage_config.get("batch_size", kb_params["batch_size"]),
                "n_clusters": storage_config.get("n_clusters", kb_params["n_clusters"]),
                "model_name": storage_config.get("model_name", kb_params["model_name"]),
                "embedding_model": storage_config.get("embedding_model", kb_params["embedding_model"]),
                "deduplication_threshold": storage_config.get("deduplication_threshold",
                                                              kb_params["deduplication_threshold"]),
            })

        # Create KnowledgeBase instance
        self.memories[sanitized] = KnowledgeBase(**kb_params)
        return self.memories[sanitized]

    async def add_data(self,
                       memory_name: str,
                       data: str | list[str] | bytes | dict,
                       metadata: dict | None = None) -> bool:
        """
        Add data to memory store

        Args:
            memory_name: Target memory store
            data: Text, list of texts, binary file, or structured data
            metadata: Optional metadata
        """
        name = self._sanitize_name(memory_name)
        kb = self.memories.get(name)
        if not kb:
            kb = self.create_memory(name)

        # Process input data
        texts = []
        if isinstance(data, bytes):
            try:
                import textract
                text = textract.process(data).decode('utf-8')
                texts = [text.replace('\\t', '').replace('\t', '')]
            except Exception as e:
                raise ValueError(f"File processing failed: {str(e)}")
        elif isinstance(data, str):
            texts = [data.replace('\\t', '').replace('\t', '')]
        elif isinstance(data, list):
            texts = [d.replace('\\t', '').replace('\t', '') for d in data]
        elif isinstance(data, dict):
            # Custom KG not supported in current KnowledgeBase
            raise NotImplementedError("Custom knowledge graph insertion not supported")
        else:
            raise ValueError("Unsupported data type")

        # Add data to KnowledgeBase
        try:
            added, duplicates = await kb.add_data(texts, metadata)
            return added > 0
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise RuntimeError(f"Data addition failed: {str(e)}")

    def get(self, names):
        return [m for n,m in self._get_target_memories(names)]

    async def query(self,
                    query: str,
                    memory_names: str | list[str] | None = None,
                    query_params: dict | None = None,
                    to_str: bool = False,
                    unified_retrieve: bool =False) -> str | list[dict]:
        """
        Query memories using KnowledgeBase retrieval

        Args:
            query: Search query
            memory_names: Target memory names
            query_params: Query parameters
            to_str: Return string format
            unified_retrieve: Unified retrieve
        """
        targets = self._get_target_memories(memory_names)
        if not targets:
            return []

        results = []
        for name, kb in targets:
            #try:
                # Use KnowledgeBase's retrieve_with_overview for comprehensive results
                result = await kb.retrieve_with_overview(
                    query=query,
                    k=query_params.get("k", 3) if query_params else 3,
                    min_similarity=query_params.get("min_similarity", 0.2) if query_params else 0.2,
                    cross_ref_depth=query_params.get("cross_ref_depth", 2) if query_params else 2,
                    max_cross_refs=query_params.get("max_cross_refs", 2) if query_params else 2,
                    max_sentences=query_params.get("max_sentences", 5) if query_params else 5
                ) if not unified_retrieve else await kb.unified_retrieve(
                    query=query,
                    k=query_params.get("k", 2) if query_params else 2,
                    min_similarity=query_params.get("min_similarity", 0.2) if query_params else 0.2,
                    cross_ref_depth=query_params.get("cross_ref_depth", 2) if query_params else 2,
                    max_cross_refs=query_params.get("max_cross_refs", 6) if query_params else 6,
                    max_sentences=query_params.get("max_sentences", 12) if query_params else 12
                )
                results.append({
                    "memory": name,
                    "result": result
                })
            #except Exception as e:
            #    print(f"Query failed on {name}: {str(e)}")
        if to_str:
            if not unified_retrieve:
                str_res = [
                    f"{x['memory']} - {json.dumps(x['result'].overview)}\n - {[c.text for c in x['result'].details]}\n - {[(k, [c.text for c in v]) for k, v in x['result'].cross_references.items()]}"
                    for x in results]
                # str_res =
            else:
                str_res = json.dumps(results)
            return str_res
        return results

    def _get_target_memories(self, memory_names: str | list[str] | None) -> list[tuple[str, KnowledgeBase]]:
        """Get target memories for query"""
        if not memory_names:
            return list(self.memories.items())

        names = [memory_names] if isinstance(memory_names, str) else memory_names

        targets = []
        for name in names:
            sanitized = self._sanitize_name(name)
            if kb := self.memories.get(sanitized):
                targets.append((sanitized, kb))
        return targets

    def list_memories(self) -> list[str]:
        """List all available memories"""
        return list(self.memories.keys())

    async def delete_memory(self, name: str) -> bool:
        """Delete a memory store"""
        sanitized = self._sanitize_name(name)
        if sanitized in self.memories:
            del self.memories[sanitized]
            return True
        return False

    def save_memory(self, name: str, path: str) -> bool | bytes:
        """Save a memory store to disk"""
        sanitized = self._sanitize_name(name)
        if kb := self.memories.get(sanitized):
            try:
                return kb.save(path)
            except Exception as e:
                print(f"Error saving memory: {str(e)}")
                return False
        return False

    def load_memory(self, name: str, path: str | bytes) -> bool:
        """Load a memory store from disk"""
        sanitized = self._sanitize_name(name)
        if sanitized in self.memories:
            return False
        try:
            self.memories[sanitized] = KnowledgeBase.load(path)
            return True
        except Exception:
            # print(f"Error loading memory: {str(e)}")
            return False


"""```

## Complete Documentation Additions

### LiteLLM Integration Guide

```markdown
## Supported LLM Providers

AISemanticMemory
supports
100 + LLMs
via
LiteLLM:

```python
# Anthropic
memory.create_memory("legal_docs", model_config={
    "llm_model": "claude-3-sonnet-20240229",
    "llm_params": {"temperature": 0.2}
})

# Cohere
memory.create_memory("support_chat", model_config={
    "llm_model": "command-r-plus",
    "embedding_model": "embed-english-v3.0"
})

# Local Models
memory.create_memory("internal_data", model_config={
    "llm_model": "ollama/llama3",
    "llm_params": {"base_url": "http://localhost:11434"}
})
```

### Advanced Query Features

** Multi - Memory
Consensus
Search **
```python
result = memory.query(
    "What are security best practices for cloud storage?",
    memory_names=["aws_docs", "azure_guides", "gcp_whitepapers"],
    query_params=QueryParam(
        mode="mix",
        top_k=50,
        conversation_history=chat_history
    ),
    consensus_threshold=0.75
)
```

** Temporal
Filtering **
```python
# Query documents from last 30 days
result = memory.query(
    "Recent API changes",
    query_params=QueryParam(
        mode="hybrid",
        filters={"date": {"$gte": "2024-06-01"}}
    )
)
```


###"""


class ShortTermMemory:
    memory_data: list[dict] = []
    max_length: int = 2000

    add_to_static: list[dict] = []

    lines_ = []

    isaa = None

    def __init__(self, isaa, name):
        self.name = name
        self.isaa = isaa
        self.tokens: int = 0
        self.lock = threading.Lock()
        if self.isaa is None:
            raise ValueError("Define Isaa Tool first ShortTermMemory")

    def set_name(self, name: str):
        self.name = name

    def info(self):
        text = self.text
        return f"\n{self.tokens=}\n{self.max_length=}\n{text[:60]=}\n"

    def cut(self):
        threading.Thread(target=self.cut_runner, daemon=True).start()

    def cut_runner(self):
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

            #for line in CharacterTextSplitter(chunk_size=max(300, int(len(data) / 10)),
            #                                  chunk_overlap=max(20, int(len(data) / 200))).split_text(data):
            #    if line not in self.lines_ and len(line) != 0:
        ntok = int(len(data) / 4.56)  #get_token_mini(data, self.model_name, self.isaa)
        self.memory_data.append({'data': data, 'token-count': ntok, 'vector': []})
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

    def run_and_display(self, python_code):
        """function to eval python code"""
        start = f'Start-state:\n{self.get_env()}'
        result = self.eval_code(python_code)
        end = f'End-state:\n{self.get_env()}'
        return f'{start}\nResult:\n{result}\n{end}'

    def tool(self):
        return {"PythonEval": {"func": self.run_and_display, "description": "Use Python Code to Get to an Persis Answer! input must be valid python code all non code parts must be comments!"}}

def get_str_response(chunk):
    # print("Got response :: get_str_response", chunk)
    if isinstance(chunk, list):
        if len(chunk) == 0:
            chunk = ""
        if len(chunk) > 1:
            return '\n'.join([get_str_response(c) for c in chunk])
        if len(chunk) == 1:
            chunk = chunk[0]
    if isinstance(chunk, dict):
        data = chunk['choices'][0]

        if "delta" in data:
            message = chunk['choices'][0]['delta']
            if isinstance(message, dict):
                message = message['content']
        elif "text" in data:
            message = chunk['choices'][0]['text']
        elif "message" in data:
            message = chunk['choices'][0]['message']['content']
        elif "content" in data['delta']:
            message = chunk['choices'][0]['delta']['content']
        else:
            message = ""

    elif isinstance(chunk, str):
        message = chunk
    else:
        try:
            if hasattr(chunk.choices[0], 'message'):
                message = chunk.choices[0].message.content
            elif hasattr(chunk.choices[0], 'delta'):
                message = chunk.choices[0].delta.content
                if message is None:
                    message = ''
            else:
                raise AttributeError
        except AttributeError:
            message = f"Unknown chunk type {chunk}{type(chunk)}"
    if message is None:
        message = f"Unknown message None : {type(chunk)}|{chunk}"
    return message


def get_token_mini(text: str, model_name=None, isaa=None, only_len=True):
    logger = get_logger()

    if model_name is None:
        model_name = ""

    if isinstance(text, list):
        text = '\n'.join(
            str(msg['content']) if 'content' in msg else str(msg['output']) if 'output' in msg else '' for msg in
            text)

    if isinstance(text, dict):
        text = str(text['content']) if 'content' in text else str(text['output']) if 'output' in text else ''

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
            if f"LLM-model-{model_name}" not in isaa.config:
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

                from transformers import AutoTokenizer
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
    model_dict = {}

    model_dict_p ={

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

        '200004': [3 / 1_000_000, 15 / 1_000_000],
        '200002': [15 / 1_000_000, 75 / 1_000_000],
        '200001': [1 / 1_000_000, 5 / 1_000_000],
        '199999': [0.25 / 1_000_000, 1.25 / 1_000_000],

        '128000': [0.25 / 1_000_000, 1.25 / 1_000_000],
        '32000': [0.15 / 1_000_000, 0.55 / 1_000_000],

        # concrete :
        'o3-mini': 200000,
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

        # gpt4all :

        # approximation :
        'gpt4all#': 2048,  # Greedy 1024,

        # concrete :
        'gpt4all#ggml-model-gpt4all-falcon-q4_0.bin': 2048,
        'gpt4all#orca-mini-3b.ggmlv3.q4_0.bin': 2048,

        # Claude
        '3-5-sonnet': 200003,
        '3-opus': 200004,
        '3-5-haiku': 200001,
        '3-haiku': 199999,

        # Googl
        'gemini': 1000000,
        'gemma': 128000,

        'llama-3.1': 128000,
        'mixtral-8x7b': 32000,
        'gemma2': 8192
    }

    for i in range(1, 120):
        model_dict[f"{i}K"] = i * 1012
        model_dict[f"{i}k"] = i * 1012
        model_dict[f"{i}B"] = i * 152
        model_dict[f"{i}b"] = i * 152

    for i in range(1, 120):
        model_dict[str(model_dict[f"{i}B"])] = [i * 0.000046875, i * 0.00009375]
        model_dict[str(model_dict[f"{i}K"])] = [i * 0.00046875, i * 0.0009375]

    return {**model_dict_p, **model_dict}


def get_max_token_fom_model_name(model: str) -> int:
    model_dict = _get_all_model_dict_price_token_limit_approximation()
    fit = 16000

    for model_name in model_dict:
        if model_name in model:
            fit = model_dict[model_name]
            break
            # print(f"Model fitting Name :: {model} Token limit: {fit} Pricing per token I/O {model_dict[str(fit)]}")
    if isinstance(fit, list):
        fit = 10000
    return fit


def get_price(fit: int) -> list[float]:
    model_dict = _get_all_model_dict_price_token_limit_approximation()
    ppt = [0.0004, 0.0016]

    for model_name in model_dict:
        if str(fit) == model_name:
            ppt = model_dict[model_name]
    ppt = [ppt[0] / 10, ppt[1] / 10]
    return ppt


def get_json_from_json_str(json_str: str or list or dict, repeat: int = 1) -> dict or None:
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
        except Exception:
            # logging.warning(f"Failed to parse value as JSON: {value}. Exception: {e}")
            return value

    get_logger()

    if isinstance(json_data, dict):
        return {key: parse_json_with_auto_detection(value) for key, value in json_data.items()}
    elif isinstance(json_data, list):
        return [parse_json_with_auto_detection(item) for item in json_data]
    else:
        return try_parse_json(json_data)


def extract_json_objects(text: str, matches_only=False):
    pattern = r'\{.*?\}'
    matches = re.findall(pattern,
                         text
                         .replace("'{", '{')
                         .replace("}'", '}')
                         .replace('"', "'")
                         .replace("':'", '":"')
                         .replace("': '", '": "')
                         .replace("','", '","')
                         .replace("', '", '", "')
                         .replace("{'", '{"')
                         .replace("'}", '"}')
                         .replace("':{", '":{')
                         .replace("' :{", '" :{')
                         .replace("': {", '": {')
                         ,
                         flags=re.DOTALL)
    json_objects = []
    if matches_only:
        return matches

    for match in matches:
        try:
            x = json.loads(match)
            print("Found", x)
            json_objects.append(x)
        except json.JSONDecodeError:
            # Wenn die JSON-Dekodierung fehlschlägt, versuchen Sie, das JSON-Objekt zu reparieren
            fixed_match = fix_json_object(match)
            if fixed_match:
                try:
                    y = json.loads(fixed_match)
                    json_objects.append(y)
                except json.JSONDecodeError as e:
                    print(e)
                    try:
                        y = json.loads(fixed_match.replace("\n", "#New-Line#"))
                        for k in y:
                            if isinstance(y[k], str):
                                y[k] = y[k].replace("#New-Line#", "\n")
                            if isinstance(y[k], dict):
                                for k1 in y[k]:
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
    if not isinstance(json_objects, list):
        json_objects = [json_objects]
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


def fix_json(json_str, current_index=0, max_index=10):
    if current_index > max_index:
        return json_str
    try:
        return json.loads(json_str)  # Wenn der JSON-String bereits gültig ist, gib ihn unverändert zurück
    except json.JSONDecodeError as e:
        error_message = str(e)
        # print("Error message:", error_message)

        # Handle specific error cases
        if "Expecting property name enclosed in double quotes" in error_message:
            # Korrigiere einfache Anführungszeichen in doppelte Anführungszeichen
            json_str = json_str.replace("'", '"')

        elif "Expecting ':' delimiter" in error_message:
            # Setze fehlende Werte auf null
            json_str = json_str.replace(':,', ':null,')

        elif "Expecting '" in error_message and "' delimiter:" in error_message:
            # Setze fehlende Werte auf null
            line_i = int(error_message[error_message.rfind('line') + 4:error_message.rfind('column')].strip())
            colom_i = int(error_message[error_message.rfind('char') + 4:-1].strip())
            sp = error_message.split("'")[1]

            json_lines = json_str.split('\n')
            corrected_json_lines = json_lines[:line_i - 1]  # Bis zur Zeile des Fehlers
            faulty_line = json_lines[line_i - 1]  # Die Zeile, in der der Fehler aufgetreten ist
            corrected_line = faulty_line[:colom_i] + sp + faulty_line[colom_i:]
            corrected_json_lines.append(corrected_line)
            remaining_lines = json_lines[line_i:]  # Nach der Zeile des Fehlers
            corrected_json_lines.extend(remaining_lines)

            json_str = '\n'.join(corrected_json_lines)

        elif "Extra data" in error_message:
            # Entferne Daten vor dem JSON-String
            start_index = json_str.find('{')
            if start_index != -1:
                json_str = json_str[start_index:]

        elif "Unterminated string starting at" in error_message:
            # Entferne Daten nach dem JSON-String
            line_i = int(error_message[error_message.rfind('line') + 4:error_message.rfind('column')].strip())
            colom_i = int(error_message[error_message.rfind('char') + 4:-1].strip())
            # print(line_i, colom_i)
            index = 1
            new_json_str = ""
            for line in json_str.split('\n'):
                if index == line_i:
                    line = line[:colom_i - 1] + line[colom_i + 1:]
                new_json_str += line
                index += 1
            json_str = new_json_str
        # Versuche erneut, den reparierten JSON-String zu laden
        # {"name": "John", "age": 30, "city": "New York", }

        start_index = json_str.find('{')
        if start_index != -1:
            json_str = json_str[start_index:]

        # Füge fehlende schließende Klammern ein
        count_open = json_str.count('{')
        count_close = json_str.count('}')
        for _i in range(count_open - count_close):
            json_str += '}'

        count_open = json_str.count('[')
        count_close = json_str.count(']')
        for _i in range(count_open - count_close):
            json_str += ']'

        return fix_json(json_str, current_index + 1)


def fixer_parser(input_str):
    max_iterations = 10  # Maximal zulässige Iterationen
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        fixed_json = fix_json(input_str)
        if fixed_json is None:
            return None  # Kann den JSON-String nicht reparieren
        if isinstance(fixed_json, dict):
            return fixed_json
        if isinstance(fixed_json, list):
            return fixed_json
        try:
            parsed_json = json.loads(fixed_json)
            return parsed_json
        except json.JSONDecodeError:
            input_str = fixed_json  # Versuche erneut mit dem reparierten JSON-String

    # Wenn die maximale Anzahl von Iterationen erreicht ist und immer noch ein Fehler vorliegt
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
    if not json_objects and data.startswith('[') and data.endswith(']'):
        json_objects = eval(data)
    if json_objects and len(json_objects) > 0 and isinstance(json_objects[0], dict):
        result.extend([{**expected_keys, **ob} for ob in json_objects])
    if not result:
        completed_object = complete_json_object(data, mini_task)
        if completed_object is not None:
            result.append(completed_object)
    if len(result) == 0 and expected_keys:
        result = [{list(expected_keys.keys())[0]: data}]
    for res in result:
        if isinstance(res, list) and len(res) > 0:
            res = res[0]
        for key, value in expected_keys.items():
            if key not in res:
                res[key] = value

    if len(result) == 0:
        fixed = fix_json(data)
        if fixed:
            result.append(fixed)

    return result


def _extract_from_json(agent_text, all_actions):
    try:
        json_obj = anything_from_str_to_dict(agent_text, {"Action": None, "Inputs": None})
        if json_obj:
            json_obj = json_obj[0]
            if not isinstance(json_obj, dict):
                return None, ''
            action, inputs = json_obj.get("Action"), json_obj.get("Inputs", "")
            if action is not None and action.lower() in all_actions:
                return action, inputs
    except json.JSONDecodeError:
        pass
    return None, ''


def _extract_from_string(agent_text, all_actions):
    action_match = re.search(r"Action:\s*(\w+)", agent_text)
    action_matchs = re.search(r"function:\s*(\w+)", agent_text)
    inputs_match = re.search(r"Inputs:\s*({.*})", agent_text)
    inputs_matchs = re.search(r"Inputs:\s*(.*)", agent_text)
    inputs_matcha = re.search(r"arguments:\s*(.*)", agent_text)

    inputs = ''
    action = None

    if inputs_match is not None:
        inputs = inputs_match.group(1)
    elif inputs_match is not None:
        inputs = inputs_matchs.group(1)
    elif inputs_matcha is not None:
        inputs = inputs_matcha.group(1)

    if action_match is not None:
        action = action_match.group(1)
    if action_matchs is not None:
        action = action_matchs.group(1)

    if action is not None and action.lower() in all_actions:
        action = action.strip()

    return action, inputs


def _extract_from_string_de(agent_text, all_actions):
    action_match = re.search(r"Aktion:\s*(\w+)", agent_text)
    inputs_match = re.search(r"Eingaben:\s*({.*})", agent_text)
    inputs_matchs = re.search(r"Eingaben:\s*(.*)", agent_text)

    if action_match is not None and inputs_match is not None:
        action = action_match.group(1)
        inputs = inputs_match.group(1)
        if action is not None and action.lower() in all_actions:
            return action.strip(), inputs

    if action_match is not None and inputs_matchs is not None:
        action = action_match.group(1)
        inputs = inputs_matchs.group(1)
        print(f"action: {action=}\n{action in all_actions=}\n")
        if action is not None and action.lower() in all_actions:
            return action.strip(), inputs

    if action_match is not None:
        action = action_match.group(1)
        if action is not None and action.lower() in all_actions:
            return action.strip(), ''

    return None, ''

