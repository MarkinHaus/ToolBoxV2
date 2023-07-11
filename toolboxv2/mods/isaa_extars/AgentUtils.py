from pebble import concurrent
import math
from datetime import datetime
import subprocess
import pickle
import requests
from duckduckgo_search import ddg_suggestions
import os
import time
from toolboxv2 import Style, get_logger
import tiktoken
from langchain.agents import AgentType
from langchain.vectorstores import Chroma
from toolboxv2.utils.toolbox import get_app
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
# Data Science
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import platform, socket, re, uuid, json, psutil
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


def get_tokens(text, model_name, only_len=True):
    if '/' in model_name or not (model_name.endswith("4") or model_name.endswith("5")):
        model_name = 'gpt2'

    tokens = tiktoken.encoding_for_model(model_name).encode(text)

    if only_len:
        return len(tokens)
    else:
        return tokens


def get_ip():
    response = requests.get('https://api64.ipify.org?format=json').json()
    return response["ip"]


@concurrent.process(timeout=12)
def get_location():
    ip_address = get_ip()
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_data = f"city: {response.get('city')},region: {response.get('region')},country: {response.get('country_name')},"

    return location_data


def getSystemInfo():
    return f"{time.time()=}"
    # try:
    ip = '0.0.0.0'
    try:
        socket.gethostbyname(socket.gethostname())
    except Exception as e:
        self.logger.error(Style.RED(str(e)))
        pass
    info = {'time': datetime.today().strftime('%Y-%m-%d %H:%M:%S'), 'platform': platform.system(),
            'platform-release': platform.release(), 'platform-version': platform.version(),
            'architecture': platform.machine(), 'hostname': socket.gethostname(),
            'ip-address': ip,
            'mac-address': ':'.join(re.findall('..', '%012x' % uuid.getnode())), 'processor': platform.processor(),
            'ram': str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"}

    try:
        process = get_location()
        info['location'] = process.result()
    except TimeoutError and Exception:
        info['location'] = "Berlin Schöneberg"

    return info


class Scripts:
    def __init__(self, filename):
        self.scripts = {}
        self.filename = filename

    def create_script(self, name, description, content, script_type="python"):
        self.scripts[name] = {"description": description, "content": content, "type": script_type}

    def run_script(self, name):
        if name not in self.scripts:
            return "Script not found!"
        script = self.scripts[name]
        with open(f"{name}.{script['type']}", "w") as f:
            f.write(script["content"])
        if script["type"] == "python":
            result = subprocess.run(["python", f"{name}.py"], capture_output=True, text=True)
        elif script["type"] == "bash":
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
        return f"Index: {self.index}, Question: {self.question}," \
               f" Left child key: {left_value}, Right child key: {right_value}"


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

    def add(self, name, tasks):
        if '/' in name or '\\' in name or ' ' in name:
            name = name.strip()
            name = name.replace('/', '-').replace('\\', '-').replace(' ', '_')
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
        if name in self.chains:
            del self.chains[name]
        else:
            print(f"Chain '{name}' not found.")

    def get(self, name):
        print(f"AgentChain##############{ name in list(self.chains_h.keys())}############# { name}, {list(self.chains_h.keys())}")
        if name in list(self.chains_h.keys()):
            return self.chains_h[name]
        return []

    def init_chain(self, name):
        self.save_to_file(name)
        self.live_chains[name] = self.get(name)

    def add_task(self, name, task):
        if name in self.chains:
            self.chains[name].append(task)
        else:
            print(f"Chain '{name}' not found.")

    def remove_task(self, name, task_index):
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
                    print(f"Die Aufgabe {task_idx}  hat keinen 'args'-Schlüssel.")
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
            except Exception as e:
                print(Style.RED(f"Beim Laden der Datei '{file_path}' ist ein Fehler aufgetreten: {e}"))
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
            chain_data = {"name": name, "tasks": tasks}
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
            chain_data = {"name": name, "tasks": tasks}

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
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2',
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

    def init_store(self, name):
        if not name in self.vector_store.keys():
            self.vector_store[name] = self.get_sto_bo(name)
        lo = False
        if not os.path.exists(self.vector_store[name]['db-path']):
            os.makedirs(self.vector_store[name]['db-path'], exist_ok=True)
        else:
            lo = True
        self.vector_store[name]['db'] = Chroma(collection_name=name,
                                               embedding_function=self.embedding,
                                               persist_directory=self.vector_store[name][
                                                   'db-path'])

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
    def cleanup_list(data: list[str]):

        result = []

        for doc in data:
            if doc and len(doc.strip()) > 10:
                result.append(doc)
        del data
        return result

    def add_data(self, name, data=None):

        if name not in self.vector_store.keys():
            self.init_store(name)

        if data is not None:
            if len(data) < 0:
                return
            self.vector_store[name]['text'] += data

        if not self.vector_store[name]['db']:
            self.init_store(name)

        if (not self.vector_store[name]['text']) or len(self.vector_store[name]['text']) == 0:
            return

        if isinstance(self.vector_store[name]['text'], str):
            vec = self.embedding.embed_query(self.vector_store[name]['text'])
            self.vector_store[name]['db'].add_texts([self.vector_store[name]['text']])
            self.vector_store[name]['vectors'].append(vec)

        if isinstance(self.vector_store[name]['text'], list):
            self.vector_store[name]['text'] = self.cleanup_list(self.vector_store[name]['text'])
            try:
                self.vector_store[name]['db'].add_texts(self.vector_store[name]['text'])
            except ValueError:
                l = len(self.vector_store[name]['text'])
                self.vector_store[name]['db'].add_texts(self.vector_store[name]['text'][:l])
                self.vector_store[name]['db'].add_texts(self.vector_store[name]['text'][l:])
            for vec in self.embedding.embed_documents(self.vector_store[name]['text']):
                self.vector_store[name]['vectors'].append(vec)

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
        if not name in self.vector_store.keys():
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
        if marginal:
            return self.vector_store[name]['db'].max_marginal_relevance_search(text)

        return self.vector_store[name]['db'].similarity_search_with_score(text)

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


class CollectiveMemory:
    collection = []

    do_mem = True
    memory = None
    token_in_use = 1
    text_mem = []
    text_len = 1
    mean_token_len = 1

    isaa = None

    def __init__(self, isaa):
        self.isaa = isaa

    def text(self, context):
        if self.isaa is None:
            raise ValueError("Define Isaa Tool first AgentConfig")
        if not context or context == "None":
            return f"active memory contains {self.token_in_use} tokens for mor informations Input similar information"
        if self.do_mem:
            if self.isaa is not None:
                return self.isaa.get_context_memory().get_context_for(context)
            return None
        # "memory will performe a vector similarity search using memory"
        relevant_memory = self.memory.get_relevant(context, 10)
        if len(relevant_memory) == 0:
            l = ""
            for i in ddg_suggestions(context)[:3]:
                l += i['phrase'] + " ,"
            return f"No data faund in memory try : {l}"

        return "\n#-#".join(relevant_memory)

    def text_add(self, data):
        if self.do_mem:
            return " NO MEMORY Avalabel"
        if not data:
            return

        if isinstance(data, str):
            data = {'data': data, 'token-count': self.mean_token_len + 1, 'vector': []}

        print(
            Style.RED(f"ADD DATA : ColetiveMemory :{len(self.collection)} {data['token-count']} {self.token_in_use=}"),
            end="\r")
        self.token_in_use += data['token-count']
        if data['data'] not in self.text_mem:
            self.text_mem += [data['data']]
            self.memory.add(data['data'])
            self.text_len += len(data['data'])
            self.collection.append(data)
            self.mean_token_len = self.text_len / self.token_in_use

        return f"Data Saved"

    def __str__(self):
        if self.do_mem:
            return " NO MEMORY Avalabel"
        return f"\n{len(self.collection)=}\n{self.memory.get_stats()=}\n" \
               f"{self.token_in_use=}\n{len(self.text_mem)=}\n" \
               f"{self.text_len=}\n{self.mean_token_len=}\n"


class ObservationMemory:
    memory_data: list[dict] = []
    tokens: int = 0
    max_length: int = 1000
    model_name: str = "text-embedding-ada-002"

    add_to_static: list[dict] = []

    isaa = None

    def __init__(self, isaa):
        self.isaa = isaa

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
        for line in CharacterTextSplitter(chunk_size=max(300, int(len(data) / 10)),
                                          chunk_overlap=max(20, int(len(data) / 200))).split_text(data):
            if line:
                ntok = get_tokens(line, self.model_name)
                self.memory_data.append({'data': line, 'token-count': ntok, 'vector': []})
                tok += ntok

        self.tokens += tok

        # print("Tokens add to ShortTermMemory:", tok, " max is:", self.max_length)
        if self.tokens > self.max_length:
            self.cut()

    def cut(self):

        if self.isaa is None:
            raise ValueError("Define Isaa Tool first AgentConfig")

        tok = 0
        all_mem = []
        while self.tokens > self.max_length:
            if len(self.memory_data) == 0:
                break
            # print("Tokens in ShortTermMemory:", self.tokens, end=" | ")
            memory = self.memory_data[-1]
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
    memory_data: list[dict] = []
    tokens: int = 0
    max_length: int = 2000
    model_name: str = "text-embedding-ada-002"

    add_to_static: list[dict] = []

    lines_ = []

    isaa = None

    def __init__(self, isaa, name):
        self.name = name
        self.isaa = isaa
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
        while self.tokens > self.max_length:
            if len(self.memory_data) == 0:
                break
            # print("Tokens in ShortTermMemory:", self.tokens, end=" | ")
            memory = self.memory_data[-1]
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

        return memorys

    @text.setter
    def text(self, data):
        tok = 0
        if not isinstance(data, str):
            print(f"DATA text edd {type(data)} data {data}")

        for line in CharacterTextSplitter(chunk_size=max(300, int(len(data) / 10)),
                                          chunk_overlap=max(20, int(len(data) / 200))).split_text(data):
            if line not in self.lines_ and len(line) != 0:
                ntok = get_tokens(line, self.model_name)
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
        self.global_env = globals().copy()
        self.local_env = {}

    def eval_code(self, code):
        try:
            result = eval(code, self.global_env, self.local_env)
            return self.format_output(result)
        except Exception as e:
            return self.format_output(e)

    def get_env(self):
        # global_env_str = self.format_env(self.global_env)
        local_env_str = self.format_env(self.local_env)
        return f"Locals:\n{local_env_str}"

    @staticmethod
    def format_output(output):
        return f"Ergebnis: {output}"

    @staticmethod
    def format_env(env):
        return "\n".join(f"{key}: {value}" for key, value in env.items())

    def run_and_display(self, code):
        # Anfangszustand anzeigen
        end = ""
        end += f"Startzustand:\n{self.get_env()}"

        # Code ausführen
        result = self.eval_code(code)

        # Endzustand anzeigen
        end += f"\nEndzustand:\n{self.get_env()}"

        # Ergebnis anzeigen
        end += f"\nAusführungsergebnis:\n{result}"
        return end


class AgentConfig:
    available_modes = ['tools', 'free', 'planning', 'execution', 'generate']  # [ 'talk' 'conversation','q2tree', 'python'
    max_tokens = 3950

    python_env = PyEnvEval()

    capabilities = """Resourceful: Isaa is able to efficiently utilize its wide range of capabilities and resources to assist the user.
Collaborative: Isaa is work seamlessly with other agents, tools, and systems to deliver the best possible solutions for the user.
Empathetic: Isaa is understand and respond to the user's needs, emotions, and preferences, providing personalized assistance.
Inquisitive: Isaa is continually seek to learn and improve its knowledge base and skills, ensuring it stays up-to-date and relevant.
Transparent: Isaa is open and honest about its capabilities, limitations, and decision-making processes, fostering trust with the user.
Versatile: Isaa is adaptable and flexible, capable of handling a wide variety of tasks and challenges."""
    system_information = f"""
system information's : {getSystemInfo()}
"""

    def __init__(self, isaa, name="agentConfig"):

        self.isaa = isaa

        if self.isaa is None:
            raise ValueError("Define Isaa Tool first AgentConfig")

        self.name: str = name
        self.mode: str = "talk"
        self.model_name: str = "gpt-3.5-turbo-0613"

        self.agent_type: AgentType = AgentType("structured-chat-zero-shot-react-description")  # "zero-shot-react-description"
        self.max_iterations: int = 2
        self.verbose: bool = True

        self.personality = ""
        self.goals = ""
        self.tools: dict = {
            # "test_tool": {"func": lambda x: x, "description": "only for testing if tools are available",
            #              "format": "test_tool(input:str):"}
        }
        self.toolsL = []

        self.last_prompt = ""

        self.task_list: list[str] = []
        self.task_list_done: list[str] = []
        self.step_between: str = ""

        self.pre_task: str or None = None
        self.task_index = 0

        self.token_left = 2000
        self.temperature = 0.06
        self.messages_sto = {}
        self._stream = False
        self._stream_reset = False
        self.stop_sequence = ["\n\n\n", "Observation:", "Execute:"]
        self.completion_mode = "text"
        self.add_system_information = True

        self.init_mem_state = False
        self.context: None or ShortTermMemory = None
        self.observe_mem: None or ObservationMemory = None
        self.edit_text: None or ShortTermMemory = None
        self.short_mem: None or ShortTermMemory = None

        self.init_memory()

        self.binary_tree: IsaaQuestionBinaryTree or None = None

    def init_memory(self):
        self.init_mem_state = True
        self.short_mem: ShortTermMemory = ShortTermMemory(self.isaa, f'{self.name}-ShortTerm')
        self.edit_text: ShortTermMemory = ShortTermMemory(self.isaa, f'{self.name}-EditText')
        self.edit_text.max_length = 5400
        self.context: ShortTermMemory = ShortTermMemory(self.isaa, f'{self.name}-ContextMemory')
        self.observe_mem: ObservationMemory = ObservationMemory(self.isaa)

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

    def clone(self):

        return

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

    @property
    def a_messages(self) -> list:
        key = f"{self.name}-{self.mode}"

        if key not in self.messages_sto.keys():
            self.init_message(key)

        last_prompt = ""
        for msg in self.messages_sto[key]:
            last_prompt += msg['content']
        i = 0
        tokens = get_tokens(last_prompt, self.model_name)
        while self.max_tokens < tokens:
            i += 1
            self.messages_sto[key] = []
            self.init_message(key)
            if i > 2:
                temp_message = []
                mas_text_sum = self.isaa.mas_text_summaries
                for msg in self.messages_sto[key]:
                    temp_message.append({'role': msg['role'], 'content': mas_text_sum(msg['content'])})
                self.messages_sto[key] = temp_message
            if i > 3:
                self.messages_sto[key] = [self.messages_sto[key][-2], self.messages_sto[key][-1]]
            if i > 4:
                self.messages_sto[key] = [self.messages_sto[key][-1]]
                break
            last_prompt = ""
            for msg in self.messages_sto[key]:
                last_prompt += msg['content']
            tokens = get_tokens(last_prompt, self.model_name)

        self.last_prompt = last_prompt

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

        pl = get_tokens(prompt, self.model_name) + 2
        self.token_left = self.max_tokens-pl
        print("TOKEN LEFT : ", self.token_left, "Token in Prompt :", pl, "Max tokens :", self.max_tokens)
        if pl > self.max_tokens:
            self.short_mem.cut()
        if self.token_left < 0:
            self.token_left *= -1
            self.short_mem.max_length = self.token_left
            self.short_mem.cut()
        self.last_prompt = prompt
        return prompt

    def __str__(self):

        return f"\n{self.name=}\n{self.mode=}\n{self.model_name=}\n{self.agent_type=}\n{self.max_iterations=}" \
               f"\n{self.verbose=}\n{self.personality[:45]=}\n{self.goals[:45]=}" \
               f"\n{str(self.tools)[:45]=}\n{self.task_list=}\n{self.task_list_done=}\n{self.step_between=}\n\nshort_mem\n{self.short_mem.info()}\nObservationMemory\n{self.observe_mem.info()}\nCollectiveMemory\n{str(CollectiveMemory(self.isaa))}\n"

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

        prompt = f"Answer the following questions as best you can." \
                 f" You have access to a live python interpreter write run python code" \
                 f"\ntake all (Observations) into account!!!" \
                 f"Personality:'''{self.personality}'''\n\n" + \
                 f"Goals:'''{self.goals}'''\n\n" + \
                 f"Capabilities:'''{self.capabilities}'''\n\n" + \
                 f"Permanent-Memory:\n'''{CollectiveMemory(self.isaa).text(context=task)}'''\n\n" + \
                 f"Resent Agent response:\n'''{self.observe_mem.text}'''\n\n" + \
                 f"\n\nBegin!\n\n" \
                 f"Task:'{task}\n{self.short_mem.text}\nAgent : "

        if self.mode == 'planning':
            prompt = """
You are a planning agent , and your job is to create the most efficient plan for a given input.
There are {{""" + f"\n{tools}\n" + """}} different functions that must be called in the correct order and with the
correct inputs. Your goal is to find a plan that accomplishes the task. Create a plan for the input
{{""" + f"{task}" + """}} by providing the following information:

1.    Function(s) and Input(s) you wish to invoke, selecting only those useful for executing the plan.
2.    focusing on efficiency and minimizing steps.

Actual Observations: {{""" + f"{self.observe_mem.text}" + """}}

you have aces to a live python wirte valid python code and it will be executed.

Note that your plan should be clear and understandable. Strive for the most efficient solution to accomplish the task.
Use only the features that are useful for executing the plan. Return a detailed Plan.
Begin.

Plan:"""

        if self.mode == 'execution':
            prompt = """
You are an execution agent, and your task is to implement the plan created by a planning agent.
You can call functions using the following syntax:

Action: Function-Name
Inputs: Inputs
Execute:
Observations:
Conclusion: <conclusion>
Outline of the next step:
Action: Function-Name
...
Final Answer:
// Return Final Answer: wen you ar don execution

To successfully execution the plan, it is important to pay attention to writing "Execute:" to execute the function.
If additional user input is needed during the execution process,
use the syntax "Question:" followed by a question to obtain the required information from the user. ending with "\\nUser:"

Please execute the plan created by a planning agent by following these steps:

1)    Analyze the plan provided by a planning agent and understand the order of functions and inputs.
2)    Call the functions in the specified order using the syntax described above.
3)    Ask questions to the user if additional information is needed and integrate the answers into the plan.
4)    Make sure to write "Execute:" to execute each function.
5)    Verify if the plan has been successfully implemented and provide feedback to the planning agent if necessary.

Ensure that your implementation is clear and comprehensible
so that other people can follow the progress and provide support if needed.

you have aces to a live python wirte valid python code and it will be executed.
You have access to following functions : {{""" + f"\n{tools}" + """}}

The Plan to execute : {{""" + f"\n{task_list}\n" + """}}
Resent Observations : {{""" + f"{self.observe_mem.text}" + """}}
Begin!

""" + f"{self.short_mem.text}\n" + """
Current Step : """ + f"{task}\nPerform Action\n"

        if self.mode == 'generate':
            prompt = """
You are a prompt creation agent, and your task is to create effective and engaging prompts for various topics and requirements.
Your prompts should be clear and understandable, encouraging users to provide deep and interesting responses.
Please follow these steps to create a new prompt for a given topic or requirement:

1)    Carefully analyze the desired topic or requirement to gain a clear understanding of the expectations.
2)    Develop an engaging and open-ended question or prompt that encourages users to share their thoughts, experiences, or ideas.
3)    Make sure the prompt is clearly and understandably worded so that users of different knowledge and experience levels can easily understand it.
4)    Ensure the prompt is flexible enough to allow for creative and diverse responses, while also providing enough structure to focus on the desired topic or requirement.
5)    Review your prompt for grammar, spelling, and style to ensure a professional and appealing presentation.

Resent Observations : {{""" + f"{self.edit_text.text}" + """}}

Task : """ + f"\n{task}\n\nBegin!\nPrompt:"

        if self.mode in ['talk', 'conversation']:
            prompt = f"Goals:{self.goals.replace('{input}', '')}\n" + \
                     f"Capabilities:{self.capabilities.replace('{input}', '')}\n" + \
                     f"Important information: to run a tool type 'Action: $tool-name'\n" + \
                     f"Long-termContext:{CollectiveMemory(self.isaa).text(context=self.short_mem.text).replace('{input}', '')}\n" + \
                     f"\nResent Observation:{self.observe_mem.text.replace('{input}', '')}" + \
                     f"Task:{task}\n" + \
                     "{input}" + \
                     f"\n{self.short_mem.text.replace('{input}', '')}"

        if self.mode == 'tools':
            prompt = f"Answer the following questions as best you can. You have access to the following python functions\n" \
                     f"'''{tools}'''" \
                     f"\ntake all (Observations) into account!!!\nUnder no circumstances are you allowed to output 'Task:'!!!\n\n" \
                     f"Personality:'''{self.personality}'''\n\n" + \
                     f"Goals:'''{self.goals}'''\n\n" + \
                     f"Capabilities:'''{self.capabilities}'''\n\n" + \
                     f"Permanent-Memory:\n'''{CollectiveMemory(self.isaa).text(context=task)}'''\n\n" + \
                     f"Resent Agent response:\n'''{self.observe_mem.text}'''\n\n" + \
                     "Use the following format To Run an Python Function:\n\n" + \
                     "Task: the input question you must answer\n" + \
                     "Thought: you should always think about what to do\n" + \
                     "Knowleg: Knowleg from The LLM abaut the real world\n" + \
                     f"Action: the action to take, should be one of {names}\n" + \
                     f"Execute: <function_name>(<args>)\n" + \
                     f"\n\nBegin!\n\n" \
                     f"Resent:\n{self.short_mem.text}\nTask:'{task}"
            prompt = prompt.replace('{', '{{').replace('}', '}}').replace('input}', '') + '{input}'

        if self.mode == 'free':
            if self.name != "self":
                prompt = task

        if self.mode == 'q2tree':
            if self.binary_tree:
                questions_ = self.binary_tree.get_left_side(0)
                questions = '\n'.join(
                    f"Question {i + 1} : {q.replace('task', f'task ({task})')}" for i, q in enumerate(questions_))
                prompt = f"""Answer the following questions as best you can. Your:
Capabilities:
'''
Functions: ''\n{tools}''
Goals: ''\n{self.goals}''
: ''\n{self.capabilities}''
'''

I'm going to ask you {len(questions_)} questions, answer the following questions as best you can.
You are known to think in small and detailed steps to get the right result.
Task : '''{task}'''
imagen yourself doing the task.
{questions}

Answer-format:
Answer 1: ...

Begin!"""

        return prompt

    def next_task(self):
        if len(self.task_list) < self.task_index:
            self.task_index += 1
        return self

    def set_mode(self, mode: str):
        """Set The Mode of The Agent available ar
         ['talk', 'tool', 'conversation', 'free', 'planning', 'execution', 'generate']"""
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
        self.tools = tools
        return self

    def set_toolsL(self, tools: list):
        self.toolsL = tools
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
        self.max_iterations = max_iterations
        return self

    def set_verbose(self, verbose: bool):
        self.verbose = verbose
        return self

    def set_personality(self, personality: str):
        self.personality = personality
        return self

    def set_goals(self, goals: str):
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

    def set_observation_memory(self, obser_mem: ObservationMemory):
        self.observe_mem = obser_mem
        return self

    def set_model_name(self, model_name: str):
        """OpenAI modes """
        self.model_name = model_name
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
            if key == 'tools' and value:
                if agent_config.name != 'self':
                    agent_config.set_tools(value)
                continue
            setattr(agent_config, key, value)

        return agent_config


