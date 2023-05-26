import json
import logging
import math
import random
from datetime import datetime
from typing import Tuple, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from duckduckgo_search import ddg, ddg_answers, ddg_suggestions, ddg_news
import replicate
from transformers import pipeline, AutoModelWithLMHead, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, \
    AutoModelForCausalLM
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import os
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfFolder
from huggingface_hub import InferenceApi
import time
from langchain.agents import initialize_agent, tool, Tool as AgentTool
import torch

from toolboxv2 import MainTool, FileHandler, Style, App, Spinner, get_logger

from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.llms.openai import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import tiktoken
from bs4 import BeautifulSoup

import pinecone
import openai
from inspect import signature

from langchain.python import PythonREPL
from langchain.tools.python.tool import PythonREPLTool
from langchain import LLMMathChain, SerpAPIWrapper, GoogleSerperAPIWrapper
from langchain.utilities import BashProcess
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.tools.ifttt import IFTTTWebhook

from langchain.vectorstores import Chroma

from toolboxv2.utils.Style import print_to_console
from toolboxv2.utils.toolbox import Singleton, get_app

from langchain.document_loaders import PyPDFLoader

# Loaders
from langchain.schema import Document

# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TextSplitter

# Model
from langchain.chat_models import ChatOpenAI

# Embedding Support
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain

# Data Science
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

import platform, socket, re, uuid, json, psutil, logging
from langchain.indexes import VectorstoreIndexCreator

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


def get_location():
    ip_address = get_ip()
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_data = f"city: {response.get('city')},region: {response.get('region')},country: {response.get('country_name')},"

    return location_data


def getSystemInfo():
    # try:
    info = {'time': datetime.today().strftime('%Y-%m-%d %H:%M:%S'), 'platform': platform.system(),
            'platform-release': platform.release(), 'platform-version': platform.version(),
            'architecture': platform.machine(), 'hostname': socket.gethostname(),
            'ip-address': socket.gethostbyname(socket.gethostname()),
            'mac-address': ':'.join(re.findall('..', '%012x' % uuid.getnode())), 'processor': platform.processor(),
            'ram': str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB", 'location': get_location()}
    return info


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]


def get_text_from_embedding(embedding):
    return openai.Embedding.retrieve(embedding, model="text-embedding-ada-002")["data"][0]["text"]


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


class AgentChain(metaclass=Singleton):
    def __init__(self, hydrate=None, f_hydrate=None):
        self.chains = {}
        self.chains_h = {}
        self.live_chains = {}
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
        return self.chains_h[name]

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
                self.chains[name].pop(task_index)
            else:
                print(f"Task index '{task_index}' is out of range.")
        else:
            print(f"Chain '{name}' not found.")

    def test_chain(self):
        for chain_name, tasks in self.chains.items():
            for task_idx, task in enumerate(tasks):
                if "use" not in task:
                    print(f"Die Aufgabe {task_idx} in der Chain '{chain_name}' hat keinen 'use'-Schlüssel.")
                if "name" not in task:
                    print(f"Die Aufgabe {task_idx} in der Chain '{chain_name}' hat keinen 'name'-Schlüssel.")
                if "args" not in task:
                    print(f"Die Aufgabe {task_idx} in der Chain '{chain_name}' hat keinen 'args'-Schlüssel.")

    def load_from_file(self, chain_name=None):
        directory = ".data/chains"

        self.chains = self.live_chains

        if not os.path.exists(directory):
            print(f"Der Ordner '{directory}' existiert nicht.")
            return

        if chain_name is None:
            files = os.listdir(directory)
        else:
            files = [f"{chain_name}.json"]
        print(f"--------------------------------")
        for file in files:
            file_path = os.path.join(directory, file)

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
                print(f"Beim Laden der Datei '{file_path}' ist ein Fehler aufgetreten: {e}")
        print(f"--------------------------------")
        print(
            f"\n================================\nChainsLoaded : {len(self.chains.keys())}\n================================\n")

        return self

    def save_to_file(self, chain_name=None):
        directory = ".data/chains"

        if not os.path.exists(directory):
            os.makedirs(directory)

        if chain_name is None:
            chains_to_save = self.chains
        else:
            if chain_name not in self.chains:
                print(f"Die Chain '{chain_name}' wurde nicht gefunden.")
                return
            chains_to_save = {chain_name: self.chains[chain_name]}
        print(f"--------------------------------")
        for name, tasks in chains_to_save.items():
            file_path = os.path.join(directory, f"{name}.json")
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


class PineconeMemory(metaclass=Singleton):
    def __init__(self):
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_region = os.getenv("PINECONE_ENV")
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_region)
        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        table_name = "isaa-memory"
        # this assumes we don't start with memory.
        # for now this works.
        # we'll need a more complicated and robust system if we want to start with memory.
        self.vec_num = 0
        if table_name not in pinecone.list_indexes():
            pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)
        self.index = pinecone.Index(table_name)

    def add(self, data):
        vector = get_ada_embedding(data)
        # no metadata here. We may wish to change that long term.
        resp = self.index.upsert([(str(self.vec_num), vector, {"raw_text": data})])
        _text = f"Inserting data into memory at index: {self.vec_num}:\n data: {data}"
        self.vec_num += 1
        return _text

    def get(self, data):
        return self.get_relevant(data, 1)

    def clear(self):
        self.index.delete(deleteAll=True)
        return "Obliviated"

    def get_relevant(self, data, num_relevant=5):
        """
        Returns all the data in the memory that is relevant to the given data.
        :param data: The data to compare to.
        :param num_relevant: The number of relevant data to return. Defaults to 5
        """
        try:
            query_embedding = get_ada_embedding(data)
            results = self.index.query(query_embedding, top_k=num_relevant, include_metadata=True)
            sorted_results = sorted(results.matches, key=lambda x: x.score)
            return [str(item['metadata']["raw_text"]) for item in sorted_results]
        except Exception:
            return ""

    def get_stats(self):
        return self.index.describe_index_stats()


class AIContextMemory(metaclass=Singleton):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):  # "MetaIX/GPT4-X-Alpaca-30B-4bit"):
        self.memory = {
            'rep': []
        }
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = {}

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

    @staticmethod
    def get_sto_bo(name):
        return {'text': [],
                'full-text-len': 0,
                'vectors': [],
                'db': None,
                'len-represent': 0,
                'db-path': f'.data/{get_app().id}/Memory/{name}',
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
            self.vector_store[name]['text'] += data

        if not self.vector_store[name]['db']:
            self.init_store(name)

        if not self.vector_store[name]['text'] or len(self.vector_store[name]['text']) == 0:
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

        data = self.search(mem_name['key'], text, marginal=marginal)
        last = []
        final = f"Data from ({mem_name['key']}):\n"

        for res in data:
            if last != res:
                final += res.page_content + '\n\n'
            else:
                print("WARNING- same")
        return final


class CollectiveMemory(metaclass=Singleton):
    collection = []
    try:
        memory = PineconeMemory()
        do_mem = False
    except Exception:
        do_mem = True
        memory = None
    token_in_use = 1
    text_mem = []
    text_len = 1
    mean_token_len = 1

    def text(self, context):
        if not context or context == "None":
            return f"active memory contains {self.token_in_use} tokens for mor informations Input similar information"
        if self.do_mem:
            return get_tool(get_app()).get_context_memory().get_context_for(context)
        "memory will performe a vector similarity search using memory"
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
                ntok = len(tiktoken.encoding_for_model(self.model_name).encode(line))
                self.memory_data.append({'data': line, 'token-count': ntok, 'vector': []})
                tok += ntok

        self.tokens += tok

        # print("Tokens add to ShortTermMemory:", tok, " max is:", self.max_length)
        if self.tokens > self.max_length:
            self.cut()

    def cut(self):

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

        get_tool(get_app()).get_context_memory().add_data('observations', all_mem)

        print(f"Removed ~ {tok} tokens from ObservationMemory tokens in use: {self.tokens} ")


class ShortTermMemory:
    memory_data: list[dict] = []
    tokens: int = 0
    max_length: int = 2000
    model_name: str = "text-embedding-ada-002"

    add_to_static: list[dict] = []

    lines_ = []

    def __init__(self, name):
        self.name = name

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
            get_tool(get_app()).get_context_memory().add_data(self.name, all_mem)

        print(f"\nRemoved ~ {tok} tokens from {self.name} tokens in use: {self.tokens} max : {self.max_length}")

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
                ntok = len(tiktoken.encoding_for_model(self.model_name).encode(line))
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


class AgentConfig:
    available_modes = ['talk', 'tools', 'conversation', 'free', 'planning', 'execution', 'generate', 'q2tree']
    max_tokens = 4097

    capabilities = """1. Invoke Agents: Isaa should be able to invoke and interact with various agents and tools, seamlessly integrating their expertise and functionality into its own responses and solutions.
    2. Context Switching: Isaa should be capable of switching between different agents or tools depending on the user's needs and the requirements of the task at hand.
    3. Agent Coordination: Isaa should effectively coordinate the actions and outputs of multiple agents and tools, ensuring a cohesive and unified response to the user's requests.
    4. Error Handling: Isaa should be able to detect and handle errors or issues that may arise while working with agents and tools, providing alternative solutions or gracefully recovering from failures.
    5. Agent Learning: Isaa should continuously learn from its interactions with various agents and tools, refining its understanding of their capabilities and improving its ability to utilize them effectively.
    6. Performance Evaluation: Isaa should regularly evaluate the performance of the agents and tools it interacts with, identifying areas for improvement and optimizing their use in future tasks."""

    system_information = f"""
system information's : {getSystemInfo()}

"""

    def __init__(self):
        self.name: str = 'agentConfig'
        self.mode: str = "talk"
        self.model_name: str = "gpt-3.5-turbo"

        self.agent_type: str = "zero-shot-react-description"
        self.max_iterations: int = 2
        self.verbose: bool = True

        self.personality = ""
        self.goals = ""
        self.short_mem: ShortTermMemory = ShortTermMemory('ShortTerm')
        self.edit_text: ShortTermMemory = ShortTermMemory('EditText')
        self.edit_text.set_name("CodeMemory")
        self.edit_text.max_length = 5400
        self.context: ShortTermMemory = ShortTermMemory('Context')
        self.context.set_name("ContextMemory")
        self.observe_mem: ObservationMemory = ObservationMemory()
        self.tools: dict = {
            "test_tool": {"func": lambda x: x, "description": "only for testing if tools are available",
                          "format": "test_tool(input:str):"}
        }

        self.last_prompt = ""

        self.task_list: list[str] = []
        self.task_list_done: list[str] = []
        self.step_between: str = ""

        self.task_index = 0

        self.token_left = 2000
        self.temperature = 0.06
        self.messages_sto = {}
        self._stream = False
        self._stream_reset = False
        self.stop_sequence = ["\n\n", "Observation:", "Execute:"]
        self.completion_mode = "text"
        self.add_system_information = True

        self.binary_tree: IsaaQuestionBinaryTree or None = None

    def task(self, reset_step=False):
        if self.step_between:
            task = self.step_between
            if reset_step:
                self.step_between = ""
            return task
        if len(self.task_list) != 0:
            task = self.task_list[self.task_index]
            return task
        return "Task is done return Summarization for user"

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
        if key not in self.messages_sto.keys():
            return []
        messages = self.messages_sto[key]
        if create:
            messages = self._messages
        return messages

    @property
    def _messages(self) -> list:
        key = f"{self.name}-{self.mode}"

        if key not in self.messages_sto.keys():
            self.init_message(key)

        last_prompt = ""
        for msg in self.messages_sto[key]:
            last_prompt += msg['content']
        i = 0
        while self.max_tokens < len(tiktoken.encoding_for_model(self.model_name).encode(last_prompt)):
            i += 1
            self.messages_sto[key] = []
            self.init_message(key)
            if i > 2:
                temp_message = []
                mas_text_sum = get_tool(get_app()).mas_text_summaries
                for msg in self.messages_sto[key]:
                    temp_message.append({'role': msg['role'], 'content': mas_text_sum(msg['content'])})
                self.messages_sto[key] = temp_message
            if i > 3:
                self.messages_sto[key] = [self.messages_sto[key][-2], self.messages_sto[key][-1]]
            if i > 4:
                self.messages_sto[key] = [self.messages_sto[key][-1]]
                break
            last_prompt = []
            for msg in self.messages_sto[key]:
                last_prompt += msg['content']

        self.last_prompt = last_prompt

        return self.messages_sto[key]

    @property
    def prompt(self) -> str:
        if not self.short_mem.model_name:
            self.short_mem.model_name = self.model_name
        if not self.observe_mem.model_name:
            self.observe_mem.model_name = self.model_name

        prompt = (self.system_information if self.add_system_information else '') + self.get_prompt()

        pl = len(tiktoken.encoding_for_model(self.model_name).encode(prompt)) + 2
        self.token_left = self.max_tokens - pl
        if self.token_left < 0:
            print(f"No tokens left cut short_mem mem to {self.token_left * -1}")
            self.token_left *= -1
            self.short_mem.max_length = self.token_left
            self.short_mem.cut()
        if self.token_left > 2800:
            self.token_left = 2800
        self.last_prompt = prompt
        return prompt

    def __str__(self):

        return f"\n{self.name=}\n{self.mode=}\n{self.model_name=}\n{self.agent_type=}\n{self.max_iterations=}" \
               f"\n{self.verbose=}\n{self.personality[:45]=}\n{self.goals[:45]=}" \
               f"\n{str(self.tools)[:45]=}\n{self.task_list=}\n{self.task_list_done=}\n{self.step_between=}\nshort_mem\n{self.short_mem.info()}\nObservationMemory\n{self.observe_mem.info()}\nCollectiveMemory\n{str(CollectiveMemory())}\n"

    def get_prompt(self):

        tools = ""
        names = []
        for key, value in self.tools.items():
            format_ = value['format'] if 'format' in value.keys() else f"{key}('function input')"
            if format_.endswith("('function input')"):
                value['format'] = format_
            tools += f"{key.strip()}: {value['description'].strip()} - {format_.strip()}\n"
            names.append(key)
        task = self.task(reset_step=True)
        task_list = '\n'.join(self.task_list)

        prompt = f"Answer the following questions as best you can. You have access to the following python functions\n" \
                 f"'''{tools}'''" \
                 f"to run a function response with \nAction: <function_name>(<args>)\nExecute:" + \
                 f"\ntake all (Observations) into account!!!\nUnder no circumstances are you allowed to output " \
                 f"'Task:'!!!\n\n" \
                 f"Personality:'''{self.personality}'''\n\n" + \
                 f"Goals:'''{self.goals}'''\n\n" + \
                 f"Capabilities:'''{self.capabilities}'''\n\n" + \
                 f"Permanent-Memory:\n'''{CollectiveMemory().text(context=task)}'''\n\n" + \
                 f"Resent Agent response:\n'''{self.observe_mem.text}'''\n\n" + \
                 f"\n\nBegin!\n\n" \
                 f"Task:'{task}\n{self.short_mem.text}\nAgent : "

        if self.mode == 'planning':
            prompt = """
You are a planning agent, and your job is to create the shortest and most efficient plan for a given input {{x}}.
There are {{""" + f"\n{tools}\n" + """}} different functions that must be called in the correct order and with the correct inputs. Your goal is to find the shortest plan that accomplishes the task. Please create a plan for the input {{""" + f"{task}" + """}} by providing the following information:

1.    Function(s) and Input(s) you wish to invoke, selecting only those useful for executing the plan.
2.    focusing on efficiency and minimizing steps.

Actual Observations: {{""" + f"{self.observe_mem.text}" + """}}

Note that your plan should be clear and understandable. Strive for the shortest and most efficient solution to accomplish the task. Use only the features that are useful for executing the plan.
Begin.

Plan:"""

        if self.mode == 'execution':
            prompt = """
You are an execution agent, and your task is to implement the plan created by a planning agent.
You can call functions using the following syntax:

Action: Function-Name
Inputs: Inputs
Execute:

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
                     f"Long-termContext:{CollectiveMemory().text(context=self.short_mem.text).replace('{input}', '')}\n" + \
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
                     f"Permanent-Memory:\n'''{CollectiveMemory().text(context=task)}'''\n\n" + \
                     f"Resent Agent response:\n'''{self.observe_mem.text}'''\n\n" + \
                     "Placeholders: there are two types of place holders.  " \
                     "\n1 : <information_from_agent> if you see a word enclosed with <> this character you must fill it with information!" \
                     "\n2 : [datime] if you want the system to insert an information return []." \
                     "\nallways fill placeholders of type 1 !!!\n\n " + \
                     "Use the following format To Run an Python Function:\n\n" + \
                     "Task: the input question you must answer\n" + \
                     "Thought: you should always think about what to do\n" + \
                     "Knowleg: Knowleg from The LLM abaut the real world\n" + \
                     f"Action: the action to take, should be one of {names}\n" + \
                     f"Execute: <function_name>(<args>)\n" + \
                     f"\n\nBegin!\n\n" \
                     f"Resent:\n{self.short_mem.text}\nTask:'{task}"
            prompt = prompt.replace('{input}', '') + '{input}'

        if self.mode == 'free':
            prompt = task

        if self.mode == 'q2tree':
            if self.binary_tree:
                questions_ = self.binary_tree.get_left_side(0)
                questions = '\n'.join(
                    f"Question {i + 1} : {q.replace('task', f'task ({task})')}" for i, q in enumerate(questions_))
                prompt = f"""Answer the following questions as best you can. You have access to
Capabilities:
'''
Functions: ''\n{tools}''
Goals: ''\n{self.goals}''
Capabilities: ''\n{self.capabilities}''
'''

I'm going to ask you {len(questions_)} +1 question, answer the following questions as best you can.
You are known to think in small and detailed steps to get the right result.
Task : '''{task}'''
imagen yourself doing the task.
{questions}

After the question you answered negatively, output the following N/A !
Answer-format:
Answer 1: ...
. ...

Answer: Let's work this out in a step by step way to be sure we have the right answer.

Begin!
"""

        return prompt

    def next_task(self):
        self.task_index += 1
        return self

    def set_mode(self, mode: str):
        """Set The Mode of The Agent available ar
         ['talk', 'tool', 'conversation', 'free', 'planning', 'execution', 'generate']"""
        if mode in self.available_modes:
            self.mode = mode
        else:
            raise ValueError(f"Ungültiger Modus '{mode}', verfügbare Modi sind: {self.available_modes}")

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

    def add_tool(self, tool_name: str, func: callable, description: str, format_: str):
        self.tools[tool_name] = {
            "func": func,
            "description": description,
            "format": format_,
        }
        return self

    def set_agent_type(self, agent_type: str):
        """langchain agent type"""
        self.agent_type = agent_type
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

    def set_observation_memory(self, obser_mem: ObservationMemory):
        self.observe_mem = obser_mem
        return self

    def set_model_name(self, model_name: str):
        """OpenAI modes """
        self.model_name = model_name
        return self


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "isaa"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "VIOLET2"
        self.inference = InferenceApi
        self.config = {'genrate_image-init': False,
                       'agents-name-lsit': []
                       }
        self.per_data = {}
        self.keys = {
            "KEY": "key~~~~~~~",
            "Config": "config~~~~"
        }
        self.initstate = {}
        self.genrate_image = image_genrating_tool
        self.observation_term_mem_file = f".data/{app.id}/Memory/observationMemory/"
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["Run", "Starts Inference"],
                    ["add_api_key", "Adds API Key"],
                    ["login", "Login"],
                    ["new-sug", "Add New Question or Class to Config"],
                    ["run-sug", "Run Huggingface Pipeline"],
                    ["info", "Show Config"],
                    ["lode", "lode models"],
                    ["isaa", "Run Isaa name input"],
                    ["image", "genarate image input"],
                    ],
            "name": "isaa",
            "Version": self.show_version,
            "info": self.info,
            "isaa": self.run_isaa_wrapper,
            "image": self.genrate_image_wrapper,
        }
        self.app_ = app
        self.print_stream = print
        self.agent_collective_senses = False
        self.global_stream_override = False
        self.pipes_device = 1
        self.speak = lambda x, *args, **kwargs: x

        FileHandler.__init__(self, "issa.config", app.id if app else __name__)
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=None, color=self.color, on_exit=self.on_exit)

    def run_isaa_wrapper(self, command):
        if len(command) < 1:
            return "Unknown command"

        return self.run_agent(command[0], command[1:])

    def genrate_image_wrapper(self, command):
        if len(command) != 1:
            return "Unknown command"

        return self.genrate_image(command[0], self.app_)

    def show_version(self):
        self.print("Version: ", self.version)

    def add_str_to_config(self, command):
        if len(command) != 3:
            self.logger.error('Invalid command must be key value')
            return False
        self.config[command[1]] = command[2]

    def on_start(self):
        self.load_file_handler()
        config = self.get_file_handler(self.keys["Config"])
        if config is not None:
            self.config = eval(config)
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

    def on_exit(self):
        for key in self.config.keys():
            if key.endswith("-init"):
                self.config[key] = False
        self.add_to_save_file_handler(self.keys["Config"], str(self.config))
        self.save_file_handler()

    def info(self):
        self.print(self.config)
        return self.config

    def init_all_pipes_default(self):
        self.init_pipeline('question-answering', "deepset/roberta-base-squad2")
        time.sleep(2)
        self.init_pipeline('summarization', "pinglarin/summarization_papers")
        time.sleep(2)
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
        self.init_pipeline('summarization', model)
        return self.config["summarization_pipeline"](text, **kwargs)

    def text_classification(self, text, model="distilbert-base-uncased-finetuned-sst-2-english", **kwargs):
        self.init_pipeline('text-classification', model)
        return self.config["text-classification_pipeline"](text, **kwargs)

    def toolbox_interface(self):
        @tool("toolbox", return_direct=False)
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
        @tool("toolbox_infos", return_direct=False)
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
        @tool("Image", return_direct=False)
        def function(query: str) -> str:
            """Generate image with Stable diffusion"""
            try:
                image_genrating_tool(query, self.app_)
            except NameError as e:
                return "Das hat leider nicht geklappt ein Fehler tip versuche es auf englisch, benutze synonyme" \
                       " ist bei der ausfürgung des Tools aufgetreten Fehler meldung : " + str(e)
            return "Das bild wird in kürze angezeigt"

        return function

    def load_OpenAI_models(self, names: list):
        for model in names:
            if f'OpenAI-model-{model}-init' not in self.initstate.keys():
                self.initstate[f'OpenAI-model-{model}-init'] = False

            if not self.initstate[f'OpenAI-model-{model}-init']:
                self.initstate[f'OpenAI-model-{model}-init'] = True
                if model.startswith('gpt'):
                    self.config[f'OpenAI-model-{model}'] = ChatOpenAI(model_name=model,
                                                                      openai_api_key=self.config['OPENAI_API_KEY'],
                                                                      streaming=True)
                else:
                    self.config[f'OpenAI-model-{model}'] = OpenAI(model_name=model,
                                                                  openai_api_key=self.config['OPENAI_API_KEY'])

    def get_OpenAI_models(self, name: str):
        if f'OpenAI-model-{name}' not in self.config.keys():
            self.load_OpenAI_models([name])
        return self.config[f'OpenAI-model-{name}']

    def add_tool(self, name, func, dis, form, config: AgentConfig):

        self.print(f"ADDING TOOL:{name} to {config.name}")

        config.tools.update({name: {"func": func, "description": dis, "format": form}})

    def get_default_agent_config(self, name) -> AgentConfig():
        config = AgentConfig()
        config.name = name

        def run_agent(agent_name, text, mode_over_lode: bool or str = False):
            if agent_name:
                return self.run_agent(agent_name, text, mode_over_lode=mode_over_lode)
            return "Provide Information in The Action Input: fild or function call"

        def search_text(x):
            responses = ddg(x, region='wt-wt', safesearch='Off', time='y')
            qa = ddg_answers(x, related=True)
            response = ""

            if responses:
                for res in responses[:4]:
                    response += f"\ntitle:{res['title']}\nhref:{res['href']}\n" \
                                f"body:{self.mas_text_summaries(res['body'], min_length=600)}\n\n"

            if qa:
                for res in qa[:4]:
                    response += f"\nurl:{res['url']}\n" \
                                f"text:{self.mas_text_summaries(res['text'], min_length=600)}\n\n"
            if len(response) == 0:
                return "No data found"
            return response

        def search_news(x):
            responses = ddg_news(x, region='wt-wt', safesearch='Off', time='d', max_results=5)

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
            x = self.get_context_memory().get_context_for(x)
            if x:
                return x
            return "No relavent memory available"

        if name == "self":
            self.config["self_agent_agents_"] = ["todolist"]

            config.mode = "free"
            config.agent_tye = "gpt-3.5-turbo"
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
                "todolist": {"func": lambda x: run_agent('todolist', x),
                             "description": "Run agent to crate a todo list "
                                            "for a given project provide detaild informatino. and a task to do"
                    , "format": "todolist(<task>)"},
                "memory_search": {"func": lambda x: memory_search(x),
                                  "description": "Serch for simmilar memory imput <context>"},
                "search_web": {"func": lambda x: run_agent('search', x),
                               "description": "Run agent to search the web for nested informations"
                    , "format": "search(<task>)"},
                "programming": {"func": lambda x: run_agent('code', x),
                                "description": "Run agent to generate code input Task"
                    , "format": "programming(<task>)"},

                "image-generator": {"func": lambda x: image_genrating_tool(x, self.app_),
                                    "description": "Run to generate image"
                    , "format": "reminder(<detaild_discription>)"},

            }

        if name == "todolist":

            def priorisirung(x):
                config.step_between = "take time to gently think about the problem and consider the whole picture."
                if len(config.task_list) != 0:
                    config.step_between = config.task_list[0]
                return run_agent(config.name, x, mode_over_lode="talk")

            config. \
                set_model_name("text-davinci-003"). \
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
            config.model_name = "text-davinci-003"

            config.agent_type = "zero-shot-react-description"
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
                "browse_url": {"func": lambda x: browse_url(x),
                               "description": "browse web page via URL syntax <url>|<qustion>"},
                "search_text": {"func": lambda x: search_text(x),
                                "description": "Use Duck Duck go to search the web systax <qustion>"},
                "search_news": {"func": lambda x: search_news(x),
                                "description": "Use Duck Duck go to search the web for new get time"
                                               "reladet data systax <qustion>"}
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
                .set_model_name("gpt-4") \
                .set_max_iterations(1) \
                .set_completion_mode("chat")

            config.stop_sequence = ["\n\n\n"]

        if name == "thinkm":
            config. \
                set_mode("free") \
                .set_model_name("gpt-3.5-turbo") \
                .set_max_iterations(1) \
                .set_completion_mode("chat")

            config.stop_sequence = ["\n\n\n"]

        if name == "code":
            config. \
                set_mode("free") \
                .set_model_name("code-davinci-edit-001") \
                .set_max_iterations(3) \
                .set_completion_mode("edit")

        if name.startswith("chain_search"):

            config.mode = "tools"
            config.model_name = "text-davinci-003"

            config.agent_type = "self-ask-with-search"
            config.max_iterations = 3

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
            path += config.name + ".mem"
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

        if self.global_stream_override:
            config.stream = True

        return config

    def remove_agent_config(self, name):
        del self.config[f'agent-config-{name}']
        self.config["agents-name-lsit"].remove(name)

    def get_agent_config_class(self, agent_name="Normal") -> AgentConfig():

        if "agents-name-lsit" not in self.config.keys():
            self.config["agents-name-lsit"] = []

        if agent_name not in self.config["agents-name-lsit"]:
            self.config["agents-name-lsit"].append(agent_name)
            config = self.get_default_agent_config(agent_name)
            self.config[f'agent-config-{agent_name}'] = config

            self.print(f"\nINIT AGENT: {agent_name}:{config.name} {config.mode}\n")

            return config

        if f'agent-config-{agent_name}' in self.config.keys():
            config = self.config[f'agent-config-{agent_name}']
            self.print(f"Using AGENT: {config.name} {config.mode}\n")
        else:
            config = AgentConfig()

        return config

    @staticmethod
    def process_completion(text, config=AgentConfig()):

        if len(config.task_list) == 0:
            config.step_between = text

        model_name = config.model_name
        ret = ""
        if config.stream:
            ret = {'choices': [{'text': "", 'delta': {'content': ''}}]}

        if config.completion_mode == 'text':

            if model_name.startswith('gpt-'):
                model_name = "text-davinci-003"

            ret = openai.Completion.create(
                model=model_name,
                prompt=config.prompt,
                # max_tokens=config.token_left,
                temperature=config.temperature,
                n=1,
                stream=config.stream,
                logprobs=3,
                stop=config.stop_sequence,
            )

            if not config.stream:
                ret = ret.choices[0].text

        elif config.completion_mode == 'chat':
            if text:
                config.add_message("user", text)
            messages = config.get_messages(create=True)
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

    def test_use_tools(self, agent_text, config=AgentConfig()) -> tuple[bool, str, str]:
        if not agent_text:
            return False, "", ""

        if config.mode == "free":

            for text in agent_text.split('\n'):

                if text.startswith("Execute:"):
                    text = text.replace("Execute:", "").strip()
                    for key, value in config.tools.items():
                        if key + '(' in text:
                            return True, key, text

                if text.startswith("Action:"):
                    text = text.replace("Action:", "").strip()
                    for key, value in config.tools.items():
                        if key + '(' in text:
                            return True, key, text

        if config.mode == "execution":

            tool_name = ""
            inputs = ""
            valid_tool = False
            run_key = ""
            lines = agent_text.split('\n')
            i = 0
            for text in lines:
                print('testing line', text, text.startswith("Action:"), valid_tool)
                if text.startswith("Action:") and not valid_tool:
                    tool_name = text.replace("Action:", "").strip()
                    self.print(Style.GREYBG(f"{tool_name}, {config.tools.keys()}, {valid_tool}"))
                    for key in config.tools.keys():
                        if tool_name in key or tool_name.startswith(key) or tool_name.endswith(key):
                            valid_tool = True
                            run_key = key
                if not valid_tool:
                    i += 1
                if text.startswith("Inputs:"):
                    print('Get inputs', text)
                    inputs = text.replace("Inputs:", "")

            if valid_tool:
                self.print(Style.GREYBG(f"{len(lines)}, {config.name}, {valid_tool}"))
                if len(lines) == i:
                    return True, run_key, inputs
                return True, run_key, ",".join(lines[i:])

        res = self.question_answering("What is the name of the action ?", agent_text)
        if res['score'] > 0.3:
            pos_actions: list[str] = res['answer'].replace("\n", "").split(' ')
            items = set(config.tools.keys())
            print(agent_text[res['end']:].split('\n'))
            text_list = agent_text[res['end']:].split('\n')
            text = text_list[0]
            if len(text_list) > 1:
                text = text_list[1]
            for i in text_list:
                if i:
                    text = i
            for p_ac in pos_actions:
                if p_ac in items:
                    print()
                    self.print(f"AI Execute Action {p_ac} {text}|")
                    return True, p_ac, text.replace('Inputs:', '')

        return False, "", ""

    @staticmethod
    def test_task_done(agent_text, config=AgentConfig()):

        done = False

        if not ":" in agent_text:
            done = True

        for line in agent_text.split("\n"):

            if line.startswith("Answer:"):
                done = True

            if line.startswith("Thought: I now know the final answer"):
                done = True

        return done

    def run_tool(self, command, function_name, config=AgentConfig()):

        args = command.replace(function_name + "(", "").replace(function_name, "").split(",")

        if not function_name in config.tools.keys():
            self.print(f"Unknown Function {function_name} valid ar : {config.tools.keys()}")
            return f"Unknown Function {function_name} valid ar : {config.tools.keys()}"

        tool = config.tools[function_name]

        sig = signature(tool['func'])
        args_len = len(sig.parameters)
        self.print(f"Runing : {function_name}")
        self.print(
            f"signature : {sig} | fuction args len : {args_len} | providet nubers of args {len(args)}")

        observation = "Problem running function"

        if args_len == 0:
            observation = tool['func']()

        if args_len == len(args):
            observation = tool['func'](*args)

        if args_len == 1 and len(args) > 1:
            observation = tool['func'](",".join(args))

        path = self.observation_term_mem_file
        if not self.agent_collective_senses:
            path += config.name + ".mem"
        else:
            path += 'CollectiveObservationMemory.mem'

        with open(path, "w") as f:
            try:
                f.write(str(observation))
            except UnicodeEncodeError:
                self.print("Memory not encoded properly")

        if isinstance(observation, dict):
            # try:
            observation = self.summarize_dict(observation, config)

        if not observation or observation == None:
            observation = "Problem running function try run with mor detaiels"

        config.short_mem.text = observation

        return observation

    def run_agent(self, name: str or AgentConfig, text: str, mode_over_lode: str or None = None):
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
        system = ""
        self.logger.info(f"stream mode: {config.stream} mode : {self.config[f'agent-config-{name}'].mode}")
        if self.config[f'agent-config-{name}'].mode == "talk":
            sto, config.add_system_information = config.add_system_information, False
            prompt = PromptTemplate(
                input_variables=["input"],
                template=config.prompt,
            )
            out = LLMChain(prompt=prompt, llm=self.get_OpenAI_models(config.model_name)).run(text)
            config.add_system_information = sto
        elif self.config[f'agent-config-{name}'].mode == "tools":
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
                # def ovservation(x):
                #    out = config.tools[tool_name]["func"](x)
                #    config.obser_mem.text = out
                #    return out
                # fuc = lambda x: ovservation(x)

                tools.append(
                    AgentTool(
                        name=tool_name,
                        func=config.tools[tool_name]["func"],
                        description=config.tools[tool_name]["description"]
                    )
                )

            out = initialize_agent(tools, prompt=prompt,
                                   llm=self.get_OpenAI_models(config.model_name),
                                   agent=config.agent_type, verbose=config.verbose,
                                   return_intermediate_steps=True,
                                   max_iterations=config.max_iterations)(text)

            out = self.summarize_dict(out, config)
            config.add_system_information = sto
        elif self.config[f'agent-config-{name}'].mode == "conversation":
            sto, config.add_system_information = config.add_system_information, False
            prompt = PromptTemplate(
                input_variables=["input"],
                template=config.prompt,
            )
            out = LLMChain(prompt=prompt, llm=self.get_OpenAI_models(config.model_name)).run(text)
            config.add_system_information = sto
        elif self.config[f'agent-config-{name}'].mode in ["execution", 'free']:
            self.logger.info(f"stream mode: {config.stream}")
            out = self.stream_read_line_llm(text, config)
            if not config.stream:
                self.print_stream("execution-free : " + out)
            print()
            config.short_mem.text = out
            t0 = time.time()
            self.logger.info(f"analysing repose")
            self.logger.info(f"analysing test_use_tools")
            with Spinner('analysing repose', symbols='+'):
                use_tool, name, command_ = self.test_use_tools(out, config)
            self.logger.info(f"analysing test_task_done")
            task_done = self.test_task_done(out, config)
            self.logger.info(f"don analysing repose in t-{time.time() - t0}")
            self.print(f"Using-tools: {name} {command_}")
            if use_tool:
                ob = self.run_tool(command_, name, config)
                config.observe_mem.text = ob
                out += "\nObservation: " + ob
            if task_done:  # new task
                # self.speek("Ist die Aufgabe abgeschlossen?")
                if config.short_mem.tokens > 50:
                    with Spinner('Saving work Memory', symbols='t'):
                        config.short_mem.clear_to_collective()
                if len(config.task_list) > config.task_index:
                    config.next_task()
        else:
            out = self.stream_read_line_llm(text, config)
            if not config.stream:
                self.print_stream(out)

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

        config.short_mem.text = f"\n\n{config.name} RESPONSE:\n{out}\n\n"

        if mode_over_lode:
            mode_over_lode, config.mode = config.mode, mode_over_lode

        return out

    def execute_thought_chain(self, user_text: str, agent_tasks, config: AgentConfig, speak=lambda x: x):
        chain_ret = []
        chain_data = {}
        uesd_mem = {}
        ret = ""
        steps = 0

        default_mode_ = config.mode
        default_completion_mode_ = config.completion_mode
        sto_config = None
        chain_mem = self.get_context_memory()
        for task in agent_tasks:

            config.mode = "free"
            config.completion_mode = "chat"

            sum_sto = ""

            keys = list(task.keys())

            task_name = task["name"]
            use = task["use"]
            args = task["args"].replace("$user-input", user_text)

            if use == 'agent':
                sto_config, config = config, self.get_agent_config_class(task_name)

            self.print(f"Running task {args}")

            default_mode = config.mode
            default_completion_mode = config.completion_mode

            if 'mode' in keys:
                config.mode = task['mode']
            if 'completion-mode' in keys:
                config.completion_mode = task['completion-mode']

            if "infos" in keys:
                config.short_mem.text += task['infos']

            chain_data['$edit-text-mem'] = config.edit_text.text

            speak(f"Chain running {task_name} at step {steps} with the input : {args}")

            if 'chuck-run-all' in keys:

                for chunk in chain_data[task['chuck-run-all']]:

                    if not chunk:
                        continue

                    args_ = args.replace(task['chuck-run-all'], str(chunk))

                    if use == "tool":
                        ret = self.run_tool(args_, task_name, config)

                    if use == "agent":
                        if config.mode == 'free':
                            config.task_list.append(args_)
                        ret = self.run_agent(config, args_, mode_over_lode=config.mode)

                    if use == 'function':
                        if 'function' in keys:
                            if callable(task['function']) and chain_ret:
                                task['function'](chain_ret[-1][1])

                    if 'short-mem' in keys:
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

            elif 'chuck-run' in keys:
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
                for chunk_vec in rep:
                    ret_ = ''
                    if not chunk_vec:
                        continue

                    chunk = chain_mem.hydrate_vectors(uesd_mem[task['chuck-run']], chunk_vec)

                    args_ = args.replace(task['chuck-run'], str(chunk[0].page_content))

                    if use == "tool":
                        ret_ = self.run_tool(args_, task_name, config)

                    if use == "agent":
                        if config.mode == 'free':
                            config.task_list.append(args_)
                        ret_ = self.run_agent(config, args_, mode_over_lode=config.mode)

                    if use == 'function':
                        if 'function' in keys:
                            if callable(task['function']) and chain_ret:
                                ret_ = task['function'](chain_ret[-1][1])

                    ret_chunk.append(ret_)

                    if 'short-mem' in keys:
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
                continue

            else:
                for c_key in chain_data.keys():
                    if c_key in args:
                        args = args.replace(c_key, chain_data[c_key])
                if use == "tool":
                    ret = self.run_tool(args, task_name, config)
                if use == "agent":
                    if config.mode == 'free':
                        config.task_list.append(args)
                    ret = self.run_agent(config, args + " Answer: Let's work this out in a step by step " \
                                                        "way to be sure we have the right answer")
                if use == 'function':
                    if 'function' in keys:
                        if callable(task['function']) and chain_ret:
                            task['function'](chain_ret[-1][1])

                if 'short-mem' in keys:
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

            if 'validate' in keys:
                self.print("Validate task")
                try:
                    pipe_res = self.text_classification(ret)
                    self.print(f"Validation :  {pipe_res[0]}")
                    if pipe_res[0]['score'] > 0.8:
                        if pipe_res[0]['label'] == "NEGATIVE":
                            print('🟡')
                            if 'on-error' in keys:
                                if task['validate'] == 'inject':
                                    task['inject'](ret)
                                if task['validate'] == 'return':
                                    task['inject'](ret)
                                    chain_ret.append([task, ret])
                                    return "an error occurred", chain_ret
                        else:
                            print(f'🟢')
                except Exception as e:
                    print(f"Error in validation : {e}")
            if 'to-edit-text' in keys:
                config.edit_text.text = ret

            speak(f"Step {steps} with response {ret}")

            if "return" in keys:
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

            self.print(Style.ITALIC(Style.GREY(f'Chain at step : {steps}\nreturned : {str(ret)[:150]}...')))

            chain_ret.append([task, ret])

            steps += 1
            if sto_config:
                config = sto_config
                sto_config = None

            config.mode = default_mode
            config.completion_mode = default_completion_mode

        chain_sum_data = self.summarize_ret_list(chain_ret)

        config.mode = default_mode_
        config.completion_mode = default_completion_mode_

        return self.run_agent(self.get_agent_config_class("think"),
                              f"Produce a verbal output and summarization of what happened for"
                              f" the user (max 1 paragraph) using the given information {chain_sum_data}"
                              f" validate if the task was executed successfully if u see an max iteration"
                              f" limit &or error the task failed"), chain_ret

    def execute_2tree(self, user_text, tree, config: AgentConfig):
        config.binary_tree = tree
        config.stop_sequence = ["\n\n\n", "N/A", 'n/a']
        alive = True
        res_um = 'Plan for The Task:'
        res = ''
        while alive:
            input("Enter")
            res = self.run_agent(config, user_text, mode_over_lode='q2tree')
            tree_depth = config.binary_tree.get_depth(config.binary_tree.root)
            don, next_on, speak = False, 0, res
            str_ints_list_to = list(range(tree_depth + 1))
            for line in res.split("\n"):
                if line.startswith("Answer"):
                    print(F"detected LINE:{line[:10]}")
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
                new_tree = config.binary_tree.cut_tree('L' * next_on)
                config.binary_tree = new_tree

        return res, res_um

    def stream_read_line_llm(self, text, config, r=2.0):
        p_token_num = len(tiktoken.encoding_for_model(config.model_name).encode(text))
        self.print(f"TOKENS: {p_token_num} | left = {config.token_left}")
        if p_token_num > config.token_left:
            text = self.mas_text_summaries(text)
        try:
            if not config.stream:
                with Spinner(
                    f"Generating response {config.name} {config.model_name} {config.mode} {config.completion_mode}"):
                    res = self.process_completion(text, config)
                return res

            print(f"Generating response (/) stream (\\) {config.name} {config.model_name} {config.mode} "
                  f"{config.completion_mode}")
            min_typing_speed, max_typing_speed, res = 0.02, 0.01, ""
            try:
                for line in self.process_completion(text,
                                                    config):  # openai.error.InvalidRequestError if error try  is_text_completion = False
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

                    for i, word in enumerate(ai_text):
                        self.print_stream(word, end="", flush=True)
                        typing_speed = random.uniform(min_typing_speed, max_typing_speed)
                        time.sleep(typing_speed)
                        # type faster after each word
                        min_typing_speed = min_typing_speed * 0.07
                        max_typing_speed = max_typing_speed * 0.06
                    res += str(ai_text)
            except requests.exceptions.ChunkedEncodingError as ex:
                print(f"Invalid chunk encoding {str(ex)}")
                self.print(f"\t\t| Retry level: {r} ", end="\r")
                with Spinner("ChunkedEncodingError", symbols='c'):
                    time.sleep(2 * (3 - r))
                if r > 0:
                    print('\n\n')
                    return self.stream_read_line_llm(text + '\n' + res, config, r - 1)
            return res
        except openai.error.RateLimitError:
            self.print(f"\t\t| Retry level: {r} ", end="\r")
            if r > 0:
                self.logger.info(f"Waiting {5 * (8 - r)} seconds")
                with Spinner("Waiting RateLimitError", symbols='+'):
                    time.sleep(5 * (8 - r))
                self.print(f"\n Retrying {r} ", end="\r")
                return self.stream_read_line_llm(text, config, r - 1)
            else:
                self.logger.error("The server is currently overloaded with other requests. Sorr   y about that!")
                return "The server is currently overloaded with other requests. Sorry about that!"

        except openai.error.InvalidRequestError:
            self.print(f"{' ' * 30}| Retry level: {r} ", end="\r")
            with Spinner("Waiting InvalidRequestError", symbols='b'):
                time.sleep(2)
            if r > 1.5:
                if config.short_mem.tokens > config.edit_text.tokens:
                    config.short_mem.max_length = int(config.short_mem.max_length * 0.45)
                    config.short_mem.cut()
                if config.short_mem.tokens < config.edit_text.tokens:
                    config.edit_text.max_length = int(config.edit_text.max_length * 0.75)
                    config.edit_text.cut()
                return self.stream_read_line_llm(text, config, r - 0.5)
            elif r > .75:
                return self.stream_read_line_llm(self.mas_text_summaries(text), config, r - 0.25)
            elif r > 0.25:
                config.stream = False
                res = self.stream_read_line_llm(self.mas_text_summaries(text), config, r - 0.25)
                config.stream = True
                return res
            else:
                self.logger.error("The server is currently overloaded with other requests. Sorry about that!")
                return "The System cannot correct the text input for the agent."
        return "No Output providet"

    def mas_text_summaries(self, text, min_length=1600):
        len_text = len(text)
        if len_text < min_length:
            return text

        cap = 800
        max_length = 45
        summary_chucks = ""
        chucks = []

        if len(text) >= 6200:
            cap = 1200
            max_length = 80

        if len(text) >= 10200:
            cap = 2000
            max_length = 160

        if len(text) >= 70200:
            cap = 2500
            max_length = 412

        summary = ""

        def summary_func(x):
            return self.summarization(x, max_length=max_length)

        num_tokens = cap / 4.9
        if num_tokens > 1023:
            cap = int(cap / (num_tokens / 1023))

        while len(text) > cap:
            chucks.append(text[:cap])
            text = text[cap:]
            self.print(f"SYSTEM: TEXT len : {len(text)}")

        if len(text) < max_length:
            chucks[-1] += "\n" + text
        else:
            chucks.append(text)
        self.print(f"SYSTEM: chucks to summary: {len(chucks)} cap : {cap}")

        with Spinner("Generating summary", symbols='d'):
            summaries = summary_func(chucks)

        for i, chuck_summary in enumerate(summaries):
            summary_chucks += chuck_summary['summary_text'] + "\n"
            self.print(f"SYSTEM: all summary_chucks : {len(summary_chucks)}")

        summary = summary_chucks
        if len(summaries) > 8:
            if len(summary_chucks) < 100000:
                summary = summary_func(summary_chucks)[0]['summary_text']
            else:
                summary = self.mas_text_summaries(summary_chucks)
        self.print(
            f"SYSTEM: final summary from {len_text}:{len(summaries)} -> {len(summary)} compressed {len_text / len(summary):.2f}X\n")

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
        for i, step in enumerate(ret_list):
            if isinstance(step, list):
                step_content = f"\naction {i}" + str(step[0]['use']) + ": " + str(step[0]['name'])
                step_content += f"\ntool_input {i} " + str(step[0]['args'])
                if step[1]:  # assuming the second item in the list is always of interest
                    step_content += f"\nlog {i} " + str(step[1][0])  # assuming the list contains a string
                chucs.append(self.mas_text_summaries(step_content))  # assuming this function exists
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
            return ConversationalRetrievalChain.from_llm(self.get_OpenAI_models(config.model_name), retriever=retriever)
        return None

    @staticmethod
    def get_chain(hydrate=None, f_hydrate=None) -> AgentChain:
        logger = get_logger()
        logger.info(Style.GREYBG(f"AgentChain requested"))
        agent_chain = AgentChain(hydrate, f_hydrate)
        logger.info(Style.Bold(f"AgentChain instance, returned"))
        return agent_chain

    @staticmethod
    def get_context_memory(model_name=None) -> AIContextMemory:
        logger = get_logger()
        logger.info(Style.GREYBG(f"AIContextMemory requested"))
        if model_name:
            cm = AIContextMemory(model_name)
        else:
            cm = AIContextMemory()
        logger.info(Style.Bold(f"AIContextMemory instance, returned"))
        return cm


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


def initialize_gi(app: App, model_name):
    app.logger.info(f'initializing gi {model_name}')
    mod = get_tool(app)
    mod.config['genrate_image-init-in'] = model_name

    if not mod.config['genrate_image-init']:
        mod.config[f'replicate'] = replicate.Client(api_token=mod.config[f'REPLICATE_API_TOKEN'])

    mod.config['genrate_image-init'] = True

    # mod.config[f'genrate_image{model_name}'] = StableDiffusionPipeline.from_pretrained(model_name, revision="fp16",
    #                                                                                    torch_dtype=torch.float16)
    # mod.config[f'genrate_image{model_name}'].scheduler = DPMSolverMultistepScheduler.from_config(
    #     mod.config[f'genrate_image{model_name}'].scheduler.config)

    model = mod.config[f'replicate'].models.get(model_name)
    mod.config[f'genrate_image{model_name}'] = model.versions.get(
        "db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf")


def genrate_image(inputs, app: App, model="stability-ai/stable-diffusion"):
    mod = get_tool(app)
    if not mod.config['genrate_image-init']:
        initialize_gi(app, model)
    if 'genrate_image-in' not in mod.config.keys():
        mod.config['genrate_image-in'] = model

    if mod.config['genrate_image-in'] != model:
        initialize_gi(app, model)

    return mod.config[f'genrate_image{model}'].predict(**inputs)  # (text).images


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
    summary = summarize(text)
    return """ "Result" : """ + summary


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
