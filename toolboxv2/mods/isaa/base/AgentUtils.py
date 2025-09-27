import json
import locale
import os
import pickle
import platform
import re
import subprocess
import sys
import threading

import requests
import tiktoken

from toolboxv2 import Singleton, get_logger, remove_styles
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


def get_location():
    ip_address = get_ip()
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_data = f"city: {response.get('city')},region: {response.get('region')},country: {response.get('country_name')},"

    return location_data, ip_address


import shutil
import tempfile
from pathlib import Path


def detect_shell() -> tuple[str, str]:
    """
    Detects the best available shell and the argument to execute a command.
    Returns:
        A tuple of (shell_executable, command_argument).
        e.g., ('/bin/bash', '-c') or ('powershell.exe', '-Command')
    """
    if platform.system() == "Windows":
        if shell_path := shutil.which("pwsh"):
            return shell_path, "-Command"
        if shell_path := shutil.which("powershell"):
            return shell_path, "-Command"
        return "cmd.exe", "/c"

    shell_env = os.environ.get("SHELL")
    if shell_env and shutil.which(shell_env):
        return shell_env, "-c"

    for shell in ["bash", "zsh", "sh"]:
        if shell_path := shutil.which(shell):
            return shell_path, "-c"

    return "/bin/sh", "-c"


def safe_decode(data: bytes) -> str:
    encodings = [sys.stdout.encoding, locale.getpreferredencoding(), 'utf-8', 'latin-1', 'iso-8859-1']
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace')


import asyncio


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


import io
import xml.etree.ElementTree as ET
import zipfile

# Import der PyPDF2-Bibliothek
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


def extract_text_natively(data: bytes, filename: str = "") -> str:
    """
    Extrahiert Text aus verschiedenen Dateitypen mit nativen Python-Methoden
    oder reinen Python-Bibliotheken (speziell PyPDF2 für PDFs).

    Args:
        data (bytes): Der Inhalt der Datei als Bytes.
        filename (str, optional): Der Originaldateiname, um den Typ zu bestimmen.

    Returns:
        str: Der extrahierte Text.

    Raises:
        ValueError: Wenn der Dateityp nicht unterstützt wird oder die Verarbeitung fehlschlägt.
        ImportError: Wenn PyPDF2 für die Verarbeitung von PDF-Dateien benötigt, aber nicht installiert ist.
    """
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ''

    # 1. DOCX-Verarbeitung (nativ mit zipfile und xml)
    if data.startswith(b'PK\x03\x04'):
        try:
            docx_file = io.BytesIO(data)
            text_parts = []
            with zipfile.ZipFile(docx_file) as zf:
                namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                body_path = "word/document.xml"
                if body_path in zf.namelist():
                    xml_content = zf.read(body_path)
                    tree = ET.fromstring(xml_content)
                    for para in tree.iter(f"{namespace}p"):
                        texts_in_para = [node.text for node in para.iter(f"{namespace}t") if node.text]
                        if texts_in_para:
                            text_parts.append("".join(texts_in_para))
                return "\n".join(text_parts)
        except (zipfile.BadZipFile, ET.ParseError):
            pass  # Fährt fort, falls es eine ZIP-Datei, aber kein gültiges DOCX ist

    # 2. PDF-Verarbeitung (mit PyPDF2)
    if data.startswith(b'%PDF-'):
        if PyPDF2 is None:
            raise ImportError(
                "Die Bibliothek 'PyPDF2' wird benötigt, um PDF-Dateien zu verarbeiten. Bitte installieren Sie sie mit 'pip install PyPDF2'.")

        try:
            # Erstelle ein In-Memory-Dateiobjekt für PyPDF2
            pdf_file = io.BytesIO(data)
            # Verwende PdfFileReader aus PyPDF2
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)

            text_parts = []
            # Iteriere durch die Seiten
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                # Extrahiere Text mit extractText()
                page_text = page.extractText()
                if page_text:
                    text_parts.append(page_text)

            return "\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"PDF-Verarbeitung mit PyPDF2 fehlgeschlagen: {e}")

    # 3. Fallback auf reinen Text (TXT)

    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return data.decode('latin-1')
        except Exception as e:
            raise ValueError(f"Text-Dekodierung fehlgeschlagen: {e}")


class AISemanticMemory(metaclass=Singleton):
    def __init__(self,
                 base_path: str = "/semantic_memory",
                 default_model: str = os.getenv("BLITZMODEL"),
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
                       metadata: dict | None = None, direct=False) -> bool:
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
                text = extract_text_natively(data, filename="" if metadata is None else metadata.get("filename", ""))
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
            added, duplicates = await kb.add_data(texts, metadata, direct=direct)
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
                if result.overview:
                    results.append({
                        "memory": name,
                        "result": result
                    })
            #except Exception as e:
            #    print(f"Query failed on {name}: {str(e)}")
        if to_str:
            str_res = ""
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

    def save_all_memories(self, path: str) -> bool:
        """Save all memory stores to disk"""
        for name, kb in self.memories.items():
            try:
                kb.save(os.path.join(path, f"{name}.pkl"))
            except Exception as e:
                print(f"Error saving memory: {str(e)}")
                return False
        return True

    def load_all_memories(self, path: str) -> bool:
        """Load all memory stores from disk"""
        for file in os.listdir(path):
            if file.endswith(".pkl"):
                try:
                    self.memories[file[:-4]] = KnowledgeBase.load(os.path.join(path, file))
                except EOFError:
                    return False
                except FileNotFoundError:
                    return False
                except Exception as e:
                    print(f"Error loading memory: {str(e)}")
                    return False
        return True

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
                transformers = __import__("transformers")
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

                def hugging_tokenize(x):
                    return tokenizer.tokenize(x)

                encode = hugging_tokenize

            except ValueError:
                pass
            except ImportError:
                encode = lambda x: round(len(x)*0.4)
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


import os
from dataclasses import asdict, dataclass, field


@dataclass
class LLMMode:
    name: str
    description: str
    system_msg: str
    post_msg: str | None = None
    examples: list[str] | None = None

    def __str__(self):
        return f"LLMMode: {self.name} (description) {self.description}"

@dataclass
class ModeController(LLMMode):
    shots: list = field(default_factory=list)

    def add_shot(self, user_input, agent_output):
        self.shots.append([user_input, agent_output])

    def add_user_feedback(self):

        add_list = []

        for index, shot in enumerate(self.shots):
            print(f"Input : {shot[0]} -> llm output : {shot[1]}")
            user_evalution = input("Rank from 0 to 10: -1 to exit\n:")
            if user_evalution == '-1':
                break
            else:
                add_list.append([index, user_evalution])

        for index, evaluation in add_list:
            self.shots[index].append(evaluation)

    def auto_grade(self):
        pass

    @classmethod
    def from_llm_mode(cls, llm_mode: LLMMode, shots: list | None = None):
        if shots is None:
            shots = []

        return cls(
            name=llm_mode.name,
            description=llm_mode.description,
            system_msg=llm_mode.system_msg,
            post_msg=llm_mode.post_msg,
            examples=llm_mode.examples,
            shots=shots
        )



@dataclass
class ControllerManager:
    controllers: dict[str, ModeController] = field(default_factory=dict)

    def rget(self, llm_mode: LLMMode, name: str = None):
        if name is None:
            name = llm_mode.name
        if not self.registered(name):
            self.add(name, llm_mode)
        return self.get(name)

    def registered(self, name):
        return name in self.controllers

    def get(self, name):
        if name is None:
            return None
        if name in self.controllers:
            return self.controllers[name]
        return None

    def add(self, name, llm_mode, shots=None):
        if name in self.controllers:
            return "Name already defined"

        if shots is None:
            shots = []

        self.controllers[name] = ModeController.from_llm_mode(llm_mode=llm_mode, shots=shots)

    def list_names(self):
        return list(self.controllers.keys())

    def list_description(self):
        return [d.description for d in self.controllers.values()]

    def __str__(self):
        return "LLMModes \n" + "\n\t".join([str(m).replace('LLMMode: ', '') for m in self.controllers.values()])

    def save(self, filename: str | None, get_data=False):

        data = asdict(self)

        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(data, f)

        if get_data:
            return json.dumps(data)

    @classmethod
    def init(cls, filename: str | None, json_data: str | None = None):

        controllers = {}

        if filename is None and json_data is None:
            print("No data provided for ControllerManager")
            return cls(controllers=controllers)

        if filename is not None and json_data is not None:
            raise ValueError("filename and json_data are provided only one accepted filename or json_data")

        if filename is not None:
            if os.path.exists(filename) and os.path.isfile(filename):
                with open(filename) as f:
                    controllers = json.load(f)
            else:
                print("file not found")

        if json_data is not None:
            controllers = json.loads(json_data)

        return cls(controllers=controllers)
