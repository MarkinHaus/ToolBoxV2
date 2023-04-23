from typing import List

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

import logging

from toolboxv2 import MainTool, FileHandler

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain


def ez_chain(llm):
    return load_qa_chain(llm, chain_type="stuff")


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.1"
        self.name = "isaa_pinecone"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "VIOLETBG"
        self.config = {}
        self.isaa_instance = {"Stf": {},
                              "DiA": {}}
        self.keys = {
            "KEY": "key~~~~~~~",
            "Config": "config~~~~"
        }
        self.tools = {
            "all": [["Version", "Shows current Version"],

                    ],
            "name": "isaa",
            "Version": self.show_version,

        }
        self.pinecone = None
        self.get_chain = ez_chain
        FileHandler.__init__(self, "issa.config", app.id if app else __name__)
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=None, color=self.color, on_exit=self.on_exit)

    def show_version(self):
        self.print("Version: ", self.version)

    def on_start(self):
        self.load_file_handler()
        config = self.get_file_handler(self.keys["Config"])
        self.logger.info("simpel pinecone online")
        self.pinecone.init(
            api_key=self.config["PINECONE_API_KEY"],  # find at app.pinecone.io
            environment=self.config["PINECONE_API_ENV"]  # next to api key in console
        )
        if config is not None:
            self.config = eval(config)

    def on_exit(self):
        self.add_to_save_file_handler(self.keys["Config"], str(self.config))
        self.save_file_handler()
        self.file_handler_storage.close()

    def build_generator_default_open_ai(self, command: List[str, str, str, Embeddings]):
        if len(command) != 4:
            raise ValueError("Command must contain str name embeddings")

        self.config[command[0]] = {
            "chunk_size": int(command[1]),
            "chunk_overlap": int(command[2]),
            "embeddings": OpenAIEmbeddings(openai_api_key=command[3]),
            "step_between": lambda x: x,
            "index_name": command[0],
            "docs": None
        }

    def generate_docs_from_loader(self, command: List[str, BaseLoader]):
        if len(command) != 2:
            raise ValueError("Command must contain str name loader")

        if command[2] not in self.config.keys():
            return None

        chunk_size = self.config[command[0]]["chunk_size"]
        chunk_overlap = self.config[command[0]]["chunk_overlap"]

        data = self.config[command[1]].load()

        self.print(f'You have {len(data)} document(s) in your data')
        self.print(f'There are {len(data[0].page_content)} characters in your document')

        texts = [self.config[command[0]]["step_between"](text)
                 for text in RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                            hunk_overlap=chunk_overlap)
                 .split_documents(data)]

        self.print(f'Now you have {len(texts)} documents')

        if self.config[command[0]]["docs"]:
            self.config[command[0]]["docs"] = Pinecone.add_texts(self.config[command[0]]["docs"],
                                                                 [t.page_content for t in texts])
        else:
            self.config[command[0]]["docs"] = Pinecone.from_texts([t.page_content for t in texts],
                                                                  self.config[command[0]]["embeddings"],
                                                              index_name=self.config[command[0]]["index_name"])

    def similarity_search(self, command: List[str, str]):

        if len(command) != 2:
            return False
        if command[0] not in self.config.keys():
            return False
        if not self.config[command[0]]["docs"]:
            return False

        return self.config[command[0]]["docs"].similarity_search(command[1], include_metadata=True)


    def crate_cone(self, command: List[str]):
        pass
