# !pip -q install git+https://github.com/huggingface/transformers # need to install from github
# !pip install -q datasets loralib sentencepiece
# !pip -q install bitsandbytes accelerate
# !pip -q install langchain

import logging

from toolboxv2 import MainTool, FileHandler

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.1"
        self.name = "isaa_alpaca"
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
                    ["runE", "Starts Inference"],
                    ["runCov", "Starts Inference"],
                    ],
            "name": "isaa",
            "Version": self.show_version,
        }
        self.llm_chain = None
        self.conversation = None
        self.tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")

        self.base_model = LlamaForCausalLM.from_pretrained(
            "chavinlo/alpaca-native",
            device_map='auto',
            offload_folder="isaa_data/dump_data"
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.base_model,
            tokenizer=self.tokenizer,
            max_length=256,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.2
        )

        self.local_llm = HuggingFacePipeline(pipeline=self.pipe)

        FileHandler.__init__(self, "issa.config", app.id if app else __name__)
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=None, color=self.color, on_exit=self.on_exit)

    def show_version(self):
        self.print("Version: ", self.version)

    def on_start(self):
        self.load_file_handler()
        config = self.get_file_handler(self.keys["Config"])
        self.logger.info("simpel alpaca online")
        if config is not None:
            self.config = eval(config)

    def on_exit(self):
        self.add_to_save_file_handler(self.keys["Config"], str(self.config))
        self.save_file_handler()
        self.file_handler_storage.close()

    def set_up(self, config):
        if not config:
            self.config = {

            }
        self.config = config


