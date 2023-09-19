from dotenv import load_dotenv

from toolboxv2 import get_logger
from toolboxv2.utils.toolbox import Singleton

# Load environment variables from .env file
load_dotenv()

class OverRideConfig:
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return None  # Oder einen anderen Standardwert zurückgeben

    def __setitem__(self, key, value):
        setattr(self, key, value)

class ConfigIsaa:
    name: str = "isaa"
    genrate_image_in: str = "stability-ai/stable-diffusion"
    genrate_image_init: bool = False
    agents_name_list: list = []
    text_splitter3_init: bool
    text_splitter2_init: bool
    text_splitter1_init: bool
    text_splitter0_init: bool
    WOLFRAM_ALPHA_APPID: str
    HUGGINGFACEHUB_API_TOKEN: str
    OPENAI_API_KEY: str
    REPLICATE_API_TOKEN: str
    IFTTTKey: str
    SERP_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_API_ENV: str
    price: dict



def get_config(config_class):
    return {attr_name: getattr(config_class, attr_name) for attr_name in dir(config_class)}


class Config(metaclass=Singleton):
    def __init__(self):
        self.configs = {}
        self.configs_class = [ConfigIsaa]
        self.configs_ = {ConfigIsaa.name: ConfigIsaa}
        self.scopes = []
        self.scope = ""

    def initialize(self):
        get_logger().info("initialize configs")
        for config_c in self.configs_class:
            if "name" in dir(config_c):
                self.scopes.append(config_c.name)
                self.scope = config_c.name
                self.configs[config_c.name] = get_config(config_c)
                get_logger().info(f"Added {config_c.name}")
            else:
                get_logger().error(f"Error no name attr in : {config_c}")

    def gets(self, index):
        return self.configs[self.scope][index]

    def get(self):
        return self.configs_[self.scope]

    def get_scope(self):
        return self.scope

    def get_scopes(self):
        return self.scopes

    def set(self, index, value):
        self.configs[self.scope][index] = value

    def set_scope(self, scope):
        if scope in self.scopes:
            self.scope = scope
