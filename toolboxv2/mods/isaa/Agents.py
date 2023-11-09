import time
from dataclasses import dataclass, field, asdict
from functools import reduce
from random import uniform
from typing import Optional, List, TypeAlias, NewType, TypeVar, Dict, Callable, Any, Generator

from enum import Enum
from typing import Optional

from litellm.llms.prompt_templates.factory import prompt_factory, llama_2_chat_pt, falcon_chat_pt, falcon_instruct_pt, \
    mpt_chat_pt, wizardcoder_pt, phind_codellama_pt, hf_chat_template, default_pt, ollama_pt
from litellm.utils import trim_messages, check_valid_key, get_valid_models, get_max_tokens

from litellm import longer_context_model_fallback_dict, ContextWindowExceededError, BudgetManager

from .AgentUtils import ShortTermMemory, AIContextMemory, get_token_mini, get_max_token_fom_model_name

import litellm
import logging
from litellm import completion, model_list
from litellm.caching import Cache

import gpt4all

from ..__init__ import get_logger

# litellm.cache = Cache()
litellm.set_verbose = True


def stram_print(text):
    min_typing_speed, max_typing_speed = 0.01, 0.005
    for i, word in enumerate(text):
        if not word:
            continue
        print(word, end="", flush=True)
        typing_speed = uniform(min_typing_speed, max_typing_speed)
        time.sleep(typing_speed)
        min_typing_speed = min_typing_speed * 0.04
        max_typing_speed = max_typing_speed * 0.03


def get_str_response(chunk):
    if isinstance(chunk, dict):
        data = chunk['choices'][0]

        if "delta" in data.keys():
            message = chunk['choices'][0]['delta']
        elif "text" in data.keys():
            message = chunk['choices'][0]['text']
        elif "content" in data['delta'].keys():
            message = chunk['choices'][0]['delta']['content']
        else:
            message = ""

    elif isinstance(chunk, str):
        message = chunk
    else:
        try:
            message = chunk.choices[0].message.content
        except AttributeError:
            message = f"Unknown chunk type {type(chunk)}"
    return message


def add_to_kwargs_if_not_none(**values):
    return {k: v for k, v in values.items() if v}


@dataclass(frozen=True)
class Providers(Enum):
    ANTHROPIC = "Anthropic"
    OPENAI = "OpenAI"
    REPLICATE = "Replicate"
    COHERE = "Cohere"
    HUGGINGFACE = "Huggingface"
    OPENROUTER = "OpenRouter"
    AI21 = "AI21"
    VERTEXAI = "VertexAI"
    BEDROCK = "Bedrock"
    SAGEMAKER = "Sagemaker"
    TOGETHERAI = "TogetherAI"
    ALEPHALPHA = "AlephAlpha"
    PALM = "Palm"
    NLP = "NLP"
    VLLM = "vllm"
    PETALS = "Petals"
    LOCAL = "Local"
    MYAPI = "Myapi"


@dataclass
class LLMTrim(Enum):
    litellm = "LLMTrim"
    isaa = "IsaaTrim"


@dataclass(frozen=True)
class CompletionError(Enum):
    Rate_Limit_Errors = "RateLimitErrors"
    Invalid_Request_Errors = "InvalidRequestErrors"
    Authentication_Errors = "AuthenticationErrors"
    Timeout_Errors = "TimeoutErrors"
    ServiceUnavailableError = "ServiceUnavailableError"
    APIError = "APIError"
    APIConnectionError = "APIConnectionError"


@dataclass(frozen=True)
class LLMFunction:
    name: str
    description: str
    parameters: Dict[str, str]
    function: Optional[Callable[[str], str]]

    def __str__(self):
        return f"----\nname -> '{self.name}'\nparameters -> '{self.parameters}' ->\ndescription -> '{self.description}'"


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class Capabilities:
    name: str
    description: str
    trait: str
    functions: Optional[List[LLMFunction]]

    # TODO: use a agent to combine capabilities


@dataclass
class LLMMode:
    name: str
    description: str
    system_msg: str
    post_msg: Optional[str] = None
    examples: Optional[List[str]] = None


@dataclass(frozen=True)
class AgentPromptData:
    initial_prompt_value: Optional[str]
    final_prompt_value: Optional[str]

    system_pre_message: Optional[str]
    system_post_message: Optional[str]

    user_pre_message: Optional[str]
    user_post_message: Optional[str]

    assistant_pre_message: Optional[str]
    assistant_post_message: Optional[str]


@dataclass(frozen=True)
class AgentModelData:
    name: str = field(default=None, hash=True)
    model: str = field(default=None)
    provider: Providers = field(default=Providers.OPENAI)
    system_message: str = field(default="")

    temperature: Optional[int] = field(default=None)
    top_k: Optional[int] = field(default=None)
    top_p: Optional[int] = field(default=None)
    repetition_penalty: Optional[int] = field(default=None)

    repeat_penalty: Optional[int] = field(default=None)
    repeat_last_n: Optional[float] = field(default=None)
    n_batch: Optional[int] = field(default=None)

    api_key: Optional[str] = field(default=None)
    api_base: Optional[str] = field(default=None)
    api_version: Optional[str] = field(default=None)
    user_id: Optional[str] = field(default=None)

    fallbacks: Optional[List[Dict[str, str]] or List[str]] = field(default=None)
    stop_sequence: Optional[List[str]] = field(default=None)
    budget_manager: Optional[BudgetManager] = field(default=None)
    caching: Optional[bool] = field(default=None)


def get_free_agent_data_factory(name="Gpt4All", model="ggml-model-gpt4all-falcon-q4_0.bin") -> AgentModelData:
    return AgentModelData(
        name=name,
        model=model,
        provider=Providers.LOCAL,
        stop_sequence=["[!X!]"],
    )


def flatten_reduce_lambda(matrix):
    return list(reduce(lambda x, y: x + y, matrix, []))



@dataclass()
class Agent:
    amd: AgentModelData = field(default_factory=get_free_agent_data_factory)

    stream: bool = field(default=False)
    messages: List[Dict[str, str]] = field(default_factory=list)
    trim: LLMTrim = field(default=LLMTrim.litellm)
    verbose: bool = field(default=False)
    completion_with_config: bool = field(default=False)
    batch_completion: bool = field(default=False)
    stream_function: Callable[[str], None] = field(default_factory=print)

    max_tokens: Optional[float] = field(default=None)
    tasklist: Optional[List[str]] = field(default=None)
    task_index: Optional[int] = field(default=None)

    functions: Optional[List[LLMFunction]] = field(default=None)
    config: Optional[Dict[str, Any]] = field(default=None)

    batch_completion_messages: Optional[List[List[LLMMessage]]] = field(default=None)

    memory: Optional[AIContextMemory] = field(default=None)
    content_memory: Optional[ShortTermMemory] = field(default=None)

    capabilities: Optional[Capabilities] = field(default=None)
    mode: Optional[LLMMode] = field(default=None)

    last_result: Optional[Dict[str, Any]] = field(default=None)

    locale: bool = field(default=False)
    model: Optional[gpt4all.GPT4All] = field(default=None)
    hits: Optional[str] = field(default=None)

    def logger(self, log):
        print(log)
        pass

    def check_valid(self):

        if self.amd.provider.name == Providers.LOCAL.name:
            return True

        response = check_valid_key(model=self.amd.model, api_key=self.amd.api_key)

        if not response:
            print(f"Agent failed {self.amd.name}")
        return response

    def construct_first_msg(self, message: List[Dict[str, str]]) -> List[Dict[str, str]]:
        llm_prompt = self.amd.system_message
        if self.capabilities:
            llm_prompt += self.capabilities.trait

            if self.capabilities.functions:
                functions_infos = "\n".join([str(functions) for functions in self.capabilities.functions])
                message.append(asdict(LLMMessage("system", f"Callable functions:\n{functions_infos}\n--+--\n")))

        if self.mode:
            llm_prompt += self.mode.system_msg

            if self.mode.examples:
                llm_prompt += "\nExamples: \n" + '-----\n' + "\n---\n".join(self.mode.examples) + '\n---\n'

        if llm_prompt:
            message.append(asdict(LLMMessage("system", llm_prompt)))

        return message

    def get_llm_message(self, task, persist: Optional[bool] = None, fetch_memory: Optional[bool] = None,
                        isaa=None):

        llm_message = []
        memory_task = ""
        memory_context = ""

        if fetch_memory and self.memory is not None and self.content_memory is not None:
            memory_task = self.memory.get_context_for(task)
            memory_context = self.memory.get_context_for(self.content_memory.text)

        if not persist:
            llm_message = self.construct_first_msg(llm_message)
            if memory_context:
                llm_message.append(asdict(LLMMessage("system", "History summary:" + memory_context)))
            if memory_task:
                llm_message.append(asdict(LLMMessage("system", "Additional memory Informations:" + memory_task)))

        if persist:

            if not self.messages:
                self.messages = self.construct_first_msg([])

            llm_message = self.messages.copy()
            self.messages.append(asdict(LLMMessage("user", task)))

        llm_message.append(asdict(LLMMessage("user", task)))

        if self.mode is not None:
            if self.mode.post_msg is not None:
                llm_message.append(asdict(LLMMessage("system", self.mode.post_msg)))

        if self.trim.litellm:
            new_msg = trim_messages(llm_message, self.amd.model)
            if new_msg:
                llm_message = new_msg

        if self.trim.isaa and isaa:

            def get_tokens_estimation(text, only_len=True):
                if isinstance(text, list):
                    text = '\n'.join(msg['content'] for msg in text)
                tokens = get_token_mini(text, self.amd.model, isaa, only_len)
                if only_len:
                    if tokens == 0:
                        tokens = int(len(text) * (3 / 4))
                return tokens

            llm_message = isaa.short_prompt_messages(llm_message, get_tokens_estimation,
                                                     get_max_token_fom_model_name(self.amd.model))

        return llm_message

    def completion(self, llm_message, **kwargs):

        if self.amd.provider.name == Providers.LOCAL.name and self.model is None:
            self.model = gpt4all.GPT4All(self.amd.model)

        if self.amd.provider.name == Providers.LOCAL.name and self.model is not None:
            prompt = ""
            if "chat" in self.amd.model:
                if "llama-2" in self.amd.model:
                    prompt = llama_2_chat_pt(messages=llm_message)
                if "falcon" in self.amd.model:
                    prompt = falcon_chat_pt(messages=llm_message)
                if "mpt" in self.amd.model:
                    prompt = mpt_chat_pt(messages=llm_message)
            elif "wizard" in self.amd.model:
                prompt = wizardcoder_pt(messages=llm_message)
            elif "instruct" in self.amd.model:
                prompt = ollama_pt(messages=llm_message)
                if "falcon" in self.amd.model:
                    prompt = falcon_instruct_pt(messages=llm_message)
            else:
                try:
                    prompt = hf_chat_template(self.amd.model, llm_message)
                except Exception:
                    pass
                if not prompt:
                    prompt = default_pt(messages=llm_message)

            if not prompt:
                prompt = prompt_factory(self.amd.model, llm_message, self.amd.provider.name)

            if not prompt:
                print("No prompt")
                return

            if kwargs.get('mock_response', False):
                return kwargs.get('mock_response')

            stop_callback = None

            if self.amd.stop_sequence:

                self.hits = ""  # TODO : IO string wirte

                def stop_callback_func(token: int, response):
                    self.hits += response
                    if self.hits in self.amd.stop_sequence:
                        return False
                    if response == ' ':
                        self.hits = ""

                    return True

                stop_callback = stop_callback_func

            # Werte, die überprüft werden sollen
            dynamic_values = {

                'streaming': self.stream,
                'temp': self.amd.temperature,
                'top_k': self.amd.top_k,
                'top_p': self.amd.top_p,
                'repeat_penalty': self.amd.repeat_penalty,
                'repeat_last_n': self.amd.repeat_last_n,
                'n_batch': self.amd.n_batch,
                'max_tokens': self.max_tokens,
                'callback': stop_callback
            }

            # Füge Werte zu kwargs hinzu, wenn sie nicht None sind
            kwargs.update(add_to_kwargs_if_not_none(**dynamic_values))

            result = self.model.generate(
                prompt=prompt,
                **kwargs
            )

            return result

        # Werte, die überprüft werden sollen
        dynamic_values = {
            'custom_llm_provider': self.amd.provider.name,
            'temperature': self.amd.temperature,
            'top_p': self.amd.top_p,
            'top_k': self.amd.top_k,
            'stream': self.stream,
            'stop': self.amd.stop_sequence,
            'max_tokens': self.max_tokens,
            'user': self.amd.user_id,
            'api_base': self.amd.api_base,
            'api_version': self.amd.api_version,
            'api_key': self.amd.api_key,
            'logger_fn': self.logger,
            'verbose': self.verbose,
            'fallbacks': self.amd.fallbacks,
            'caching': self.amd.caching,
        }

        # Füge Werte zu kwargs hinzu, wenn sie nicht None sind
        kwargs.update(add_to_kwargs_if_not_none(**dynamic_values))

        result = completion(
            model=self.amd.model,
            messages=llm_message,
            **kwargs
        )

        return result

    def run_model(self, llm_message, **kwargs):

        if not llm_message:
            return None

        result = self.completion(llm_message, **kwargs)

        if self.amd.budget_manager:
            self.amd.budget_manager.update_cost(user=self.amd.user_id, model=self.amd.model, completion_obj=result)

        self.last_result = result
        llm_response = ""
        if not self.stream:
            llm_response = get_str_response(chunk=result)
            self.messages.append(asdict(LLMMessage("assistant", llm_response)))

        if self.stream:

            if self.stream_function is None:
                self.stream_function = stram_print

            for chunk in result:
                message = get_str_response(chunk=chunk)
                llm_response += message
                self.stream_function(message)

            self.messages.append(asdict(LLMMessage("assistant", llm_response)))

        if self.content_memory is not None:
            self.content_memory.text += llm_response

        return llm_response

    def stram_registrator(self, func: Callable[[str], None]):
        self.stream_function = func

    def init_memory(self, isaa):
        self.memory = isaa.get_context_memory()
        self.content_memory = ShortTermMemory(isaa, self.amd.name + "-ShortTermMemory'")


@dataclass
class ModeController:
    mode: LLMMode
    shots = []

    def add_shot(self, user_input, agent_output):
        self.shots.append([user_input, agent_output])

    def run_agent_on_mode(self, task, agent: Agent, persist=False, fetch_memory=False, isaa=None, **kwargs):

        if agent.mode != self.mode:
            agent.mode = self.mode

        message = agent.get_llm_message(
            task,
            persist=persist,
            fetch_memory=fetch_memory,
            isaa=isaa
        )

        result = agent.run_model(message, **kwargs)

        self.add_shot(task, result)

        return result

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

    def generate_multi_shot(self, agent: Optional[Agent], isaa=None):

        if agent is None:
            agent = Agent(
                amd=get_free_agent_data_factory()
            )

        # format prompt
        all_shots = []
        for index, shot in enumerate(self.shots):
            shot_str = f"input : {shot[0]}\noutput : {shot[1]}\n"
            if len(shot) > 2:
                shot_str += f"user-evaluation : {shot[2]}/10 points\n"
            all_shots.append(shot_str)
        all_shots_str = "-------\n".join(all_shots)

        task = (f"Generate an example set based on the following data. for {self.mode.name}"
                f" the description of thes is {self.mode.description}. data : {all_shots_str}")

        llm_message = agent.get_llm_message(task, persist=False, fetch_memory=False, isaa=isaa)

        results = agent.run_model(llm_message)

        print(results)

        self.mode.examples.append(results)
