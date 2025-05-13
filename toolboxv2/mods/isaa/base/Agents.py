import asyncio
import logging
import os
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import partial
from inspect import signature
from typing import Any, Literal

import litellm
from langchain_community.llms import HuggingFaceHub
from litellm import (
    BudgetManager,
    acompletion,
    batch_completion,
    completion,
    token_counter,
)
from litellm.utils import get_max_tokens, trim_messages
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from toolboxv2.mods.isaa.base.AgentUtils import (
    AISemanticMemory,
    ShortTermMemory,
    _extract_from_json,
    _extract_from_string,
    _extract_from_string_de,
    anything_from_str_to_dict,
    get_max_token_fom_model_name,
    get_token_mini,
)
from toolboxv2.mods.isaa.extras.filter import after_format, filter_relevant_texts

try:
    import gpt4all
except Exception:
    def gpt4all():
        return None
    gpt4all.GPT4All = None

import contextlib
import json
from dataclasses import asdict

from fuzzywuzzy import fuzz

from toolboxv2 import Singleton, Spinner, Style, get_logger
from toolboxv2.utils.extras.Style import print_prompt, stram_print


def reduce_system_messages(messages: list[dict[str, str]], similarity_threshold: int = 80) -> list[dict[str, str]]:
    """
    Reduce system messages by removing duplicates and highly similar content using fuzzy string matching.

    :param messages: List of message dictionaries with 'role' and 'content' keys
    :param similarity_threshold: Threshold for considering messages similar (0-100)
    :return: List of reduced system messages
    """
    system_messages = [msg for msg in messages if msg['role'] == 'system']
    reduced_messages = []
    reduced_messages_l = []

    for message in system_messages:
        is_unique = True

        for existing_message in reduced_messages:
            similarity = fuzz.ratio(message['content'], existing_message['content'])
            if similarity >= similarity_threshold:
                is_unique = False
                if len(message['content']) > len(existing_message['content']):
                    reduced_messages[reduced_messages_l.index(len(existing_message['content']))] = message
                    reduced_messages_l[reduced_messages_l.index(len(existing_message['content']))] = len(
                        message['content'])

        if is_unique:
            reduced_messages.append(message)
            reduced_messages_l.append(len(message['content']))

    return reduced_messages


# litellm.cache = Cache()
logger = get_logger()
litellm.set_verbose = False  # logger.level == logging.CRITICAL


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


def add_to_kwargs_if_not_none(**values):
    return {k: v for k, v in values.items() if v}


# @dataclass(frozen=True)
# class Providers(Enum):
#     DEFAULT = None
#     ANTHROPIC = "Anthropic"
#     OPENAI = "OpenAI"
#     REPLICATE = "Replicate"
#     COHERE = "Cohere"
#     HUGGINGFACE = "Huggingface"
#     OPENROUTER = "OpenRouter"
#     AI21 = "AI21"
#     VERTEXAI = "VertexAI"
#     BEDROCK = "Bedrock"
#     OLLAMA = None
#     SAGEMAKER = "Sagemaker"
#     TOGETHERAI = "TogetherAI"
#     ALEPHALPHA = "AlephAlpha"
#     PALM = "Palm"
#     NLP = "NLP"
#     VLLM = "vllm"
#     PETALS = "Petals"
#     LOCAL = "Local"
#     MYAPI = "Myapi"


@dataclass
class Trims(Enum):
    """
    The `Trims` class represents the available text trim options for LLM.
    """
    LITELLM = "Trims"
    ISAA = "IsaaTrim"


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
    parameters: dict[str, str] or list[str] or None
    function: Callable[[str], str] | None

    def __str__(self):
        return f"----\nname -> '{self.name}'\nparameters -> {self.parameters} \ndescription -> '{self.description}'"

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class Capabilities:
    name: str
    description: str
    trait: str
    functions: list[LLMFunction] | None

    # TODO: use a agent to combine capabilities


@dataclass
class LLMMode:
    name: str
    description: str
    system_msg: str
    post_msg: str | None = None
    examples: list[str] | None = None

    def __str__(self):
        return f"LLMMode: {self.name} (description) {self.description}"


@dataclass(frozen=True)
class AgentPromptData:
    initial_prompt_value: str | None
    final_prompt_value: str | None

    system_pre_message: str | None
    system_post_message: str | None

    user_pre_message: str | None
    user_post_message: str | None

    assistant_pre_message: str | None
    assistant_post_message: str | None


@dataclass
class AgentModelData:
    name: str = field(default=None, hash=True)
    model: str = field(default=None)
    model_path: str = field(default=None)
    provider: str | None = field(default=None)
    system_message: str = field(default="")

    temperature: int | None = field(default=None)
    top_k: int | None = field(default=None)
    top_p: int | None = field(default=None)
    repetition_penalty: int | None = field(default=None)

    repeat_penalty: int | None = field(default=None)
    repeat_last_n: float | None = field(default=None)
    n_batch: int | None = field(default=None)

    api_key: str | None = field(default=None)
    api_base: str | None = field(default=None)
    api_version: str | None = field(default=None)
    user_id: str | None = field(default=None)

    fallbacks: (list[dict[str, str]] or list[str]) | None = field(default=None)
    stop_sequence: list[str] | None = field(default=None)
    budget_manager: BudgetManager | None = field(default=None)
    caching: bool | None = field(default=None)


def get_free_agent_data_factory(name="Gpt4All", model="groq/deepseek-r1-distill-qwen-32b") -> AgentModelData:
    return AgentModelData(
        name=name,
        model=model,
        provider=None,
        stop_sequence=["[!X!]"],
    )


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
class LLMFunctionRunner:
    args: list or None = field(default=None)
    kwargs: dict or None = field(default=None)
    llm_function: LLMFunction or None = field(default=None)

    def validate(self):
        if self.llm_function is None:
            return False
        return not (self.llm_function.parameters is not None and self.args is None and self.kwargs is None)

    def __call__(self):
        if not self.validate():
            return "Error Invalid arguments"
        try:
            return self.llm_function(*self.args, **self.kwargs)
        except Exception as e:
            return "Error " + str(e)


# Define Task data class
@dataclass
class Task:
    id: str
    description: str
    priority: int
    estimated_complexity: float  # Range 0.0 to 1.0
    time_sensitivity: float  # Range 0.0 to 1.0
    created_at: datetime


@dataclass
class sTask(BaseModel):
    """.2f"""
    description: str
    priority: int
    estimated_complexity: float  # Range 0.0 to 1.0
    time_sensitivity: float  # Range 0.0 to 1.0


# Define RankingSystem class
class RankingSystem:
    def __init__(self):
        self.ranking_criteria = {
            "complexity": 0.4,
            "priority": 0.3,
            "time_sensitivity": 0.3
        }

    def rank_task(self, task: Task) -> float:
        score = (
            task.estimated_complexity * self.ranking_criteria["complexity"] +
            task.priority * self.ranking_criteria["priority"] +
            task.time_sensitivity * self.ranking_criteria["time_sensitivity"]
        )
        return score


# Define TaskStack class
class TaskStack:
    def __init__(self):
        self.tasks: list[Task] = []
        self.current_task: Task | None = None
        self._lock = threading.Lock()
        self.ranking_system = RankingSystem()

    def add_task(self, task: Task):
        with self._lock:
            self.tasks.append(task)
            self._sort_tasks()

    def _sort_tasks(self):
        self.tasks.sort(key=lambda x: self.ranking_system.rank_task(x), reverse=True)

    def get_next_task(self) -> Task | None:
        with self._lock:
            if self.tasks:
                self.current_task = self.tasks.pop(0)
                return self.current_task
            return None

    def remove_task(self, task_id: str):
        with self._lock:
            self.tasks = [t for t in self.tasks if t.id != task_id]

    def emtpy(self):
        return len(self.tasks) == 0

    def __len__(self):
        return len(self.tasks)


# Define AgentState Enum
class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"


# Define TaskStatus class
@dataclass
class TaskStatus:
    task_id: str
    status: str  # queued, running, completed, error
    progress: float  # Range 0.0 to 1.0
    result: Any | None = None
    error: str | None = None


class StFilter(metaclass=Singleton):
    filter: SentenceTransformer | None = None

@dataclass
class Agent:
    amd: AgentModelData = field(default_factory=AgentModelData)

    main_run_model : str | None = None

    stream: bool = field(default=False)
    messages: list[dict[str, str]] = field(default_factory=list)
    trim: str = field(default="IsaaTrim")
    max_history_length: int = field(default=10)
    similarity_threshold: int = field(default=75)
    verbose: bool = field(default=logger.level == logging.DEBUG)
    batch_completion: bool = field(default=False)
    stream_function: Callable[[str], bool or None] = field(default_factory=print)

    max_tokens: int | None = field(default=None)

    taskstack: TaskStack | None = field(default_factory=TaskStack)
    executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1))

    status_dict: dict[str, TaskStatus] = field(default_factory=dict)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    state: AgentState = AgentState.IDLE
    world_model: dict[str, str] = field(default_factory=dict)

    post_callback: Callable | None = field(default=None)
    progress_callback: Callable | None = field(default=None)

    functions: list[LLMFunction] | None = field(default=None)
    add_function_to_prompt: bool | None = field(default=None)

    config: dict[str, Any] | None = field(default=None)

    batch_completion_messages: list[list[LLMMessage]] | None = field(default=None)

    memory: AISemanticMemory | None = field(default=None)
    content_memory: ShortTermMemory | None = field(default=None)

    capabilities: Capabilities | None = field(default=None)
    mode: (LLMMode or ModeController) | None = field(default=None)
    last_result: dict[str, Any] | None = field(default=None)

    model: (gpt4all.GPT4All or HuggingFaceHub) | None = field(default=None)
    hits: str | None = field(default=None)

    next_fuction: str | None = field(default=None)
    llm_function_runner: LLMFunctionRunner | None = field(default=None)

    rformat: dict | None = field(default=None)

    user_input: str | None = field(default=None)

    vision: bool | None = field(default=None)
    audio: bool | None = field(default=None)

    if_for_fuction_use_overrides: bool = False


    def show_world_model(self):
        if not self.world_model:
            return "balnk"
        return "Key <> Value\n" + "\n".join([f"{k} <> {v}" for k, v in self.world_model.items()])

    async def flow_world_model(self, query):

        prompt = f"Determine if to change the current world model ##{self.show_world_model()}## basd on the new information :" + query

        class WorldModelAdaption(BaseModel):
            """world model adaption action['remove' or 'add' or 'change' or None] ;
            key from the existing word model or new one ; key format xyz.abc like Person.Tom
            information changed or added. informations must be in str relation graph format like 'Person:Name, Works:at, Startup:Complot'"""
            action: str | None = field(default=None)
            key: str | None = field(default=None)
            informations: str | None = field(default=None)

        model_action = await self.a_format_class(WorldModelAdaption, prompt)
        self.print_verbose(str(model_action))
        if model_action.get("action") is None or model_action.get("key") is None:
            return

        if ("remove" in model_action["action"] or "del" in model_action["action"]) and model_action["key"] in self.world_model:
            del self.world_model[model_action["key"]]

        if model_action["informations"] is None:
            return

        self.world_model[model_action["key"]] = model_action["informations"]

    def run_in_background(self):
        """Start a task in background mode"""
        self._stop_event.clear()

        if self.state != AgentState.RUNNING:
            self.state = AgentState.RUNNING
            self.executor.submit(self._background_worker)

        return self.state

    def stop(self):
        self._stop_event.set()

    def _background_worker(self):
        """Background worker that processes queued tasks"""
        while not self._stop_event.is_set():
            try:
                # Get task with timeout to allow checking stop_event
                task = self.taskstack.get_next_task()
                if task is None:
                    self.state = AgentState.IDLE
                    return
                if task.id not in self.status_dict:
                    task_status = TaskStatus(
                        task_id=task.id,
                        status="queued",
                        progress=0.0
                    )
                    self.status_dict[task.id] = task_status

                status = self.status_dict[task.id]
                status.status = "starting"
                self.progress_callback(status)

                try:
                    # Run the main agent logic
                    asyncio.run(self.run(task))

                except Exception as e:
                    status.status = "error"
                    status.error = str(e)

                self.progress_callback(status)

            except queue.Empty:
                continue

        self.state = AgentState.STOPPED

    async def run(self, user_input_or_task: str or Task, with_memory=None, with_functions=None, max_iterations=3, chat_session=None, with_split=False, **kwargs):

        persist = False
        task_from = "user"
        persist_mem = False
        out = None
        # print(user_input_or_task)
        if max_iterations <= 0:
            return "overflow max iterations"
        if isinstance(user_input_or_task, str):
            task = self._to_task(user_input_or_task)
            user_input = user_input_or_task
        elif isinstance(user_input_or_task, Task):
            task = user_input_or_task
            user_input = task.description
        else:
            raise ValueError("Invalid user input or task")

        # Update progress as we go through stages
        stage = [1]

        if self.progress_callback is None:
            self.progress_callback = lambda x: None

        async def update_progress(total_stages: int = 13):
            status = self.status_dict.get(task.id, None)
            if status is None:
                return
            status.progress = float(f"{stage[0] / total_stages:.2f}")
            stage[0] += 1
            status.status = "running" if stage[0] < total_stages else "completed"
            if asyncio.iscoroutinefunction(self.progress_callback):
                await self.progress_callback(status)
            else:
                self.progress_callback(status)

        await update_progress()
        await self.flow_world_model(user_input)
        message = [{"role": "system", "content": f"World Model(read only): {self.show_world_model()}"}]

        if chat_session is not None:
            history = " ==== CHAT HISTORY ===\n"+ "\n".join(f"{x['role'].upper()}: {x['content']}" for x in chat_session.get_past_x(self.max_history_length)) + " === HISTORY END ==="
        else:
            history = " ==== CHAT HISTORY ===\nASSISTANT: "+ str(self.last_result)+ " === HISTORY END ==="

        if len(history) > 75:
            message.append({"role": "system", "content": history})

        if with_split is None:

            class DoSplit(BaseModel):
                """Deside if u need to split ths Task in sub tasks, only ste try on complex tasks"""
                split: bool = field(default=False)

            split_task = self.format_class(DoSplit, user_input)["split"]

        else:
            split_task = with_split

        self.print_verbose(f'Split {split_task}')
        await update_progress()
        if split_task:

            class TaskList(BaseModel):
                """sTask Breakdown Format:
1. Each subtask should be represented as a structured object with:
   - description: Clear, actionable description of the subtask
   - priority: Integer ranking (1 being highest priority)
   - estimated_complexity: Float value between 0.0 (simplest) and 1.0 (most complex)
   - time_sensitivity: Float value indicating urgency (0.0 for least urgent, 1.0 for most urgent)

2. Requirements for breakdown:
   - Break the main task into logical, self-contained subtasks
   - Order subtasks by dependencies and priority
   - Ensure each description is specific and measurable
   - Consider interdependencies when assigning priority
   - Factor in both technical complexity and business impact when estimating complexity
   - Account for deadlines and dependencies in time sensitivity ratings

3. include all Informations in the description !
"""
                sub_tasks: list[sTask]

            sub_tasks = self.format_class(TaskList, user_input)["sub_tasks"]

            self.print_verbose(f"Subtasks {len(sub_tasks)}")
            st_d = '\n'.join([f'{i}) '+(s["description"] if isinstance(s, dict) else s)+ '\n' for i, s in enumerate(sub_tasks)])
            self.print_verbose(f"Subtasks\n {st_d}")

            if len(sub_tasks) > 1:
                for subt in sub_tasks[1:][::-1]:
                    subt_ = Task(
                        id=str(uuid.uuid4())[:16],
                        description=subt["description"] if isinstance(subt, dict) else subt,
                        priority=subt["priority"] if isinstance(subt, dict) else 1,
                        estimated_complexity=subt["estimated_complexity"] if isinstance(subt, dict) else 1,
                        time_sensitivity=subt["time_sensitivity"] if isinstance(subt, dict) else 1,
                        created_at=datetime.now()
                    )
                    self.taskstack.add_task(subt_)
            st = (sub_tasks[0]["description"] if isinstance(sub_tasks[0], dict) else sub_tasks[0])
            if len(sub_tasks) > 0 and st != "" and st != "default description":
                _sub_tasks = sub_tasks[0]
                task = Task(
                    id=task.id,
                    description=st+ task.description if len(sub_tasks) == 1 else '',
                    priority=_sub_tasks["priority"] if isinstance(sub_tasks[0], dict) else 1,
                    estimated_complexity=_sub_tasks["estimated_complexity"] if isinstance(sub_tasks[0], dict) else 1,
                    time_sensitivity=_sub_tasks["time_sensitivity"] if isinstance(sub_tasks[0], dict) else 1,
                    created_at=datetime.now()
                )

        # Stage 2: Memory handling
        # Stage 1: Initialize and validate inputs
        await update_progress()
        if with_functions is None:
            class WithFunctions(BaseModel):
                f"""Deside if u need to call one function to perform this task {[f.name for f in self.functions] if self.functions is not None else ''}"""
                use_function: bool = field(default=False)

            with_functions = self.format_class(WithFunctions, user_input).get("use_function", True)
            self.print_verbose(f'Auto {with_functions=}')

        await update_progress()
        if with_memory is None:
            class WithMemory(BaseModel):
                """Deside if u need to get memory data to perform this task"""
                use_memory: bool = field(default=False)

            with_memory = self.format_class(WithMemory, user_input).get("use_memory", True)
            self.print_verbose(f'Auto {with_memory=}')
        # Stage 4: Function execution
        await update_progress()
        if with_memory:
            persist_mem = True
            message = await self.get_memory(user_input, message, callback=update_progress)

        # Stage 3: Function execution
        await update_progress()

        sto_model = None
        if self.main_run_model is not None:
            sto_model = self.amd.model
            self.amd.model = self.main_run_model

        if with_functions is False:
            self.add_function_to_prompt = False

            with Spinner(message="Fetching llm_message...", symbols='+'):
                llm_message = self.get_llm_message(user_input, persist=persist,
                                                   task_from=task_from, message=message)
            await update_progress()

            iterations = max_iterations
            while out is None and iterations > 0:
                iterations -= 1
                out = await self.a_run_model(llm_message=llm_message, persist_local=persist, persist_mem=persist_mem, **kwargs)
            await update_progress()

            await update_progress()

        elif with_functions is True:
            self.add_function_to_prompt = True

            with Spinner(message="Fetching llm_message...", symbols='+'):
                llm_message = self.get_llm_message(user_input, persist=persist,
                                                   task_from=task_from, message=message)
            await update_progress()

            iterations = max_iterations
            save_stream = self.stream
            self.stream = False
            while out is None and iterations > 0:
                iterations -= 1
                out = await self.a_run_model(llm_message=llm_message, persist_local=persist, persist_mem=persist_mem, **kwargs)
            await update_progress()
            self.stream = save_stream

            if self.if_for_fuction_use(out):
                res = await self.execute_fuction(persist=persist, persist_mem=persist_mem)
                out += f"(system tool '{self.llm_function_runner.llm_function.name if self.llm_function_runner is not None else '-#-'}' inputs : {self.llm_function_runner.args if self.llm_function_runner is not None else '-#-'} output): {res}"
                #out += await self.a_mini_task(out_, "system", """Return a Markdown-formatted output that includes both the function call and its context. Evaluate the result:
                #If an error or failure occurs, simply indicate that an error occurred without attempting to fix it.
                #Otherwise, format the function output neatly so it can be used directly.""", message=[x for x in llm_message if x.get('role') != "system"])
            await update_progress()

        else:
            if sto_model is not None:
                self.amd.model = sto_model
            raise ValueError(f"Could not {with_functions=} must be true or false")

        if sto_model is not None:
            self.amd.model = sto_model
        stage[0] = 1
        await update_progress(1)

        if task.id not in self.status_dict:
            task_status = TaskStatus(
                task_id=task.id,
                status="queued",
                progress=0.0
            )
            self.status_dict[task.id] = task_status

        self.status_dict[task.id].status = "completed"
        self.status_dict[task.id].result = out
        self.status_dict[task.id].progress = 1.0

        return out

    def _to_task(self, query):
        class TaskStats(BaseModel):
            """Esitmate priority HIGH = 3,MEDIUM = 2,LOW = 1, complexity between 0 and 1"""
            priority: int
            complexity: float

        ts = self.format_class(TaskStats, query)
        task = Task(
            id=str(uuid.uuid4())[:16],
            description=query,
            priority=ts["priority"],
            estimated_complexity=ts["complexity"],
            time_sensitivity=1 - ts["complexity"],
            created_at=datetime.now()
        )
        self.status_dict[task.id] = TaskStatus(
            task_id=task.id,
            status="queued",
            progress=0.0
        )
        return task

    async def get_memory(self, ref, massages=None, callback=None, memory_names=None, **kwargs):

        if massages is None:
            massages = []

        if callback is None:
            async def callback(*a,**k):
                pass
        update_progress = callback

        def add_unique_message(role: str, content: str):
            new_message = asdict(LLMMessage(role, content.strip()))
            if new_message not in massages:
                massages.append(new_message)

        class MemoryQuery(BaseModel):
            """ref question or  context for this task"""
            query: str

        class RefMemory(BaseModel):
            queries: list[MemoryQuery]

        queries = self.format_class(RefMemory, ref)

        queries = queries["queries"][:3]
        await update_progress()
        if self.memory:
            for query in queries:
                query = query["query"]
                memory_task = await self.memory.query(query, memory_names=[self.amd.name] if memory_names is None else memory_names, to_str=True, **kwargs)
                if memory_task:
                    mem_data = json.dumps(memory_task, indent=2)
                    add_unique_message("system", "(system General-context) :" + mem_data)
        await update_progress()
        if self.content_memory and len(self.content_memory.text) > 260:
            class Context(BaseModel):
                """Persise and relevant context only for {ref[:5000]}"""
                context: str

            context = self.format_class(Context, self.content_memory.text + f"Persise and relevant context only for {ref[:5000]}")["context"]
            if context:
                add_unique_message("system", "(system memory-context) :" + context)
            await update_progress()
        self.print_verbose(f"Addet {len(massages)} entry(s)")
        return massages

    def mini_task(self, user_task, task_from="user", mini_task=None, message=None, persist=False):
        if message is None:
            message = []
        if mini_task is not None:
            message.append({'role': 'system', 'content': mini_task})
        self.add_function_to_prompt = False
        if isinstance(user_task, str):
            llm_message = self.get_llm_message(user_task, persist=persist, task_from=task_from, message=message)
        elif isinstance(user_task, list):
            llm_message = self.get_batch_llm_messages(user_task, task_from=task_from, message=message.copy())
        else:
            raise ValueError(f"Invalid mini_task type valid ar str or List[str] is {type(mini_task)} {mini_task}")
        return self.run_model(llm_message=llm_message, persist_local=persist, batch=isinstance(mini_task, list))

    async def a_mini_task(self, user_task, task_from="user", mini_task=None, message=None, persist=False):
        if message is None:
            message = []
        if mini_task is not None:
            message.append({'role': 'system', 'content': mini_task})
        self.add_function_to_prompt = False
        if isinstance(user_task, str):
            llm_message = self.get_llm_message(user_task, persist=persist, task_from=task_from, message=message)
        elif isinstance(user_task, list):
            llm_message = await self.get_batch_llm_messages(user_task, task_from=task_from, message=message.copy())
        else:
            raise ValueError(f"Invalid mini_task type valid ar str or List[str] is {type(mini_task)}")
        return await self.a_run_model(llm_message=llm_message, persist_local=persist, batch=isinstance(mini_task, list))

    def format_class(self, format_class, task, **kwargs):
        tstrem = self.stream
        self.stream = False
        llm_message = self.get_llm_message(task, persist=False, **kwargs)
        if 'claude' in self.amd.model and llm_message[0]['role'] != 'user':
            llm_message = [{'role':'user','content':'start :)'}] +llm_message

        try:
            resp = self.completion(
                llm_message=llm_message,
                response_format=format_class,
            )

            c = self.format_helper(resp)
        except litellm.exceptions.BadRequestError as e:
            if 'failed_generation' not in str(e):
                raise e
            c = str(e).split('"failed_generation":')[-1][:-3]
        res = after_format(c)
        self.stream = tstrem
        print(res)
        return res

    async def a_format_class(self, format_class, task, **kwargs):
        tstrem = self.stream
        self.stream = False
        llm_message = self.get_llm_message(task, persist=False, **kwargs)
        if 'claude' in self.amd.model and llm_message[0]['role'] != 'user':
            llm_message = [{'role':'user','content':'start :)'}] +llm_message
        # print_prompt(llm_message)
        try:
            resp = await self.acompletion(
                llm_message=llm_message,
                response_format=format_class,
            )

            c = self.format_helper(resp)
        except litellm.exceptions.BadRequestError as e:
            if 'failed_generation' not in str(e):
                raise e
            c = str(e).split('"failed_generation":')[-1][:-3]
        # print(resp)
        self.last_result = c

        try:
            res = after_format(c)
            self.stream = tstrem
            return res
        except Exception as e:
            self.print_verbose(f"Error formatting, Retrying... {e}")
            llm_message = [{'role': 'system', 'content': f'retry error : {e}'}] + llm_message
            resp = await self.acompletion(
                llm_message=llm_message,
                response_format=format_class,
            )
            self.stream = tstrem
            # print(resp)
            c = self.format_helper(resp)
            res = after_format(c)
            self.stream = tstrem
            return res

    def format_helper(self, resp):
        c = None
        if not self.stream:
            with contextlib.suppress(ValueError):
                c = resp.choices[0].message.tool_calls[0].function.arguments
        if c is None:
            c = self.parse_completion(resp)
        return c

    def function_invoke(self, name, **kwargs):
        if self.functions is None:
            return "no functions"
        fuction_list = [f.function for f in self.functions if f.name == name]
        if len(fuction_list):
            try:
                return fuction_list[0](**kwargs)
            except Exception as e:
                return f"Error in fuction {name} :" + str(e)
        return f"function {name} not found"

    def reset_context(self):
        self.messages = []
        self.world_model = {}
        self.content_memory.text = ""

    def check_valid(self):

        if self.amd.name is None:
            print(self.amd)
            return False

        if self.amd.provider is not None and self.amd.provider.upper() in ["LOCAL"]:
            return True

        response = True  # check_valid_key(model=self.amd.model, api_key=self.amd.api_key)

        if not response:
            self.print_verbose(f"Agent Failed {self.amd.name} {self.amd.model}")

        self.print_verbose(f"Agent Parsed {self.amd.name} {self.amd.model}")
        return response

    def construct_first_msg(self) -> list[dict[str, str]]:
        llm_prompt = self.amd.system_message
        self.print_verbose("construct first msg")
        cfunctions_infos = []
        message = []
        if self.capabilities:
            llm_prompt += '\n' + self.capabilities.trait

            if self.capabilities.functions:
                cfunctions_infos = [functions for functions in self.capabilities.functions if
                                    functions not in (self.functions if self.functions else [])]

        if self.mode:
            llm_prompt += '\n' + self.mode.system_msg

            if self.mode.examples:
                llm_prompt += "\nExamples: \n" + '-----\n' + "\n---\n".join(
                    self.mode.examples) + '\n END of Examples!\n'

        if self.add_function_to_prompt and (self.functions or len(cfunctions_infos)):
            functions_infos = "\n".join(
                [str(functions) for functions in (self.functions if self.functions else []) + cfunctions_infos])
            functions_infos = functions_infos.replace("_empty", 'str')
            message.append({'role': 'system', 'content': "calling a function by using this exact syntax (json) : {"
                                                         "'Action':str, 'Inputs':str or dict}\nWhere Action is equal to "
                                                         "the function name and Inputs to the function args. use str for "
                                                         "single input function and a kwarg dict for multiple inputs!! (in one line do not use line brakes or special enclosing!)"
                                                         "USE THIS FORMAT\n" + f"Callable functions:\n{functions_infos}\n--+--\nTemplate Call {{'Action':str, 'Inputs': {{function inputs args or kwargs}}}} all function calls must include 'Action' AND Inputs' as key!\nAfter Calling a function type 3 '.' and new line ...\n"})

        if llm_prompt:
            message.append({'role': 'system', 'content': llm_prompt})
        return message

    async def get_batch_llm_messages(self, user_input: list[str], fetch_memory: bool | None = None,
                               message=None, task_from: str = 'user'):
        llm_messages = []
        for task in user_input:
            msg = self.get_llm_message(user_input=task, persist=False, message=message, task_from=task_from)
            if fetch_memory:
                msg = await self.get_memory(task, msg)
            llm_messages.append(msg)
        return llm_messages

    def get_llm_message(self, user_input: str, persist: bool | None = None,
                        message=None, task_from: str = 'user'):
        llm_message = message
        if llm_message is None:
            llm_message = []

        self.user_input = user_input

        # Helper function to add a message to llm_message without duplicates
        def add_unique_message(role: str, content: str):
            new_message = asdict(LLMMessage(role, content.strip()))
            if new_message not in llm_message:
                llm_message.append(new_message)

        # Add initial system message if it's the first call
        if not persist or len(self.messages) == 0:
            llm_message.extend(self.construct_first_msg())

        if persist and len(self.messages) > 1 and "system" not in [x.get("role") for x in self.messages]:
            [add_unique_message(m['role'], m['content']) for m in self.construct_first_msg() if
             'content' in m and 'role' in m]

        # Add the current task
        llm_message.append(asdict(LLMMessage(task_from, user_input.strip())))
        # Add mode-specific message if applicable
        if self.mode and self.mode.post_msg:
            add_unique_message("system", self.mode.post_msg)

        # Trim the message history
        llm_message = self.trim_msg(llm_message)

        # Handle persistence and update content memory
        if persist:
            self.messages.extend(llm_message)
            if self.content_memory:
                self.content_memory.text += f"\nUSER:{user_input}\nRESPONSE:"
            llm_message = self.messages

        # Organize messages: system messages first, then chat history
        system_messages = [msg for msg in llm_message if msg['role'] == 'system']
        chat_history = [msg for msg in llm_message if msg['role'] != 'system']

        # Limit chat history length and offload excess
        max_history_length = self.max_history_length  # Adjust as needed
        if len(chat_history) > max_history_length:
            self.offloaded_history = chat_history[:-max_history_length]
            chat_history = chat_history[-max_history_length:]

        # Reduce system messages
        reduced_system_messages = reduce_system_messages(system_messages,
                                                         similarity_threshold=self.similarity_threshold)

        # Combine organized messages
        llm_message = reduced_system_messages + chat_history

        self.print_verbose(f"Returning llm message {len(llm_message)}")
        return llm_message

    def trim_msg(self, llm_message=None, isaa=None):

        if self.trim == 'IsaaTrim' and isaa:


            # print("================================\n", self.prompt_str(llm_message),
            # "\n================================\n")
            def get_tokens_estimation(text, only_len=True):
                if isinstance(text, list):
                    text = '\n'.join(msg['content'] for msg in text if isinstance(msg['content'], str))

                tokens = get_token_mini(text, self.amd.model, isaa, only_len)
                if only_len and tokens == 0:
                    tokens = int(len(text) * (3 / 4))
                return tokens

            new_msg = isaa.short_prompt_messages(llm_message.copy(), get_tokens_estimation,
                                                 get_max_token_fom_model_name(self.amd.model))
            om = ''.join([c['content'] for c in llm_message])
            nm = ''.join([c['content'] for c in new_msg])
            nt = get_tokens_estimation(nm, True)
            self.print_verbose(f"Timing with IsaaTrim from {len(om)} to {len(nm)}")
            self.print_verbose(f"Timing with IsaaTrim place {get_max_token_fom_model_name(self.amd.model)-nt}")
            self.print_verbose(f" tokens {get_tokens_estimation(om, True)} to {nt} max {get_max_token_fom_model_name(self.amd.model)} ")
            if new_msg:
                llm_message = new_msg

        else:  #         if self.trim == 'Trims':
            self.print_verbose("Timing with Trims")
            with Spinner(message="Sorten prompt lit...", symbols='d'):
                new_msg = trim_messages(llm_message, self.amd.model)
                if new_msg:
                    llm_message = new_msg
        return llm_message

    def set_rformat(self, specification: dict):
        if isinstance(specification, dict):
            self.rformat = {"type": "json_object", "json_schema": specification, "strict": True}
        elif isinstance(specification, str):
            self.rformat = {"type": "text", "schema": specification, "strict": True}

    def reset_rformat(self):
        self.rformat = None

    def prompt_str(self, llm_message):
        llm_message = self.trim_msg(llm_message)
        prompt = "\n".join(f"{d.get('role')}:{d.get('content')}" for d in llm_message)

        return prompt

    def completion(self, llm_message, batch=False, **kwargs):
        self.print_verbose("Starting completion")

        if self.vision:
            llm_message = llm_message.copy()
            for msg in llm_message:
                if msg.get('role') != 'assistant':
                    msg['content'] = self.content_add_immage(msg['content'])

        if self.amd.provider is not None and self.amd.provider.upper() == "GPT4All" and self.model is None:
            self.model = gpt4all.GPT4All(self.amd.model)

        if self.amd.provider is not None and self.amd.provider.upper() == "GPT4All" and self.model is not None:
            prompt = self.prompt_str(llm_message)

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
            self.print_verbose("Local Completion don")
            return result

        # Werte, die überprüft werden sollen
        dynamic_values = {
            'response_format': self.rformat,
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
            'verbose': self.verbose,
            # 'fallbacks': self.amd.fallbacks,
            'caching': self.amd.caching,
            'functions': [{"name": f.name, "description": f.description, "parameters": f.parameters} for f in
                                   self.functions] if self.add_function_to_prompt else None,
            'custom_llm_provider': self.amd.provider if self.amd.provider is not None and self.amd.provider.upper() != "DEFAULT" else None
        }

        if 'claude' in self.amd.model:
            dynamic_values['drop_params'] = True

        if self.add_function_to_prompt:
            litellm.add_function_to_prompt = True

        # Füge Werte zu kwargs hinzu, wenn sie nicht None sind
        kwargs.update(add_to_kwargs_if_not_none(**dynamic_values))

        if batch:
            result = batch_completion(
                model=self.amd.model,
                messages=llm_message,
                # fallbacks=os.getenv("FALLBACKS_MODELS").split(','),
                **kwargs
            )
        else:
            # print("Model completion", self.amd.model, llm_message, kwargs)
            result = completion(
                model=self.amd.model,
                messages=llm_message,
                # fallbacks=os.getenv("FALLBACKS_MODELS").split(','),
                **kwargs
            )

        litellm.add_function_to_prompt = False
        self.print_verbose("Completion", "Done" if not self.stream else "in progress..")
        return result

    async def acompletion(self, llm_message, batch=False, **kwargs):
        self.print_verbose("Starting acompletion")

        if self.vision:
            llm_message = llm_message.copy()
            for msg in llm_message:
                if msg.get('role') != 'assistant':
                    msg['content'] = self.content_add_immage(msg['content'])

        if self.amd.provider is not None and self.amd.provider.upper() == "GPT4All" and self.model is None:
            self.model = gpt4all.GPT4All(self.amd.model)

        if self.amd.provider is not None and self.amd.provider.upper() == "GPT4All" and self.model is not None:
            prompt = self.prompt_str(llm_message)

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
            self.print_verbose("Local Completion don")
            return result

        # Werte, die überprüft werden sollen
        dynamic_values = {
            'response_format': self.rformat,
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
            'verbose': self.verbose,
            # 'fallbacks': self.amd.fallbacks,
            'caching': self.amd.caching,
            'functions': [{"name": f.name, "description": f.description, "parameters": f.parameters} for f in
                                   self.functions] if self.add_function_to_prompt else None,
            'custom_llm_provider': self.amd.provider if self.amd.provider is not None and self.amd.provider.upper() != "DEFAULT" else None
        }

        if 'claude' in self.amd.model:
            dynamic_values['drop_params'] = True

        if self.add_function_to_prompt:
            litellm.add_function_to_prompt = True

        # Füge Werte zu kwargs hinzu, wenn sie nicht None sind
        kwargs.update(add_to_kwargs_if_not_none(**dynamic_values))

        if batch:
            result = batch_completion(
                model=self.amd.model,
                messages=llm_message,
                # fallbacks=os.getenv("FALLBACKS_MODELS").split(','),
                **kwargs
            )
        else:
            # print("Model completion", self.amd.model, llm_message, kwargs)
            result = await acompletion(
                model=self.amd.model,
                messages=llm_message,
                # fallbacks=os.getenv("FALLBACKS_MODELS").split(','),
                **kwargs
            )

        litellm.add_function_to_prompt = False
        self.print_verbose("Completion", "Done" if not self.stream else "in progress..")
        return result

    def transcription(self, file_bytes, prompt="", temperature=0,
                      response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] | None = None
                      ):
        return litellm.transcription(
            model=self.amd.model,
            file=file_bytes,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
        )

    def model_function_result_passer(self, result, default=None):
        if default is None:
            default = ""
        else:
            default += "\n"

        # Check if we are in streaming mode using the self.stram flag
        if self.stream:
            # Iterate over each streaming choice chunk
            for choice in result.choices:
                delta = getattr(choice, "delta", None)
                if delta and getattr(delta, "content", None):
                        default = delta.content
        else:
            # Non-streaming mode handling
            print(result.choices[0])
            if hasattr(result.choices[0].message, "tool_calls") and result.choices[
                0].message.tool_calls and self.functions is not None:
                if len(result.choices[0].message.tool_calls) != 1:
                    default += f"taskstack added {len(result.choices[0].message.tool_calls)}"
                    for fuc_call in result.choices[0].message.tool_calls:
                        self.taskstack.add_task(
                            self._to_task(
                                f"Call this function '{fuc_call.function.name}' with these arguments: {fuc_call.function.arguments}"
                            )
                        )
                else:
                    callable_functions = [func.name.lower() for func in
                                          self.functions] if self.functions is not None else []
                    function_name = result.choices[0].message.tool_calls[0].function.name.lower()
                    if function_name in callable_functions:
                        llm_function = self.functions[callable_functions.index(function_name)]
                        self.if_for_fuction_use_overrides = True
                        d = json.loads(result.choices[0].message.tool_calls[0].function.arguments)
                        if 'properties' in d and isinstance(d['properties'], dict):
                            d = d['properties']
                        self.llm_function_runner = LLMFunctionRunner(
                            llm_function=llm_function,
                            args=(),
                            kwargs=d,
                        )
                        default += f"Calling {result.choices[0].message.tool_calls[0].function.name} with arguments {result.choices[0].message.tool_calls[0].function.arguments}"
        return default

    def parse_completion(self,result):
        llm_response = ""
        if not self.stream:
            return get_str_response(chunk=result)

        if self.stream:
            self.print_verbose("Start streaming")

            if self.stream_function is None:
                self.stream_function = stram_print

            chunks = []
            for chunk in result:
                chunks.append(chunk)
                message = get_str_response(chunk=chunk)
                message = self.model_function_result_passer(chunk, message)
                llm_response += message
                if self.stream_function(message):
                    break
            self.print_verbose("Done streaming")
            result = litellm.stream_chunk_builder(chunks)
        return self.model_function_result_passer(result, llm_response)

    def run_model(self, llm_message, persist_local=True, persist_mem=True, batch=False, **kwargs):

        if not llm_message:
            return None

        self.print_verbose("Running llm model")

        self.next_fuction = None

        if len(llm_message) > 2:
            llm_message = [{'role': 'assistant',
                            'content': f'Hello, I am an intelligent agent created to assist you. To provide the best possible response, I will first gather information about you and any relevant context. I will then analyze the requirements for a unified agent response and develop a multi-step reasoning process to address your needs. This process will involve distinct streams of thought and personality, culminating in a final, cohesive action. Please provide any additional details or instructions you may have, and I will do my best to deliver a helpful and personalized solution. To anabel a sees of time i must allways remember the [system time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]'}] + llm_message

        if len(llm_message) == 1 and llm_message[0]['role'] != 'user':
            llm_message = [{'role': 'user', 'content': 'Performe the system task!'}] + llm_message
        if llm_message[0]['role'] != 'user':
            llm_message = [{'role': 'user', 'content': '.'}] + llm_message
        # print_prompt(llm_message)

        result = None
        max_r = 2
        r_try = 0
        last_error_ = None
        llm_response = ""
        tok_input = get_token_mini(llm_message, self.amd.model)
        try:
            tok_max = get_max_tokens(self.amd.model)
        except Exception:
            tok_max = 199000
        print(f"AGENT {self.amd.name} TOKENS {tok_input} {tok_max}")
        if tok_input > tok_max:
            llm_message = self.trim_msg(llm_message)
        while result is None and r_try < max_r:
            try:
                result = self.completion(llm_message=llm_message, batch=batch, **kwargs)
                r_try = 9999
                break
            except litellm.RateLimitError as e:
                print(f"RateLimitError {e}")
                last_error_ = e
                if '413' in str(e) and 'reduce' not in str(e):
                    with Spinner("Reitlimit Waiting 1 minute"):
                        time.sleep(30)
                r_try += 1
                llm_message = self.trim_msg(llm_message)
            except litellm.InternalServerError as e:
                print(f"InternalServerError {e}")
                last_error_ = e
                r_try += 1
                # print_prompt(llm_message)
                lm = len(llm_message)
                llm_message = self.trim_msg(llm_message)
                print(f"AFTER TRIM {lm}/{len(llm_message)}")
                # print_prompt(llm_message)
                with Spinner("Waring... for api", count_down=True, time_in_s=r_try * 10):
                    time.sleep(r_try * 10)
                continue

        if result is None and last_error_ is not None:
            raise last_error_

        llm_response = self.parse_completion(result)

        if not batch:
            return self.compute_result(result, llm_message, llm_response, persist_local, persist_mem)
        return [self.compute_result(_result, llm_message, llm_response, persist_local, persist_mem) for _result in
                result]

    async def a_run_model(self, llm_message, persist_local=True, persist_mem=True, batch=False, **kwargs):

        if not llm_message:
            return None

        self.print_verbose("Running llm model")

        self.next_fuction = None

        if len(llm_message) > 2:
            llm_message = [{'role': 'assistant',
                            'content': f'Hello, I am an intelligent agent created to assist you. To provide the best possible response, I will first gather information about you and any relevant context. I will then analyze the requirements for a unified agent response and develop a multi-step reasoning process to address your needs. This process will involve distinct streams of thought and personality, culminating in a final, cohesive action. Please provide any additional details or instructions you may have, and I will do my best to deliver a helpful and personalized solution. [system time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]'}] + llm_message

        if len(llm_message) == 1 and llm_message[0]['role'] != 'user':
            llm_message = [{'role': 'user', 'content': 'Performe the system task!'}] + llm_message
        if llm_message[0]['role'] != 'user':
            llm_message = [{'role': 'user', 'content': '.'}] + llm_message
        # print_prompt(llm_message)

        result = None
        max_r = 2
        r_try = 0
        last_error_ = None
        llm_response = ""
        tok_input = get_token_mini(llm_message, self.amd.model)
        try:
            tok_max = get_max_tokens(self.amd.model)
        except Exception:
            tok_max = 199000
        print(f"AGENT {self.amd.name} TOKENS {tok_input} {tok_max}")
        if tok_input > tok_max:
            llm_message = self.trim_msg(llm_message)
        while result is None and r_try < max_r:
            try:
                result = await self.acompletion(llm_message=llm_message, batch=batch, **kwargs)
                r_try = 9999
                break
            except litellm.RateLimitError as e:
                print(f"RateLimitError {e}")
                last_error_ = e
                if '413' in str(e) and 'reduce' not in str(e):
                    with Spinner("Reitlimit Waiting 1 minute"):
                        time.sleep(30)
                r_try += 1
                llm_message = self.trim_msg(llm_message)
            except litellm.InternalServerError as e:
                print(f"InternalServerError {e}")
                last_error_ = e
                r_try += 1
                # print_prompt(llm_message)
                lm = len(llm_message)
                llm_message = self.trim_msg(llm_message)
                print(f"AFTER TRIM {lm}/{len(llm_message)}")
                # print_prompt(llm_message)
                with Spinner("Waring... for api", count_down=True, time_in_s=r_try * 10):
                    time.sleep(r_try * 10)
                continue

        if result is None and last_error_ is not None:
            raise last_error_

        llm_response = self.parse_completion(result)

        if not batch:
            return await self.acompute_result(result, llm_message, llm_response, persist_local, persist_mem)
        return [await self.acompute_result(_result, llm_message, llm_response, persist_local, persist_mem) for _result in
                result]

    async def acompute_result(self, result, llm_message, llm_response, persist_local=False, persist_mem=False) -> str:
        print_prompt(llm_message + [{'content': llm_response, 'role': 'assistant'}])
        self.last_result = llm_response
        if self.amd.budget_manager:
            self.amd.budget_manager.update_cost(user=self.amd.user_id, model=self.amd.model, completion_obj=result)

        await self.save_to_memory(llm_response, persist_local, persist_mem)

        if self.mode is not None:
            if isinstance(llm_message[-1], dict):
                _llm_message = [llm_message]
            else:
                _llm_message = llm_message
            for llm_message in _llm_message:
                # print(f"{isinstance(self.mode, ModeController)=} and {hasattr(self.mode, 'add_shot')=} and {llm_message[-1].get('content', False)=}")
                if isinstance(self.mode, ModeController) and hasattr(self.mode, 'add_shot') and llm_message[-1].get(
                    'content',
                    False):
                    self.mode.add_shot(llm_message[-1].get('content'), llm_response)

        if self.post_callback:
            await self.post_callback(llm_response)
        return llm_response

    def compute_result(self, result, llm_message, llm_response, persist_local=False, persist_mem=False) -> str:
        print_prompt(llm_message + [{'content': llm_response, 'role': 'assistant'}])
        self.last_result = llm_response
        if self.amd.budget_manager:
            self.amd.budget_manager.update_cost(user=self.amd.user_id, model=self.amd.model, completion_obj=result)

        # get_app().run_a_from_sync(self.save_to_memory,llm_response, persist_local, persist_mem)

        if self.mode is not None:
            if isinstance(llm_message[-1], dict):
                _llm_message = [llm_message]
            else:
                _llm_message = llm_message
            for llm_message in _llm_message:
                # print(f"{isinstance(self.mode, ModeController)=} and {hasattr(self.mode, 'add_shot')=} and {llm_message[-1].get('content', False)=}")
                if isinstance(self.mode, ModeController) and hasattr(self.mode, 'add_shot') and llm_message[-1].get(
                    'content',
                    False):
                    self.mode.add_shot(llm_message[-1].get('content'), llm_response)

        if self.post_callback:
            self.post_callback(llm_response)
        return llm_response

    async def save_to_memory(self, llm_response: str, persist_local=False, persist_mem=False):

        if isinstance(llm_response, list) and len(llm_response) > 0 and isinstance(llm_response[0], str):
            llm_response = '\n'.join(llm_response)
        elif isinstance(llm_response, list) and len(llm_response) == 0:
            return

        async def helper():
            if persist_local:
                self.messages.append({'content': llm_response, 'role': 'assistant'})

            if self.amd.name.startswith("TaskC"):
                return
            if persist_mem and self.memory is not None:
                self.print_verbose("persist response to persistent_memory")
                await self.memory.add_data(self.amd.name,  'CHAT - HISTORY\n'+self.user_input + '\n:\n' + str(llm_response))

            if persist_mem and self.content_memory is not None:
                self.print_verbose("persist response to content_memory")
                self.content_memory.text += llm_response

        await helper()
        # threading.Thread(target=helper, daemon=True).start()

    def if_for_fuction_use(self, llm_response):
        if self.if_for_fuction_use_overrides:
            self.if_for_fuction_use_overrides = False
            self.print_verbose("Function runner initialized")
            return True
        llm_fuction = None
        fuction_inputs = None
        self.next_fuction = None
        if self.capabilities is not None and self.capabilities.functions is not None and len(
            self.capabilities.functions) > 0:
            callable_functions = [fuction_name.name.lower() for fuction_name in self.capabilities.functions]

            self.next_fuction, fuction_inputs = self.test_use_function(llm_response, callable_functions)
            if self.next_fuction is not None:
                llm_fuction = self.capabilities.functions[callable_functions.index(self.next_fuction.lower())]

        if self.functions is not None and len(self.functions) > 0 and self.next_fuction is None:
            callable_functions = [fuction_name.name.lower() for fuction_name in self.functions]
            self.next_fuction, fuction_inputs = self.test_use_function(llm_response, callable_functions)
            if self.next_fuction is not None:
                llm_fuction = self.functions[callable_functions.index(self.next_fuction.lower())]

        if self.next_fuction is None and llm_fuction is None:
            self.llm_function_runner = LLMFunctionRunner(
                llm_function=None,
                args=None,
                kwargs=None,
            )
            self.print_verbose("No fuction called")

            return False

        args = []
        kwargs = {}

        if fuction_inputs is not None:
            args, kwargs = self.parse_arguments(fuction_inputs, llm_fuction.parameters)

        self.llm_function_runner = LLMFunctionRunner(
            llm_function=llm_fuction,
            args=args,
            kwargs=kwargs,
        )
        self.print_verbose("Function runner initialized")
        return True

    def print_verbose(self, *args, **kwargs):
        if self.verbose and self.amd.name is not None:
            print(Style.BLUE(f"AGENT:{self.amd.name}: "), end='')
            print(' '.join(args)[:250], **kwargs)

    async def execute_fuction(self, persist=True, persist_mem=True):
        if self.next_fuction is None:
            if self.verbose:
                print("No fuction to execute")
            return "No fuction to execute"

        if self.llm_function_runner is None:
            if self.verbose:
                print("No llm function runner to execute")
            return "No llm function runner to execute"

        if not self.llm_function_runner.validate():
            if self.verbose:
                print("Invalid llm function runner")
            return "Invalid llm function runner"

        result = self.llm_function_runner()
        if asyncio.iscoroutine(result):
            result = await result

        self.print_verbose(f"Fuction {self.llm_function_runner.llm_function.name} Result : {result}")

        if persist:
            self.messages.append({'content': f"(system tool {self.next_fuction}) result:{result}", 'role': "system"})

        if persist_mem and self.content_memory is not None:
            self.content_memory.text += f"F:{result}"
            self.print_verbose("Persist to content Memory")

        if persist_mem and self.memory is not None:
            await self.memory.add_data(self.amd.name, f"FUNKTION Result:{result}")
            self.print_verbose(f"Persist to Memory sapce {self.amd.name}")
        if not isinstance(result, str):
            result = str(result)
        return result

    def stram_registrator(self, func: Callable[[str], bool]):
        self.print_verbose("StramRegistrator")
        self.stream_function = func

    def init_memory(self, isaa, name: str = None):
        if name is None or name == 'None':
            name = self.amd.name
        if name is None:
            raise ValueError("Invalid Agent")
        if name == 'None':
            return
        self.print_verbose("Initializing Memory")
        self.memory = isaa.get_memory()
        self.content_memory = ShortTermMemory(isaa, name + "-ShortTermMemory")

    def save_memory(self):
        self.print_verbose("Saving memory")
        if self.content_memory is not None:
            self.print_verbose("Saved memory to collective")
            self.content_memory.clear_to_collective()

    def token_counter(self, messages: list):
        return token_counter(model=self.amd.model, messages=messages)

    @staticmethod
    def fuzzy_string_match(input_string: str, match_list: list, return_info=False):
        input_string = input_string.lower()
        matches = []
        for i in list(range(len(input_string)))[::-1]:
            for match in match_list:
                if match.startswith(input_string[:i]):
                    matches.append(match)
        for i in list(range(len(input_string))):
            for match in match_list:
                if match.endswith(input_string[i:]):
                    matches.append(match)
        v_match = []
        for match in match_list:
            v_match.append(matches.count(match))
        # print(v_match)
        # print(match_list)
        if return_info:
            return match_list, v_match
        return match_list[v_match.index(max(v_match))]

    def test_use_function(self, agent_text: str, all_actions: list[str], language='en') -> tuple[
        str or None, str or None]:
        if not agent_text:
            return None, None
        agent_text = agent_text.replace('`', '').replace('\n', '')
        # all_actions = [action_.lower() for action_ in all_actions]
        self.print_verbose("Starting tests... tools...")

        action, inputs = _extract_from_json(agent_text.replace("'", '"'), all_actions)
        # print(f"1 {action=}| {inputs=} {agent_text}")
        if action is not None:
            return action.lower(), inputs

        if language == 'de':

            # print("_extract_from_string")
            action, inputs = _extract_from_string_de(agent_text.replace("'", '"'), all_actions)
            # print(f"2 {action=}| {inputs=} {agent_text}")
            if action is not None:
                return action.lower(), inputs

        action, inputs = _extract_from_string(agent_text.replace("'", '"'), all_actions)
        # print(f"3 {action=}| {inputs=} {agent_text}")
        if action is not None:
            return action.lower(), inputs

        try:
            agent_dict = anything_from_str_to_dict(agent_text)
            # print(f"4 {agent_dict=}| {agent_text}")
            if len(agent_dict) > 0:
                action = agent_dict.get("Action", '')
                inputs = agent_dict.get("Inputs", {})
            if action is not None:
                return action, inputs
        except ValueError:
            pass

        return None, None

    @staticmethod
    def parse_arguments(command: str, parameters: list or dict) -> (list, dict):
        # Initialisierung der Ausgabeliste und des Wörterbuchs
        out_list = []
        out_dict = {}
        args = []
        param_keys = parameters if isinstance(parameters, list) else (
            list(parameters.keys()) if hasattr(parameters, 'keys') else list(parameters))

        # Überprüfung, ob der Befehl ein Wörterbuch enthält
        if isinstance(command, dict):
            command = json.dumps(command)
        if isinstance(command, list):
            args = command
        if not isinstance(command, str):
            command = str(command)

        if "{" in command and "}" in command:
            s = {}
            for x in param_keys:
                s[x] = None
            arg_dict = anything_from_str_to_dict(command, expected_keys=s)

            if isinstance(arg_dict, list) and len(arg_dict) >= 1:
                arg_dict = arg_dict[0]

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
                if comma_cont == len(param_keys) - 1:
                    split_key = ","
                elif newline_cont == len(param_keys) - 1:
                    split_key = "\n"
                elif saces_cont == len(param_keys) - 1:
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

        for i, arg in enumerate(args):
            if i < len(param_keys) and i != "callbacks":
                out_dict[param_keys[i]] = arg
            else:
                out_list.append(arg)

        return out_list, out_dict

    @staticmethod
    def content_add_immage(content):
        import base64
        from urllib.parse import urlparse

        import requests

        def parse_image_references(text):
            """
            Find image references in the format 'Image[path/url]' and analyze each match.

            Args:
                text (str): Text to search for image references

            Returns:
                list[tuple]: List of tuples (image_path_url, is_path, is_url, image_type)
            """
            from urllib.parse import urlparse

            # Pattern to match Image[...] format
            pattern = r'Image\[(.*?)\]'

            # Common image extensions
            image_extensions = {
                'jpg': 'JPEG',
                'jpeg': 'JPEG',
                'png': 'PNG',
                'gif': 'GIF',
                'pdf': 'PDF',
                'bmp': 'BMP',
                'webp': 'WEBP',
                'svg': 'SVG',
                'tiff': 'TIFF'
            }

            def analyze_match(match):
                path_or_url = match.strip()

                # Check if it's a URL
                try:
                    parsed = urlparse(path_or_url)
                    is_url = bool(parsed.scheme and parsed.netloc)
                except:
                    is_url = False

                # Check if it's a local path
                is_path = not is_url and ('/' in path_or_url or '\\' in path_or_url)

                # Get file extension and type
                extension = path_or_url.split('.')[-1].lower().split('?')[0]
                image_type = image_extensions.get(extension, 'Unknown')

                return (path_or_url, is_path, is_url, image_type)

            # Find all matches and analyze them
            matches = re.finditer(pattern, text)
            results = [analyze_match(m.group(1)) for m in matches]

            return results

        image_urls = parse_image_references(content)

        if len(image_urls) == 0:
            return content

        def is_valid_url(url):
            try:
                result = urlparse(url)
                return all([result.scheme, result.netloc])
            except ValueError:
                return False

        def encode_image_local(image_url, img_type):
            with open(image_url, "rb") as image_file:
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img_type};base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
                    },
                }

        def encode_image_url(image_url, img_type):
            response = requests.get(image_url)
            file_data = response.content
            return {
                "type": "image_url",
                "image_url": f"data:image/{img_type};base64,{base64.b64encode(file_data).decode('utf-8')}",
            }

        new_content = [{"type": "text", "text": content}]

        for image_path_url, is_path, is_url, image_type in image_urls:

            if is_url:
                is_valid = is_valid_url(image_path_url)
                new_content.append(encode_image_url(image_path_url, image_type)
                                   if is_valid else
                                   encode_image_local(image_path_url, image_type))
                continue

            if is_path:
                new_content.append(encode_image_local(image_path_url, image_type))
                continue

        return new_content

class AgentBuilder:
    isaa_reference = None

    def __init__(self, agent_class: type[Agent]):
        self.agent = agent_class()
        self.amd_attributes = {}
        self.missing_amd_fields = ["name", "model"]
        self.is_build = False

    def init_agent_memory(self, name):
        if self.isaa_reference is None:
            raise ValueError("isaa_reference is required")
        self.agent.init_memory(self.isaa_reference, name)
        return self

    def set_isaa_reference(self, isaa):
        self.isaa_reference = isaa
        return self

    def set_amd_name(self, name: str):
        self.amd_attributes['name'] = name
        if "name" not in self.missing_amd_fields:
            self.missing_amd_fields += ["name"]
        self.missing_amd_fields.remove('name')
        return self

    def set_amd_model(self, model: str):
        self.amd_attributes['model'] = model
        if "model" not in self.missing_amd_fields:
            self.missing_amd_fields += ["model"]
        self.missing_amd_fields.remove('model')
        if model.startswith('ollama'):
            return self.set_amd_api_base("http://localhost:11434")
        return self

    def set_amd_provider(self, provider: str):
        self.amd_attributes['provider'] = provider
        return self

    def set_amd_temperature(self, temperature: int):
        self.amd_attributes['temperature'] = temperature
        return self

    def set_amd_top_k(self, top_k: int):
        self.amd_attributes['top_k'] = top_k
        return self

    def set_amd_top_p(self, top_p: int):
        self.amd_attributes['top_p'] = top_p
        return self

    def set_amd_repetition_penalty(self, repetition_penalty: int):
        self.amd_attributes['repetition_penalty'] = repetition_penalty
        return self

    def set_amd_repeat_penalty(self, repeat_penalty: int):
        self.amd_attributes['repeat_penalty'] = repeat_penalty
        return self

    def set_amd_repeat_last_n(self, repeat_last_n: float):
        self.amd_attributes['repeat_last_n'] = repeat_last_n
        return self

    def set_amd_n_batch(self, n_batch: int):
        self.amd_attributes['n_batch'] = n_batch
        return self

    def set_amd_api_key(self, api_key: str):
        self.amd_attributes['api_key'] = api_key
        return self

    def set_amd_api_base(self, api_base: str):
        self.amd_attributes['api_base'] = api_base
        return self

    def set_amd_api_version(self, api_version: str):
        self.amd_attributes['api_version'] = api_version
        return self

    def set_amd_user_id(self, user_id: str):
        self.amd_attributes['user_id'] = user_id
        return self

    def set_amd_fallbacks(self, fallbacks: list[dict[str, str]] or list[str]):
        self.amd_attributes['fallbacks'] = fallbacks
        return self

    def set_amd_stop_sequence(self, stop_sequence: list[str]):
        self.amd_attributes['stop_sequence'] = stop_sequence
        return self

    def set_amd_budget_manager(self, budget_manager: BudgetManager):
        self.amd_attributes['budget_manager'] = budget_manager
        return self

    def set_amd_caching(self, caching: bool):
        self.amd_attributes['caching'] = caching
        return self

    def set_amd_system_message(self, system_message: str):
        self.amd_attributes['system_message'] = system_message
        return self

        # Fügen Sie weitere Methoden für alle Eigenschaften von AgentModelData hinzu

    def build_amd(self):
        if len(self.missing_amd_fields) != 0:
            raise ValueError(
                f"Invalid AMD configuration missing : {self.missing_amd_fields} set ar \n{self.amd_attributes}")
        # Erstellt ein AgentModelData-Objekt mit den gesetzten Attributen
        self.agent.amd = AgentModelData(**self.amd_attributes)
        if self.agent.amd.name is None:
            raise ValueError(
                f"Invalid AMD configuration missing : Data. set ar \n{self.amd_attributes}")

        return self

    # ==========================================================

    def set_post_callback(self, post_callback: Callable):
        self.agent.post_callback = post_callback
        return self

    def set_progress_callback(self, progress_callback: Callable):
        self.agent.progress_callback = progress_callback
        return self

    def set_print_verbose(self, print_verbose: Callable):
        self.agent.print_verbose = print_verbose
        return self

    def set_stream(self, stream: bool):
        self.agent.stream = stream
        return self

    def set_messages(self, messages: list[dict[str, str]]):
        self.agent.messages = messages
        return self

    def set_trim(self, trim: Trims):
        self.agent.trim = trim.name
        return self

    def set_verbose(self, verbose: bool):
        self.agent.verbose = verbose
        return self

    def set_batch_completion(self, batch_completion: bool):
        self.agent.batch_completion = batch_completion
        return self

    def set_stream_function(self, stream_function: Callable[[str], None]):
        self.agent.stream_function = stream_function
        return self

    def set_max_tokens(self, max_tokens: int | None):
        self.agent.max_tokens = max_tokens
        return self

    def set_taskstack(self, taskstack: TaskStack | None):
        self.agent.taskstack = taskstack
        return self

    def set_functions(self, functions: list[LLMFunction] | None):
        if self.agent.functions is None:
            self.agent.functions = []
        self.agent.functions += functions
        return self

    def set_config(self, config: dict[str, Any] | None):
        self.agent.config = config
        return self

    def set_batch_completion_messages(self, batch_completion_messages: list[list[LLMMessage]] | None):
        self.agent.batch_completion_messages = batch_completion_messages
        return self

    def set_memory(self, memory: AISemanticMemory | None):
        self.agent.memory = memory
        return self

    def set_content_memory(self, content_memory: ShortTermMemory | None):
        self.agent.content_memory = content_memory
        return self

    def set_content_memory_max_length(self, content_memory_max_length: int):
        if self.agent.content_memory is None:
            raise ValueError("content_memory is not set")
        self.agent.content_memory.max_length = content_memory_max_length

    def set_content_memory_isaa_instance(self, isaa):
        if self.agent.content_memory is None:
            raise ValueError("content_memory is not set")
        self.agent.content_memory.isaa = isaa

    def set_capabilities(self, capabilities: Capabilities | None):
        self.agent.capabilities = capabilities
        return self

    def set_mode(self, mode: (LLMMode or ModeController) | None):
        self.agent.mode = mode
        return self

    def set_last_result(self, last_result: dict[str, Any] | None):
        self.agent.last_result = last_result
        return self

    def set_model(self, model: (gpt4all.GPT4All or HuggingFaceHub) | None):
        self.agent.model = model
        return self

    def set_hits(self, hits: str | None):
        self.agent.hits = hits
        return self

    def set_vision(self, vision: bool | None):
        self.agent.vision = vision
        return self

    def set_add_function_to_prompt(self, add_function_to_prompt: bool | None):
        self.agent.add_function_to_prompt = add_function_to_prompt
        return self

    def build(self):
        if self.isaa_reference is None:
            raise ValueError("no isaa_reference set")
        if self.is_build:
            print("Agent was constructed! pleas delay builder instance")
            return self.agent
        self.build_amd()
        if not self.agent.check_valid():
            raise ValueError(
                f"Invalid Agent:{self.agent.amd.name} Configuration\n{self.amd_attributes}\n{self.agent.amd}")
        print(f"Agent '{self.agent.amd.name}' Agent-Model-Data build successfully")
        self.is_build = True
        self.agent.trim_msg = partial(self.agent.trim_msg, isaa=self.isaa_reference)

        if self.agent.progress_callback is None:
            self.agent.progress_callback = lambda x: print(x)
        if StFilter.filter is None:
            StFilter.filter = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.agent.filter = partial(filter_relevant_texts, model=StFilter.filter)
        return self.agent

    def save_to_json(self, file_path: str):
        clear_data = self.get_dict()
        with open(file_path, 'w') as f:
            json.dump(clear_data, f, ensure_ascii=False, indent=4)
        return clear_data

    def get_dict(self):
        clear_data = {'amd': self.amd_attributes}
        if self.amd_attributes.get('provider') is not None:
            if isinstance(self.amd_attributes['provider'], Enum):
                print(f"{self.amd_attributes['provider'].name=}")
                clear_data['amd']['provider'] = self.amd_attributes['provider'].name
            else:
                clear_data['amd']['provider'] = self.amd_attributes['provider']
        clear_data['stream'] = self.agent.stream  # : bool = field(default=False)
        clear_data['messages'] = self.agent.messages  # : List[Dict[str, str]] = field(default_factory=list)
        clear_data['trim'] = self.agent.trim  # : Trims = field(default=Trims.litellm)
        clear_data['verbose'] = self.agent.verbose  # : bool = field(default=False)

        clear_data['batch_completion'] = self.agent.batch_completion  # : bool = field(default=False)
        # clear_data['stream_function'] = self.agent.stream_function  # : Callable[[str], bool or None] = field(default_factory=print)
        clear_data['max_tokens'] = self.agent.max_tokens  # : Optional[float] = field(default=None)
        # clear_data['taskstack.tasks'] = self.agent.taskstack.tasks  # : Optional[List[str]] = field(default=None)
        # clear_data['functions'] = self.agent.functions  # : Optional[List[LLMFunction]] = field(default=None)
        clear_data['config'] = self.agent.config  # : Optional[Dict[str, Any]] = field(default=None)
        clear_data['last_result'] = self.agent.last_result  # : Optional[Dict[str, Any]] = field(default=None)
        # clear_data['model'] = self.agent.model  # : Optional[gpt4all.GPT4All or HuggingFaceHub] = field(default=None)
        clear_data['hits'] = self.agent.hits  # : Optional[str] = field(default=None)
        clear_data['next_fuction'] = self.agent.next_fuction  # : Optional[str] = field(default=None)
        # clear_data['llm_function_runner'] = self.agent.llm_function_runner  # : Optional[LLMFunctionRunner] = field(default=None
        return clear_data

    @classmethod
    def load_from_json_file(cls, file_path: str, agent_class: type[Agent]):
        try:
            with open(file_path) as f:
                data = json.load(f)
            print(f"Loaded Agent from file {data}")
        except:
            print(f"Could not read from file: {file_path}")
            return cls(agent_class)
        builder = cls(agent_class)
        return cls.load_from_json_file_dict_data(data, builder)

    @staticmethod
    def load_from_json_file_dict_data(data, builder):
        for key, value in data.items():
            if hasattr(builder.agent, key):
                if key == "amd":
                    builder.amd_attributes = value
                    if 'name' in value:
                        builder.set_amd_name(value['name'])
                    if 'model' in value:
                        builder.set_amd_name(value['model'])
                    continue
                setattr(builder.agent, key, value)
        return builder

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


def test_test_use_function():
    s = Agent.test_use_function(
        """SPESPEAK: Hello! I'd be happy to help you find the current weather in Berlin.PLAN: I will first search for up-to-date weather data using the 'searchWeb' function.FUCTION: {'Action': 'searchWeb', 'Inputs': {'x': 'current weather Berlin'}}""",
        ["searchWeb"])
    print(s)


class AgentVirtualEnv:
    def __init__(self):
        self.default = "FUNCTION"
        self.prefixes: dict[str, LLMFunction] = {}
        self.buffer: dict[str, list[str]] = {}
        self._results = ""
        self._brake_test = None
        self.brake = None

    def reset(self):
        self.brake = None
        self._results = ""
        return self

    def set_brake_test(self, l: Callable[[str], bool]):
        self._brake_test = l

    def break_test(self, text):
        if self.brake is not None:
            return self.brake
        if self._brake_test is None:
            self._brake_test = lambda x: "RESPONSE" in x
        return self._brake_test(text)

    def results(self):
        return self._results

    def results_set(self, value):
        self._results += value

    def results_reset(self):
        self._results = ""

    def register_prefix(self, name: str, description: str, parameters: dict[str, str] | list[str] | None = None):
        def decorator(func):
            if parameters is None:
                self.prefixes[name] = LLMFunction(name, description, signature(func).parameters.items(), func)
            else:
                self.prefixes[name] = LLMFunction(name, description, parameters, func)
            return func

        return decorator

    def get_llm_mode(self) -> Capabilities:
        name = "ASAPT-CustomModel"
        description = "Use a reactive framework to solve problems based on custom prefixes"

        trait = """You are an intelligent system that operates based on a strict prefix structure. Each prefix defines the nature of the task to be performed. Below are the instructions for handling various prefixes:
""" + "\n".join([f"{prefix}: {func.description}" for prefix, func in self.prefixes.items() if prefix != self.default]) + """
Example Input Stream:

THINK: I need to greet the user and perform a calculation.
THINK: lets use the add fuction
{"Action": "add", "Inputs": [5, 3]}
RESPONSE: The result of 5 + 3 is 8.
By following these prefix-based instructions, you ensure that each task is handled in a structured and effective manner, with clear separation of thought, communication, and function execution. Note: The content under the USER prefix is for internal use only and should never be directly output to the user. use actions in json format!"""

        return Capabilities(name, description, trait, list(self.prefixes.values()))
