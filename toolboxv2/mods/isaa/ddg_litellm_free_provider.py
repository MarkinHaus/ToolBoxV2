import httpx
from typing import Optional, Union, Callable, Iterator
from litellm.utils import ModelResponse
from litellm.utils import HTTPHandler
import litellm
import json

class CustomLLMError(Exception):  # use this for all your exceptions
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs



class DDGProver:
    def __init__(self) -> None:
        self._chat_vqd = ""
        self._chat_messages = []
        self._chat_tokens_count = 0
        self.client = httpx.Client()

    def completion(
        self,
        model: str,
        messages: list,
        # api_base: str,
        # custom_prompt_dict: dict,
        # print_verbose: Callable,
        # encoding,
        # api_key,
        # logging_obj,
        # optional_params: dict,
        # acompletion=None,
        # litellm_params=None,
        # logger_fn=None,
        headers=None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        # client: Optional[HTTPHandler] = None,
    ) -> str:
        # Implement the chat function logic here
        keywords = messages# [-1]['content']  # Assume the last message is the user's input
        model = model or "gpt-4o-mini"
        timeout = timeout or 30
        if headers is None:
            headers = {}
        models = {
            "claude-3-haiku": "claude-3-haiku-20240307",
            "gpt-4o-mini": "gpt-4o-mini",
            "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }

        # Get or update vqd
        if not self._chat_vqd:
            resp = self.client.get("https://duckduckgo.com/duckchat/v1/status", headers={"x-vqd-accept": "1"})
            self._chat_vqd = resp.headers.get("x-vqd-4", "")

        self._chat_messages.append({"role": "user", "content": keywords})
        self._chat_tokens_count += len(keywords) // 4 if len(keywords) >= 4 else 1

        json_data = {
            "model": models[model],
            "messages": self._chat_messages,
        }
        resp = self.client.post(
            "https://duckduckgo.com/duckchat/v1/chat",
            headers={"x-vqd-4": self._chat_vqd},
            json=json_data,
            timeout=timeout,
        )
        self._chat_vqd = resp.headers.get("x-vqd-4", "")

        data = ",".join(x for line in resp.text.rstrip("[DONE]LIMT_CVRSA\n").split("data:") if (x := line.strip()))
        data = json.loads("[" + data + "]")

        results = []
        for x in data:
            if x.get("action") == "error":
                err_message = x.get("type", "")
                if x.get("status") == 429:
                    raise CustomLLMError(status_code=429, message=err_message)
                raise CustomLLMError(status_code=500, message=err_message)
            elif message := x.get("message"):
                results.append(message)
        result = "".join(results)

        self._chat_messages.append({"role": "assistant", "content": result})
        self._chat_tokens_count += len(results)

        # Populate ModelResponse
        return result

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> Iterator[ModelResponse]:
        # Implement streaming version of the chat function
        keywords = messages[-1]['content']
        model = model or "gpt-4o-mini"
        timeout = timeout or 30

        models = {
            "claude-3-haiku": "claude-3-haiku-20240307",
            "gpt-4o-mini": "gpt-4o-mini",
            "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }

        # Get or update vqd
        if not self._chat_vqd:
            resp = self.client.get("https://duckduckgo.com/duckchat/v1/status", headers={"x-vqd-accept": "1"})
            self._chat_vqd = resp.headers.get("x-vqd-4", "")

        self._chat_messages.append({"role": "user", "content": keywords})
        self._chat_tokens_count += len(keywords) // 4 if len(keywords) >= 4 else 1

        json_data = {
            "model": models[model],
            "messages": self._chat_messages,
        }
        with self.client.stream(
            "POST",
            "https://duckduckgo.com/duckchat/v1/chat",
            headers={"x-vqd-4": self._chat_vqd},
            json=json_data,
            timeout=timeout,
        ) as resp:
            self._chat_vqd = resp.headers.get("x-vqd-4", "")
            full_response = ""
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    data = json.loads(line[5:])
                    if data.get("action") == "error":
                        err_message = data.get("type", "")
                        if data.get("status") == 429:
                            raise CustomLLMError(status_code=429, message=err_message)
                        raise CustomLLMError(status_code=500, message=err_message)
                    elif message := data.get("message"):
                        full_response += message
                        model_response.choices[0].message.content = message
                        yield model_response

        self._chat_messages.append({"role": "assistant", "content": full_response})
        self._chat_tokens_count += len(full_response)


litellm.custom_provider_map = [# 👈 KEY STEP - REGISTER HANDLER
        {"provider": "ddg", "custom_handler": DDGProver()}
    ]


from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
import httpx
import json

client: httpx.Client = httpx.Client()

class DDGChat(BaseChatModel):
    """DuckDuckGo Chat model."""

    chat_vqd: str = ""
    chat_messages: List[dict] = []
    chat_tokens_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "ddg_chat"

    def _get_chat_vqd(self) -> None:
        """Get or update the chat VQD."""
        resp = client.get("https://duckduckgo.com/duckchat/v1/status", headers={"x-vqd-accept": "1"})
        self.chat_vqd = resp.headers.get("x-vqd-4", "")

    def _convert_messages_to_ddg_format(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain message format to DDG format."""
        ddg_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                ddg_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                ddg_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                ddg_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, ChatMessage):
                ddg_messages.append({"role": message.role, "content": message.content})
        return ddg_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.chat_vqd:
            self._get_chat_vqd()

        ddg_messages = self._convert_messages_to_ddg_format(messages)
        self.chat_messages.extend(ddg_messages)

        model = kwargs.get("model", "gpt-4o-mini")
        models = {
            "claude-3-haiku": "claude-3-haiku-20240307",
            "gpt-4o-mini": "gpt-4o-mini",
            "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }

        json_data = {
            "model": models[model],
            "messages": self.chat_messages,
        }

        resp = client.post(
            "https://duckduckgo.com/duckchat/v1/chat",
            headers={"x-vqd-4": self.chat_vqd},
            json=json_data,
            timeout=kwargs.get("timeout", 30),
        )
        self.chat_vqd = resp.headers.get("x-vqd-4", "")

        data = ",".join(x for line in resp.text.rstrip("[DONE]LIMT_CVRSA\n").split("data:") if (x := line.strip()))
        data = json.loads("[" + data + "]")

        results = []
        for x in data:
            if x.get("action") == "error":
                err_message = x.get("type", "")
                if x.get("status") == 429:
                    raise ValueError(f"Rate limit exceeded: {err_message}")
                raise ValueError(f"Error: {err_message}")
            elif message := x.get("message"):
                results.append(message)

        result = "".join(results)
        self.chat_messages.append({"role": "assistant", "content": result})
        self.chat_tokens_count += len(result)

        message = AIMessage(content=result)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.chat_vqd:
            self._get_chat_vqd()

        ddg_messages = self._convert_messages_to_ddg_format(messages)
        self.chat_messages.extend(ddg_messages)

        model = kwargs.get("model", "gpt-4o-mini")
        models = {
            "claude-3-haiku": "claude-3-haiku-20240307",
            "gpt-4o-mini": "gpt-4o-mini",
            "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }

        json_data = {
            "model": models[model],
            "messages": self.chat_messages,
        }

        with client.stream(
            "POST",
            "https://duckduckgo.com/duckchat/v1/chat",
            headers={"x-vqd-4": self.chat_vqd},
            json=json_data,
            timeout=kwargs.get("timeout", 30),
        ) as resp:
            self.chat_vqd = resp.headers.get("x-vqd-4", "")
            full_response = ""
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    data = json.loads(line[5:])
                    if data.get("action") == "error":
                        err_message = data.get("type", "")
                        if data.get("status") == 429:
                            raise ValueError(f"Rate limit exceeded: {err_message}")
                        raise ValueError(f"Error: {err_message}")
                    elif message := data.get("message"):
                        full_response += message
                        yield ChatGeneration(message=AIMessage(content=message))

        self.chat_messages.append({"role": "assistant", "content": full_response})
        self.chat_tokens_count += len(full_response)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": "ddg_chat"}

    def bind_tools(self, *args, **kwargs):
        return self


if __name__ == "__main__":
    ddg = DDGProver()
    ret = ddg.completion("gpt-4o-mini", "hi")
    print(ret)
