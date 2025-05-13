import asyncio
import cProfile
import io
import logging
import multiprocessing as mp
import os
import pstats
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from inspect import signature
from types import ModuleType
from typing import Any

from pydantic import BaseModel

from ..extras import generate_test_cases
from ..extras.Style import Spinner
from .all_functions_enums import *
from .file_handler import FileHandler


@dataclass
class Headers:
    """Class representing HTTP headers with strongly typed common fields."""
    # General Headers
    accept: None | str= None
    accept_charset: None | str= None
    accept_encoding: None | str= None
    accept_language: None | str= None
    accept_ranges: None | str= None
    access_control_allow_credentials: None | str= None
    access_control_allow_headers: None | str= None
    access_control_allow_methods: None | str= None
    access_control_allow_origin: None | str= None
    access_control_expose_headers: None | str= None
    access_control_max_age: None | str= None
    access_control_request_headers: None | str= None
    access_control_request_method: None | str= None
    age: None | str= None
    allow: None | str= None
    alt_svc: None | str= None
    authorization: None | str= None
    cache_control: None | str= None
    clear_site_data: None | str= None
    connection: None | str= None
    content_disposition: None | str= None
    content_encoding: None | str= None
    content_language: None | str= None
    content_length: None | str= None
    content_location: None | str= None
    content_range: None | str= None
    content_security_policy: None | str= None
    content_security_policy_report_only: None | str= None
    content_type: None | str= None
    cookie: None | str= None
    cross_origin_embedder_policy: None | str= None
    cross_origin_opener_policy: None | str= None
    cross_origin_resource_policy: None | str= None
    date: None | str= None
    device_memory: None | str= None
    digest: None | str= None
    dnt: None | str= None
    dpr: None | str= None
    etag: None | str= None
    expect: None | str= None
    expires: None | str= None
    feature_policy: None | str= None
    forwarded: None | str= None
    from_header: None | str= None  # 'from' is a Python keyword
    host: None | str= None
    if_match: None | str= None
    if_modified_since: None | str= None
    if_none_match: None | str= None
    if_range: None | str= None
    if_unmodified_since: None | str= None
    keep_alive: None | str= None
    large_allocation: None | str= None
    last_modified: None | str= None
    link: None | str= None
    location: None | str= None
    max_forwards: None | str= None
    origin: None | str= None
    pragma: None | str= None
    proxy_authenticate: None | str= None
    proxy_authorization: None | str= None
    public_key_pins: None | str= None
    public_key_pins_report_only: None | str= None
    range: None | str= None
    referer: None | str= None
    referrer_policy: None | str= None
    retry_after: None | str= None
    save_data: None | str= None
    sec_fetch_dest: None | str= None
    sec_fetch_mode: None | str= None
    sec_fetch_site: None | str= None
    sec_fetch_user: None | str= None
    sec_websocket_accept: None | str= None
    sec_websocket_extensions: None | str= None
    sec_websocket_key: None | str= None
    sec_websocket_protocol: None | str= None
    sec_websocket_version: None | str= None
    server: None | str= None
    server_timing: None | str= None
    service_worker_allowed: None | str= None
    set_cookie: None | str= None
    sourcemap: None | str= None
    strict_transport_security: None | str= None
    te: None | str= None
    timing_allow_origin: None | str= None
    tk: None | str= None
    trailer: None | str= None
    transfer_encoding: None | str= None
    upgrade: None | str= None
    upgrade_insecure_requests: None | str= None
    user_agent: None | str= None
    vary: None | str= None
    via: None | str= None
    warning: None | str= None
    www_authenticate: None | str= None
    x_content_type_options: None | str= None
    x_dns_prefetch_control: None | str= None
    x_forwarded_for: None | str= None
    x_forwarded_host: None | str= None
    x_forwarded_proto: None | str= None
    x_frame_options: None | str= None
    x_xss_protection: None | str= None

    # Browser-specific and custom headers
    sec_ch_ua: None | str= None
    sec_ch_ua_mobile: None | str= None
    sec_ch_ua_platform: None | str= None
    sec_ch_ua_arch: None | str= None
    sec_ch_ua_bitness: None | str= None
    sec_ch_ua_full_version: None | str= None
    sec_ch_ua_full_version_list: None | str= None
    sec_ch_ua_platform_version: None | str= None

    # HTMX specific headers
    hx_boosted: None | str= None
    hx_current_url: None | str= None
    hx_history_restore_request: None | str= None
    hx_prompt: None | str= None
    hx_request: None | str= None
    hx_target: None | str= None
    hx_trigger: None | str= None
    hx_trigger_name: None | str= None

    # Additional fields can be stored in extra_headers
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Convert header keys with hyphens to underscores for attribute access."""
        # Handle the 'from' header specifically since it's a Python keyword
        if 'from' in self.__dict__:
            self.from_header = self.__dict__.pop('from')

        # Store any attributes that weren't explicitly defined in extra_headers
        all_attrs = self.__annotations__.keys()
        for key in list(self.__dict__.keys()):
            if key not in all_attrs and key != "extra_headers":
                self.extra_headers[key.replace("_", "-")] = getattr(self, key)
                delattr(self, key)

    @classmethod
    def from_dict(cls, headers_dict: dict[str, str]) -> 'Headers':
        """Create a Headers instance from a dictionary."""
        # Convert header keys from hyphenated to underscore format for Python attributes
        processed_headers = {}
        extra_headers = {}

        for key, value in headers_dict.items():
            # Handle 'from' header specifically
            if key.lower() == 'from':
                processed_headers['from_header'] = value
                continue

            python_key = key.replace("-", "_").lower()
            if python_key in cls.__annotations__ and python_key != "extra_headers":
                processed_headers[python_key] = value
            else:
                extra_headers[key] = value

        return cls(**processed_headers, extra_headers=extra_headers)

    def to_dict(self) -> dict[str, str]:
        """Convert the Headers object back to a dictionary."""
        result = {}

        # Add regular attributes
        for key, value in self.__dict__.items():
            if key != "extra_headers" and value is not None:
                # Handle from_header specially
                if key == "from_header":
                    result["from"] = value
                else:
                    result[key.replace("_", "-")] = value

        # Add extra headers
        result.update(self.extra_headers)

        return result


@dataclass
class Request:
    """Class representing an HTTP request."""
    content_type: str
    headers: Headers
    method: str
    path: str
    query_params: dict[str, Any] = field(default_factory=dict)
    form_data: dict[str, Any] | None = None
    body: Any | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Request':
        """Create a Request instance from a dictionary."""
        headers = Headers.from_dict(data.get('headers', {}))

        # Extract other fields
        return cls(
            content_type=data.get('content_type', ''),
            headers=headers,
            method=data.get('method', ''),
            path=data.get('path', ''),
            query_params=data.get('query_params', {}),
            form_data=data.get('form_data'),
            body=data.get('body')
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the Request object back to a dictionary."""
        result = {
            'content_type': self.content_type,
            'headers': self.headers.to_dict(),
            'method': self.method,
            'path': self.path,
            'query_params': self.query_params,
        }

        if self.form_data is not None:
            result['form_data'] = self.form_data

        if self.body is not None:
            result['body'] = self.body

        return result


@dataclass
class Session:
    """Class representing a session."""
    SiID: str
    level: str
    spec: str
    user_name: str
    # Allow for additional fields
    extra_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Session':
        """Create a Session instance from a dictionary."""
        # Extract known fields
        known_fields = {k: data.get(k) for k in ['SiID', 'level', 'spec', 'user_name'] if k in data}

        # Extract extra fields
        extra_data = {k: v for k, v in data.items() if k not in known_fields}

        return cls(**known_fields, extra_data=extra_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert the Session object back to a dictionary."""
        result = {
            'SiID': self.SiID,
            'level': self.level,
            'spec': self.spec,
            'user_name': self.user_name,
        }

        # Add extra data
        result.update(self.extra_data)

        return result

    @property
    def valid(self):
        return int(self.level) > 0


@dataclass
class RequestData:
    """Main class representing the complete request data structure."""
    request: Request
    session: Session
    session_id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'RequestData':
        """Create a RequestData instance from a dictionary."""
        return cls(
            request=Request.from_dict(data.get('request', {})),
            session=Session.from_dict(data.get('session', {})),
            session_id=data.get('session_id', '')
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the RequestData object back to a dictionary."""
        return {
            'request': self.request.to_dict(),
            'session': self.session.to_dict(),
            'session_id': self.session_id
        }


# Example usage:
def parse_request_data(data: dict[str, Any]) -> RequestData:
    """Parse the incoming request data into a strongly typed structure."""
    return RequestData.from_dict(data)


# Example data parsing
if __name__ == "__main__":
    example_data = {
        'request': {
            'content_type': 'application/x-www-form-urlencoded',
            'headers': {
                'accept': '*/*',
                'accept-encoding': 'gzip, deflate, br, zstd',
                'accept-language': 'de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7',
                'connection': 'keep-alive',
                'content-length': '107',
                'content-type': 'application/x-www-form-urlencoded',
                'cookie': 'session=abc123',
                'host': 'localhost:8080',
                'hx-current-url': 'http://localhost:8080/api/TruthSeeker/get_main_ui',
                'hx-request': 'true',
                'hx-target': 'estimates-guest_1fc2c9',
                'hx-trigger': 'config-form-guest_1fc2c9',
                'origin': 'http://localhost:8080',
                'referer': 'http://localhost:8080/api/TruthSeeker/get_main_ui',
                'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            'method': 'POST',
            'path': '/api/TruthSeeker/update_estimates',
            'query_params': {},
            'form_data': {
                'param1': 'value1',
                'param2': 'value2'
            }
        },
        'session': {
            'SiID': '29a2e258e18252e2afd5ff943523f09c82f1bb9adfe382a6f33fc6a8381de898',
            'level': '1',
            'spec': '74eed1c8de06886842e235486c3c2fd6bcd60586998ac5beb87f13c0d1750e1d',
            'user_name': 'root',
            'custom_field': 'custom_value'
        },
        'session_id': '0x29dd1ac0d1e30d3f'
    }

    # Parse the data
    parsed_data = parse_request_data(example_data)
    print(f"Session ID: {parsed_data.session_id}")
    print(f"Request Method: {parsed_data.request.method}")
    print(f"Request Path: {parsed_data.request.path}")
    print(f"User Name: {parsed_data.session.user_name}")

    # Access form data
    if parsed_data.request.form_data:
        print(f"Form Data: {parsed_data.request.form_data}")

    # Access headers
    print(f"User Agent: {parsed_data.request.headers.user_agent}")
    print(f"HX Request: {parsed_data.request.headers.hx_request}")

    # Convert back to dictionary
    data_dict = parsed_data.to_dict()
    print(f"Converted back to dictionary: {data_dict['request']['method']} {data_dict['request']['path']}")

    # Access extra session data
    if parsed_data.session.extra_data:
        print(f"Extra Session Data: {parsed_data.session.extra_data}")

@contextmanager
def profile_section(profiler, enable_profiling: bool):
    if enable_profiling:
        profiler.enable()
    try:
        yield
    finally:
        if enable_profiling:
            profiler.disable()


@dataclass
class ModuleInfo:
    functions_run: int = 0
    functions_fatal_error: int = 0
    error: int = 0
    functions_sug: int = 0
    calls: dict[str, list[Any]] = field(default_factory=dict)
    callse: dict[str, list[Any]] = field(default_factory=dict)
    coverage: list[int] = field(default_factory=lambda: [0, 0])
    execution_time: float = 0.0
    profiling_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionStats:
    modular_run: int = 0
    modular_fatal_error: int = 0
    errors: int = 0
    modular_sug: int = 0
    coverage: list[str] = field(default_factory=list)
    total_coverage: dict = field(default_factory=dict)
    total_execution_time: float = 0.0
    profiling_data: dict[str, Any] = field(default_factory=dict)


class AppArgs:
    init = None
    init_file = 'init.config'
    get_version = False
    mm = False
    sm = False
    lm = False
    modi = 'cli'
    kill = False
    remote = False
    remote_direct_key = None
    background_application = False
    background_application_runner = False
    docker = False
    build = False
    install = None
    remove = None
    update = None
    name = 'main'
    port = 5000
    host = '0.0.0.0'
    load_all_mod_in_files = False
    mods_folder = 'toolboxv2.mods.'
    debug = None
    test = None
    profiler = None
    hot_reload = False
    live_application = True
    sysPrint = False
    kwargs = {}
    session = None

    def default(self):
        return self

    def set(self, name, value):
        setattr(self, name, value)
        return self


class ApiOb:
    token = ""
    data = {}

    def __init__(self, data=None, token=""):
        if data is None:
            data = {}
        self.data = data
        self.token = token

    def default(self):
        return self


class ToolBoxError(str, Enum):
    none = "none"
    input_error = "InputError"
    internal_error = "InternalError"
    custom_error = "CustomError"


class ToolBoxInterfaces(str, Enum):
    cli = "CLI"
    api = "API"
    remote = "REMOTE"
    native = "NATIVE"
    internal = "INTERNAL"
    future = "FUTURE"


@dataclass
class ToolBoxResult:
    data_to: ToolBoxInterfaces or str = field(default=ToolBoxInterfaces.cli)
    data_info: Any | None = field(default=None)
    data: Any | None = field(default=None)
    data_type: None | str= field(default=None)


@dataclass
class ToolBoxInfo:
    exec_code: int
    help_text: str


class ToolBoxResultBM(BaseModel):
    data_to: str = ToolBoxInterfaces.cli.value
    data_info: str | None
    data: Any | None
    data_type: str | None


class ToolBoxInfoBM(BaseModel):
    exec_code: int
    help_text: str


class ApiResult(BaseModel):
    error: None | str= None
    origin: Any | None
    result: ToolBoxResultBM | None = None
    info: ToolBoxInfoBM | None

    def as_result(self):
        return Result(
            error=self.error.value if isinstance(self.error, Enum) else self.error,
            result=ToolBoxResult(
                data_to=self.result.data_to.value if isinstance(self.result.data_to, Enum) else self.result.data_to,
                data_info=self.result.data_info,
                data=self.result.data,
                data_type=self.result.data_type
            ) if self.result else None,
            info=ToolBoxInfo(
                exec_code=self.info.exec_code,
                help_text=self.info.help_text
            ) if self.info else None,
            origin=self.origin
        )

    def to_api_result(self):
        return self

    def print(self, *args, **kwargs):
        res = self.as_result().print(*args, **kwargs)
        if not isinstance(res, str):
            res = res.to_api_result()
        return res


class Result:
    _task = None
    def __init__(self,
                 error: ToolBoxError,
                 result: ToolBoxResult,
                 info: ToolBoxInfo,
                 origin: Any | None = None,
                 ):
        self.error: ToolBoxError = error
        self.result: ToolBoxResult = result
        self.info: ToolBoxInfo = info
        self.origin = origin

    def as_result(self):
        return self

    def as_dict(self):
        return {
            "error":self.error.value if isinstance(self.error, Enum) else self.error,
        "result" : {
            "data_to":self.result.data_to.value if isinstance(self.result.data_to, Enum) else self.result.data_to,
            "data_info":self.result.data_info,
            "data":self.result.data,
            "data_type":self.result.data_type
        } if self.result else None,
        "info" : {
            "exec_code" : self.info.exec_code,  # exec_code umwandel in http resposn codes
        "help_text" : self.info.help_text
        } if self.info else None,
        "origin" : self.origin
        }

    def set_origin(self, origin):
        if self.origin is not None:
            raise ValueError("You cannot Change the origin of a Result!")
        self.origin = origin
        return self

    def set_dir_origin(self, name, extras="assets/"):
        if self.origin is not None:
            raise ValueError("You cannot Change the origin of a Result!")
        self.origin = f"mods/{name}/{extras}"
        return self

    def is_error(self):
        if _test_is_result(self.result.data):
            return self.result.data.is_error()
        return self.info.exec_code != 0

    def is_data(self):
        return self.result.data is not None

    def to_api_result(self):
        # print(f" error={self.error}, result= {self.result}, info= {self.info}, origin= {self.origin}")
        return ApiResult(
            error=self.error.value if isinstance(self.error, Enum) else self.error,
            result=ToolBoxResultBM(
                data_to=self.result.data_to.value if isinstance(self.result.data_to, Enum) else self.result.data_to,
                data_info=self.result.data_info,
                data=self.result.data,
                data_type=self.result.data_type
            ) if self.result else None,
            info=ToolBoxInfoBM(
                exec_code=self.info.exec_code,  # exec_code umwandel in http resposn codes
                help_text=self.info.help_text
            ) if self.info else None,
            origin=self.origin
        )

    def task(self, task):
        self._task = task
        return self

    @staticmethod
    def result_from_dict(error: str, result: dict, info: dict, origin: list or None or str):
        # print(f" error={self.error}, result= {self.result}, info= {self.info}, origin= {self.origin}")
        return ApiResult(
            error=error if isinstance(error, Enum) else error,
            result=ToolBoxResultBM(
                data_to=result.get('data_to') if isinstance(result.get('data_to'), Enum) else result.get('data_to'),
                data_info=result.get('data_info', '404'),
                data=result.get('data'),
                data_type=result.get('data_type', '404'),
            ) if result else None,
            info=ToolBoxInfoBM(
                exec_code=info.get('exec_code', 404),
                help_text=info.get('help_text', '404')
            ) if info else None,
            origin=origin
        ).as_result()

    @classmethod
    def stream(cls,
               stream_generator,
               content_type="text/event-stream",
               headers=None,
               info="OK",
               interface=ToolBoxInterfaces.remote,
               cleanup_func=None):
        """
        Create a streaming response Result that properly handles all types of stream sources.

        Args:
            stream_generator: Any stream source (async generator, sync generator, iterable, or even string)
            content_type: Content-Type header (default: text/event-stream for SSE)
            headers: Additional HTTP headers
            info: Help text for the result
            interface: Interface to send data to

        Returns:
            A Result object configured for streaming
        """
        error = ToolBoxError.none
        info_obj = ToolBoxInfo(exec_code=0, help_text=info)

        # Standard SSE headers
        standard_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }

        # Apply custom headers
        all_headers = standard_headers.copy()
        if headers:
            all_headers.update(headers)

        # Handle different types of stream sources
        if content_type == "text/event-stream":
            wrapped_generator = stream_generator
            if inspect.isgenerator(stream_generator) or hasattr(stream_generator, '__iter__'):
                # Sync generator or iterable
                wrapped_generator = SSEGenerator.create_sse_stream(stream_generator, cleanup_func)

            elif isinstance(stream_generator, str):
                # String (could be a memory address or other reference)
                # Convert to a generator that yields a single string
                async def string_to_stream():
                    yield stream_generator

                wrapped_generator = SSEGenerator.create_sse_stream(string_to_stream(), cleanup_func)

            # The final generator to use
            final_generator = wrapped_generator

        else:
            # For non-SSE streams, use the original generator
            final_generator = stream_generator

        # Prepare streaming data
        streaming_data = {
            "type": "stream",
            "generator": final_generator,
            "content_type": content_type,
            "headers": all_headers
        }

        result = ToolBoxResult(
            data_to=interface,
            data=streaming_data,
            data_info="Streaming response",
            data_type="stream"
        )

        return cls(error=error, info=info_obj, result=result)

    @classmethod
    def default(cls, interface=ToolBoxInterfaces.native):
        error = ToolBoxError.none
        info = ToolBoxInfo(exec_code=-1, help_text="")
        result = ToolBoxResult(data_to=interface)
        return cls(error=error, info=info, result=result)

    @classmethod
    def json(cls, data, info="OK", interface=ToolBoxInterfaces.remote):
        """Create a JSON response Result."""
        error = ToolBoxError.none
        info_obj = ToolBoxInfo(exec_code=0, help_text=info)

        result = ToolBoxResult(
            data_to=interface,
            data=data,
            data_info="JSON response",
            data_type="json"
        )

        return cls(error=error, info=info_obj, result=result)

    @classmethod
    def text(cls, text_data, content_type="text/plain",exec_code=None,status=200, info="OK", interface=ToolBoxInterfaces.remote, headers=None):
        """Create a text response Result with specific content type."""
        if headers is not None:
            return cls.html(text_data, status= exec_code or status, info=info, headers=headers)
        error = ToolBoxError.none
        info_obj = ToolBoxInfo(exec_code=exec_code or status, help_text=info)

        result = ToolBoxResult(
            data_to=interface,
            data=text_data,
            data_info="Text response",
            data_type=content_type
        )

        return cls(error=error, info=info_obj, result=result)

    @classmethod
    def binary(cls, data, content_type="application/octet-stream", download_name=None, info="OK",
               interface=ToolBoxInterfaces.remote):
        """Create a binary data response Result."""
        error = ToolBoxError.none
        info_obj = ToolBoxInfo(exec_code=0, help_text=info)

        # Create a dictionary with binary data and metadata
        binary_data = {
            "data": data,
            "content_type": content_type,
            "filename": download_name
        }

        result = ToolBoxResult(
            data_to=interface,
            data=binary_data,
            data_info=f"Binary response: {download_name}" if download_name else "Binary response",
            data_type="binary"
        )

        return cls(error=error, info=info_obj, result=result)

    @classmethod
    def redirect(cls, url, status_code=302, info="Redirect", interface=ToolBoxInterfaces.remote):
        """Create a redirect response."""
        error = ToolBoxError.none
        info_obj = ToolBoxInfo(exec_code=status_code, help_text=info)

        result = ToolBoxResult(
            data_to=interface,
            data=url,
            data_info="Redirect response",
            data_type="redirect"
        )

        return cls(error=error, info=info_obj, result=result)

    @classmethod
    def ok(cls, data=None, data_info="", info="OK", interface=ToolBoxInterfaces.native):
        error = ToolBoxError.none
        info = ToolBoxInfo(exec_code=0, help_text=info)
        result = ToolBoxResult(data_to=interface, data=data, data_info=data_info, data_type=type(data).__name__)
        return cls(error=error, info=info, result=result)

    @classmethod
    def html(cls, data=None, data_info="", info="OK", interface=ToolBoxInterfaces.remote, data_type="html",status=200, headers=None):
        error = ToolBoxError.none
        info = ToolBoxInfo(exec_code=status, help_text=info)

        if isinstance(headers, dict):
            result = ToolBoxResult(data_to=interface, data={'html':data,'headers':headers}, data_info=data_info,
                                   data_type="special_html")
        else:
            result = ToolBoxResult(data_to=interface, data=data, data_info=data_info,
                                   data_type=data_type if data_type is not None else type(data).__name__)
        return cls(error=error, info=info, result=result)

    @classmethod
    def future(cls, data=None, data_info="", info="OK", interface=ToolBoxInterfaces.future):
        error = ToolBoxError.none
        info = ToolBoxInfo(exec_code=0, help_text=info)
        result = ToolBoxResult(data_to=interface, data=data, data_info=data_info, data_type="future")
        return cls(error=error, info=info, result=result)

    @classmethod
    def custom_error(cls, data=None, data_info="", info="", exec_code=-1, interface=ToolBoxInterfaces.native):
        error = ToolBoxError.custom_error
        info = ToolBoxInfo(exec_code=exec_code, help_text=info)
        result = ToolBoxResult(data_to=interface, data=data, data_info=data_info, data_type=type(data).__name__)
        return cls(error=error, info=info, result=result)

    @classmethod
    def error(cls, data=None, data_info="", info="", exec_code=450, interface=ToolBoxInterfaces.remote):
        error = ToolBoxError.custom_error
        info = ToolBoxInfo(exec_code=exec_code, help_text=info)
        result = ToolBoxResult(data_to=interface, data=data, data_info=data_info, data_type=type(data).__name__)
        return cls(error=error, info=info, result=result)

    @classmethod
    def default_user_error(cls, info="", exec_code=-3, interface=ToolBoxInterfaces.native, data=None):
        error = ToolBoxError.input_error
        info = ToolBoxInfo(exec_code, info)
        result = ToolBoxResult(data_to=interface, data=data, data_type=type(data).__name__)
        return cls(error=error, info=info, result=result)

    @classmethod
    def default_internal_error(cls, info="", exec_code=-2, interface=ToolBoxInterfaces.native, data=None):
        error = ToolBoxError.internal_error
        info = ToolBoxInfo(exec_code, info)
        result = ToolBoxResult(data_to=interface, data=data, data_type=type(data).__name__)
        return cls(error=error, info=info, result=result)

    def print(self, show=True, show_data=True, prifix=""):
        data = '\n' + f"{((prifix + 'Data: ' + str(self.result.data) if self.result.data is not None else 'NO Data') if not isinstance(self.result.data, Result) else self.result.data.print(show=False, show_data=show_data, prifix=prifix + '-')) if show_data else 'Data: private'}"
        origin = '\n' + f"{prifix + 'Origin: ' + str(self.origin) if self.origin is not None else 'NO Origin'}"
        text = (f"Function Exec code: {self.info.exec_code}"
                f"\n{prifix}Info's:"
                f" {self.info.help_text} {'<|> ' + str(self.result.data_info) if self.result.data_info is not None else ''}"
                f"{origin}{data if not data.endswith('NO Data') else ''}")
        if not show:
            return text
        print("\n======== Result ========\n" + text + "\n------- EndOfD -------")
        return self

    def log(self, show_data=True, prifix=""):
        from toolboxv2 import get_logger
        get_logger().debug(self.print(show=False, show_data=show_data, prifix=prifix).replace("\n", " - "))
        return self

    def __str__(self):
        return self.print(show=False, show_data=True)

    def get(self, key=None, default=None):
        data = self.result.data
        if isinstance(data, Result):
            return data.get(key=key, default=default)
        if key is not None and isinstance(data, dict):
            return data.get(key, default)
        return data if data is not None else default

    async def aget(self, key=None, default=None):
        if asyncio.isfuture(self.result.data) or asyncio.iscoroutine(self.result.data) or (
            isinstance(self.result.data_to, Enum) and self.result.data_to.name == ToolBoxInterfaces.future.name):
            data = await self.result.data
        else:
            data = self.get(key=None, default=None)
        if isinstance(data, Result):
            return data.get(key=key, default=default)
        if key is not None and isinstance(data, dict):
            return data.get(key, default)
        return data if data is not None else default

    def lazy_return(self, _=0, data=None, **kwargs):
        flags = ['raise', 'logg', 'user', 'intern']
        flag = flags[_] if isinstance(_, int) else _
        if self.info.exec_code == 0:
            return self if data is None else data if _test_is_result(data) else self.ok(data=data, **kwargs)
        if flag == 'raise':
            raise ValueError(self.print(show=False))
        if flag == 'logg':
            from .. import get_logger
            get_logger().error(self.print(show=False))

        if flag == 'user':
            return self if data is None else data if _test_is_result(data) else self.default_user_error(data=data,
                                                                                                        **kwargs)
        if flag == 'intern':
            return self if data is None else data if _test_is_result(data) else self.default_internal_error(data=data,
                                                                                                            **kwargs)

        return self if data is None else data if _test_is_result(data) else self.custom_error(data=data, **kwargs)

    @property
    def bg_task(self):
        return self._task


def _test_is_result(data: Result):
    return isinstance(data, Result)


@dataclass
class CallingObject:
    module_name: str = field(default="")
    function_name: str = field(default="")
    args: list or None = field(default=None)
    kwargs: dict or None = field(default=None)

    @classmethod
    def empty(cls):
        return cls()

    def __str__(self):
        if self.args is not None and self.kwargs is not None:
            return (f"{self.module_name} {self.function_name} " + ' '.join(self.args) + ' ' +
                    ' '.join([key + '-' + str(val) for key, val in self.kwargs.items()]))
        if self.args is not None:
            return f"{self.module_name} {self.function_name} " + ' '.join(self.args)
        return f"{self.module_name} {self.function_name}"

    def print(self, show=True):
        s = f"{self.module_name=};{self.function_name=};{self.args=};{self.kwargs=}"
        if not show:
            return s
        print(s)


def analyze_data(data):
    report = []
    for mod_name, mod_info in data.items():
        if mod_name in ['modular_run', 'modular_fatal_error', 'modular_sug']:
            continue  # Überspringen der allgemeinen Statistiken
        if mod_name in ['errors']:
            report.append(f"Total errors: {mod_info}")
            continue
        if mod_name == 'total_coverage':
            continue
        if mod_name == 'coverage':
            _ = '\t'.join(mod_info)
            report.append(f"Total coverage:\n {_}")
            continue
        report.append(f"Modul: {mod_name}")
        if not isinstance(mod_info, dict):
            report.append(f"info: {mod_info}")
            continue
        report.append(f"  Funktionen ausgeführt: {mod_info.get('functions_run', 0)}")
        report.append(f"  Funktionen mit Fatalen Fehler: {mod_info.get('functions_fatal_error', 0)}")
        report.append(f"  Funktionen mit Fehler: {mod_info.get('error', 0)}")
        report.append(f"  Funktionen erfolgreich: {mod_info.get('functions_sug', 0)}")
        if mod_info.get('coverage', [0])[0] == 0:
            c = 0
        else:
            c = mod_info.get('coverage', [0, 1])[1] / mod_info.get('coverage', [1])[0]
        report.append(f"  coverage: {c:.2f}")

        if 'callse' in mod_info and mod_info['callse']:
            report.append("  Fehler:")
            for func_name, errors in mod_info['callse'].items():
                for error in errors:
                    if isinstance(error, str):
                        error = error.replace('\n', ' - ')
                    report.append(f"    - {func_name}, Fehler: {error}")
    return "\n".join(report)


U = Any
A = Any


class MainToolType:
    toolID: str
    app: A
    interface: ToolBoxInterfaces
    spec: str

    version: str
    tools: dict  # legacy
    name: str
    logger: logging
    color: str
    todo: Callable
    _on_exit: Callable
    stuf: bool
    config: dict
    user: U | None
    description: str

    @staticmethod
    def return_result(error: ToolBoxError = ToolBoxError.none,
                      exec_code: int = 0,
                      help_text: str = "",
                      data_info=None,
                      data=None,
                      data_to=None) -> Result:
        """proxi attr"""

    def load(self):
        """proxi attr"""

    def print(self, message, end="\n", **kwargs):
        """proxi attr"""

    def add_str_to_config(self, command):
        if len(command) != 2:
            self.logger.error('Invalid command must be key value')
            return False
        self.config[command[0]] = command[1]

    def webInstall(self, user_instance, construct_render) -> str:
        """"Returns a web installer for the given user instance and construct render template"""

    async def get_user(self, username: str) -> Result:
        return self.app.a_run_any(CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=username, get_results=True)


class AppType:
    prefix: str
    id: str
    globals: dict[str, Any] = {"root": dict, }
    locals: dict[str, Any] = {"user": {'app': "self"}, }

    local_test: bool = False
    start_dir: str
    data_dir: str
    config_dir: str
    info_dir: str

    logger: logging.Logger
    logging_filename: str

    api_allowed_mods_list: list[str] = []

    version: str
    loop: asyncio.AbstractEventLoop

    keys: dict[str, str] = {
        "MACRO": "macro~~~~:",
        "MACRO_C": "m_color~~:",
        "HELPER": "helper~~~:",
        "debug": "debug~~~~:",
        "id": "name-spa~:",
        "st-load": "mute~load:",
        "comm-his": "comm-his~:",
        "develop-mode": "dev~mode~:",
        "provider::": "provider::",
    }

    defaults: dict[str, (bool or dict or dict[str, dict[str, str]] or str or list[str] or list[list]) | None] = {
        "MACRO": list[str],
        "MACRO_C": dict,
        "HELPER": dict,
        "debug": str,
        "id": str,
        "st-load": False,
        "comm-his": list[list],
        "develop-mode": bool,
    }

    config_fh: FileHandler
    _debug: bool
    flows: dict[str, Callable]
    dev_modi: bool
    functions: dict[str, Any]
    modules: dict[str, Any]

    interface_type: ToolBoxInterfaces
    REFIX: str

    alive: bool
    called_exit: tuple[bool, float]
    args_sto: AppArgs
    system_flag = None
    session = None
    appdata = None
    exit_tasks = []

    enable_profiling: bool = False
    sto = None

    def __init__(self, prefix: None | str= None, args: AppArgs | None = None):
        self.args_sto = args
        self.prefix = prefix
        """proxi attr"""

    @staticmethod
    def exit_main(*args, **kwargs):
        """proxi attr"""

    @staticmethod
    async def hide_console(*args, **kwargs):
        """proxi attr"""

    @staticmethod
    async def show_console(*args, **kwargs):
        """proxi attr"""

    @staticmethod
    async def disconnect(*args, **kwargs):
        """proxi attr"""

    def set_logger(self, debug=False):
        """proxi attr"""

    @property
    def debug(self):
        """proxi attr"""
        return self._debug

    def debug_rains(self, e):
        """proxi attr"""

    def set_flows(self, r):
        """proxi attr"""

    def run_flows(self, name, **kwargs):
        """proxi attr"""

    def rrun_flows(self, name, **kwargs):
        """proxi attr"""

    def idle(self):
        import time
        self.print("idle")
        try:
            while self.alive:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        self.print("idle done")

    async def a_idle(self):
        self.print("a idle")
        try:
            if hasattr(self, 'daemon_app'):
                self.print("serving daemon")
                await self.daemon_app.connect(self)
            else:
                self.print("serving default")
                while self.alive:
                    await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        self.print("a idle done")

    @debug.setter
    def debug(self, value):
        """proxi attr"""

    def _coppy_mod(self, content, new_mod_dir, mod_name, file_type='py'):
        """proxi attr"""

    def _pre_lib_mod(self, mod_name, path_to="./runtime", file_type='py'):
        """proxi attr"""

    def _copy_load(self, mod_name, file_type='py', **kwargs):
        """proxi attr"""

    def inplace_load_instance(self, mod_name, loc="toolboxv2.mods.", spec='app', save=True):
        """proxi attr"""

    def save_instance(self, instance, modular_id, spec='app', instance_type="file/application", tools_class=None):
        """proxi attr"""

    def save_initialized_module(self, tools_class, spec):
        """proxi attr"""

    def mod_online(self, mod_name, installed=False):
        """proxi attr"""

    def _get_function(self,
                      name: Enum or None,
                      state: bool = True,
                      specification: str = "app",
                      metadata=False, as_str: tuple or None = None, r=0):
        """proxi attr"""

    def save_exit(self):
        """proxi attr"""

    def load_mod(self, mod_name: str, mlm='I', **kwargs):
        """proxi attr"""

    async def init_module(self, modular):
        return await self.load_mod(modular)

    async def load_all_mods_in_file(self, working_dir="mods"):
        """proxi attr"""

    def get_all_mods(self, working_dir="mods", path_to="./runtime"):
        """proxi attr"""

    def remove_all_modules(self, delete=False):
        for mod in list(self.functions.keys()):
            self.logger.info(f"closing: {mod}")
            self.remove_mod(mod, delete=delete)

    async def a_remove_all_modules(self, delete=False):
        for mod in list(self.functions.keys()):
            self.logger.info(f"closing: {mod}")
            await self.a_remove_mod(mod, delete=delete)

    def print_ok(self):
        """proxi attr"""
        self.logger.info("OK")

    def reload_mod(self, mod_name, spec='app', is_file=True, loc="toolboxv2.mods."):
        """proxi attr"""

    def watch_mod(self, mod_name, spec='app', loc="toolboxv2.mods.", use_thread=True, path_name=None):
        """proxi attr"""

    def remove_mod(self, mod_name, spec='app', delete=True):
        """proxi attr"""

    async def a_remove_mod(self, mod_name, spec='app', delete=True):
        """proxi attr"""

    def exit(self):
        """proxi attr"""

    def web_context(self) -> str:
        """returns the build index ( toolbox web component )"""

    async def a_exit(self):
        """proxi attr"""

    def save_load(self, modname, spec='app'):
        """proxi attr"""

    def get_function(self, name: Enum or tuple, **kwargs):
        """
        Kwargs for _get_function
            metadata:: return the registered function dictionary
                stateless: (function_data, None), 0
                stateful: (function_data, higher_order_function), 0
            state::boolean
                specification::str default app
        """

    def run_a_from_sync(self, function, *args):
        """
        run a async fuction
        """

    def run_function(self, mod_function_name: Enum or tuple,
                     tb_run_function_with_state=True,
                     tb_run_with_specification='app',
                     args_=None,
                     kwargs_=None,
                     *args,
                     **kwargs) -> Result:

        """proxi attr"""

    async def a_run_function(self, mod_function_name: Enum or tuple,
                             tb_run_function_with_state=True,
                             tb_run_with_specification='app',
                             args_=None,
                             kwargs_=None,
                             *args,
                             **kwargs) -> Result:

        """proxi attr"""

    def fuction_runner(self, function, function_data: dict, args: list, kwargs: dict, t0=.0):
        """
        parameters = function_data.get('params')
        modular_name = function_data.get('module_name')
        function_name = function_data.get('func_name')
        mod_function_name = f"{modular_name}.{function_name}"

        proxi attr
        """

    async def a_fuction_runner(self, function, function_data: dict, args: list, kwargs: dict):
        """
        parameters = function_data.get('params')
        modular_name = function_data.get('module_name')
        function_name = function_data.get('func_name')
        mod_function_name = f"{modular_name}.{function_name}"

        proxi attr
        """

    async def run_http(self, mod_function_name: Enum or str or tuple, function_name=None, method="GET",
                       args_=None,
                       kwargs_=None,
                       *args, **kwargs):
        """run a function remote via http / https"""

    def run_any(self, mod_function_name: Enum or str or tuple, backwords_compability_variabel_string_holder=None,
                get_results=False, tb_run_function_with_state=True, tb_run_with_specification='app', args_=None,
                kwargs_=None,
                *args, **kwargs):
        """proxi attr"""

    async def a_run_any(self, mod_function_name: Enum or str or tuple,
                        backwords_compability_variabel_string_holder=None,
                        get_results=False, tb_run_function_with_state=True, tb_run_with_specification='app', args_=None,
                        kwargs_=None,
                        *args, **kwargs):
        """proxi attr"""

    def get_mod(self, name, spec='app') -> ModuleType or MainToolType:
        """proxi attr"""

    @staticmethod
    def print(text, *args, **kwargs):
        """proxi attr"""

    @staticmethod
    def sprint(text, *args, **kwargs):
        """proxi attr"""

    # ----------------------------------------------------------------
    # Decorators for the toolbox

    def _register_function(self, module_name, func_name, data):
        """proxi attr"""

    def _create_decorator(self, type_: str,
                          name: str = "",
                          mod_name: str = "",
                          level: int = -1,
                          restrict_in_virtual_mode: bool = False,
                          api: bool = False,
                          helper: str = "",
                          version: str or None = None,
                          initial=False,
                          exit_f=False,
                          test=True,
                          samples=None,
                          state=None,
                          pre_compute=None,
                          post_compute=None,
                          memory_cache=False,
                          file_cache=False,
                          row=False,
                          request_as_kwarg=False,
                          memory_cache_max_size=100,
                          memory_cache_ttl=300):
        """proxi attr"""

        # data = {
        #     "type": type_,
        #     "module_name": module_name,
        #     "func_name": func_name,
        #     "level": level,
        #     "restrict_in_virtual_mode": restrict_in_virtual_mode,
        #     "func": func,
        #     "api": api,
        #     "helper": helper,
        #     "version": version,
        #     "initial": initial,
        #     "exit_f": exit_f,
        #     "__module__": func.__module__,
        #     "signature": sig,
        #     "params": params,
        #     "state": (
        #         False if len(params) == 0 else params[0] in ['self', 'state', 'app']) if state is None else state,
        #     "do_test": test,
        #     "samples": samples,
        #     "request_as_kwarg": request_as_kwarg,

    def tb(self, name=None,
           mod_name: str = "",
           helper: str = "",
           version: str or None = None,
           test: bool = True,
           restrict_in_virtual_mode: bool = False,
           api: bool = False,
           initial: bool = False,
           exit_f: bool = False,
           test_only: bool = False,
           memory_cache: bool = False,
           file_cache: bool = False,
           row=False,
           request_as_kwarg: bool = False,
           state: bool or None = None,
           level: int = 0,
           memory_cache_max_size: int = 100,
           memory_cache_ttl: int = 300,
           samples: list or dict or None = None,
           interface: ToolBoxInterfaces or None or str = None,
           pre_compute=None,
           post_compute=None,
           api_methods=None,
           ):
        """
    A decorator for registering and configuring functions within a module.

    This decorator is used to wrap functions with additional functionality such as caching, API conversion, and lifecycle management (initialization and exit). It also handles the registration of the function in the module's function registry.

    Args:
        name (str, optional): The name to register the function under. Defaults to the function's own name.
        mod_name (str, optional): The name of the module the function belongs to.
        helper (str, optional): A helper string providing additional information about the function.
        version (str or None, optional): The version of the function or module.
        test (bool, optional): Flag to indicate if the function is for testing purposes.
        restrict_in_virtual_mode (bool, optional): Flag to restrict the function in virtual mode.
        api (bool, optional): Flag to indicate if the function is part of an API.
        initial (bool, optional): Flag to indicate if the function should be executed at initialization.
        exit_f (bool, optional): Flag to indicate if the function should be executed at exit.
        test_only (bool, optional): Flag to indicate if the function should only be used for testing.
        memory_cache (bool, optional): Flag to enable memory caching for the function.
        request_as_kwarg (bool, optional): Flag to get request if the fuction is calld from api.
        file_cache (bool, optional): Flag to enable file caching for the function.
        row (bool, optional): rather to auto wrap the result in Result type default False means no row data aka result type
        state (bool or None, optional): Flag to indicate if the function maintains state.
        level (int, optional): The level of the function, used for prioritization or categorization.
        memory_cache_max_size (int, optional): Maximum size of the memory cache.
        memory_cache_ttl (int, optional): Time-to-live for the memory cache entries.
        samples (list or dict or None, optional): Samples or examples of function usage.
        interface (str, optional): The interface type for the function.
        pre_compute (callable, optional): A function to be called before the main function.
        post_compute (callable, optional): A function to be called after the main function.
        api_methods (list[str], optional): default ["AUTO"] (GET if not params, POST if params) , GET, POST, PUT or DELETE.

    Returns:
        function: The decorated function with additional processing and registration capabilities.
    """
        if interface is None:
            interface = "tb"
        if test_only and 'test' not in self.id:
            return lambda *args, **kwargs: args
        return self._create_decorator(interface,
                                      name,
                                      mod_name,
                                      level=level,
                                      restrict_in_virtual_mode=restrict_in_virtual_mode,
                                      helper=helper,
                                      api=api,
                                      version=version,
                                      initial=initial,
                                      exit_f=exit_f,
                                      test=test,
                                      samples=samples,
                                      state=state,
                                      pre_compute=pre_compute,
                                      post_compute=post_compute,
                                      memory_cache=memory_cache,
                                      file_cache=file_cache,
                                      row=row,
                                      request_as_kwarg=request_as_kwarg,
                                      memory_cache_max_size=memory_cache_max_size,
                                      memory_cache_ttl=memory_cache_ttl)

    def print_functions(self, name=None):


        if not self.functions:
            print("Nothing to see")
            return

        def helper(_functions):
            for func_name, data in _functions.items():
                if not isinstance(data, dict):
                    continue

                func_type = data.get('type', 'Unknown')
                func_level = 'r' if data['level'] == -1 else data['level']
                api_status = 'Api' if data.get('api', False) else 'Non-Api'

                print(f"  Function: {func_name}{data.get('signature', '()')}; "
                      f"Type: {func_type}, Level: {func_level}, {api_status}")

        if name is not None:
            functions = self.functions.get(name)
            if functions is not None:
                print(f"\nModule: {name}; Type: {functions.get('app_instance_type', 'Unknown')}")
                helper(functions)
                return
        for module, functions in self.functions.items():
            print(f"\nModule: {module}; Type: {functions.get('app_instance_type', 'Unknown')}")
            helper(functions)

    def save_autocompletion_dict(self):
        """proxi attr"""

    def get_autocompletion_dict(self):
        """proxi attr"""

    def get_username(self, get_input=False, default="loot") -> str:
        """proxi attr"""

    def save_registry_as_enums(self, directory: str, filename: str):
        """proxi attr"""

    async def execute_all_functions_(self, m_query='', f_query=''):
        print("Executing all functions")
        from ..extras import generate_test_cases
        all_data = {
            "modular_run": 0,
            "modular_fatal_error": 0,
            "errors": 0,
            "modular_sug": 0,
            "coverage": [],
            "total_coverage": {},
        }
        items = list(self.functions.items()).copy()
        for module_name, functions in items:
            infos = {
                "functions_run": 0,
                "functions_fatal_error": 0,
                "error": 0,
                "functions_sug": 0,
                'calls': {},
                'callse': {},
                "coverage": [0, 0],
            }
            all_data['modular_run'] += 1
            if not module_name.startswith(m_query):
                all_data['modular_sug'] += 1
                continue

            with Spinner(message=f"In {module_name}| "):
                f_items = list(functions.items()).copy()
                for function_name, function_data in f_items:
                    if not isinstance(function_data, dict):
                        continue
                    if not function_name.startswith(f_query):
                        continue
                    test: list = function_data.get('do_test')
                    # print(test, module_name, function_name, function_data)
                    infos["coverage"][0] += 1
                    if test is False:
                        continue

                    with Spinner(message=f"\t\t\t\t\t\tfuction {function_name}..."):
                        params: list = function_data.get('params')
                        sig: signature = function_data.get('signature')
                        state: bool = function_data.get('state')
                        samples: bool = function_data.get('samples')

                        test_kwargs_list = [{}]

                        if params is not None:
                            test_kwargs_list = samples if samples is not None else generate_test_cases(sig=sig)
                            # print(test_kwargs)
                            # print(test_kwargs[0])
                            # test_kwargs = test_kwargs_list[0]
                        # print(module_name, function_name, test_kwargs_list)
                        infos["coverage"][1] += 1
                        for test_kwargs in test_kwargs_list:
                            try:
                                # print(f"test Running {state=} |{module_name}.{function_name}")
                                result = await self.a_run_function((module_name, function_name),
                                                                   tb_run_function_with_state=state,
                                                                   **test_kwargs)
                                if not isinstance(result, Result):
                                    result = Result.ok(result)
                                if result.info.exec_code == 0:
                                    infos['calls'][function_name] = [test_kwargs, str(result)]
                                    infos['functions_sug'] += 1
                                else:
                                    infos['functions_sug'] += 1
                                    infos['error'] += 1
                                    infos['callse'][function_name] = [test_kwargs, str(result)]
                            except Exception as e:
                                infos['functions_fatal_error'] += 1
                                infos['callse'][function_name] = [test_kwargs, str(e)]
                            finally:
                                infos['functions_run'] += 1

                if infos['functions_run'] == infos['functions_sug']:
                    all_data['modular_sug'] += 1
                else:
                    all_data['modular_fatal_error'] += 1
                if infos['error'] > 0:
                    all_data['errors'] += infos['error']

                all_data[module_name] = infos
                if infos['coverage'][0] == 0:
                    c = 0
                else:
                    c = infos['coverage'][1] / infos['coverage'][0]
                all_data["coverage"].append(f"{module_name}:{c:.2f}\n")
        total_coverage = sum([float(t.split(":")[-1]) for t in all_data["coverage"]]) / len(all_data["coverage"])
        print(
            f"\n{all_data['modular_run']=}\n{all_data['modular_sug']=}\n{all_data['modular_fatal_error']=}\n{total_coverage=}")
        d = analyze_data(all_data)
        return Result.ok(data=all_data, data_info=d)

    @staticmethod
    def calculate_complexity(filename_or_code):
        from radon.complexity import cc_rank, cc_visit
        if os.path.exists(filename_or_code):
            with open(filename_or_code) as file:
                code = file.read()
        else:
            code = filename_or_code

        # Calculate and print Cyclomatic Complexity
        complexity_results = cc_visit(code)
        i = -1
        avg_complexity = 0
        for block in complexity_results:
            complexity = block.complexity
            i += 1
            print(f"block: {block.name} {i} Class/Fuction/Methode : {block.letter}")
            print(f"    fullname: {block.fullname}")
            print(f"    Cyclomatic Complexity: {complexity}")
            # Optional: Get complexity rank
            avg_complexity += complexity
            rank = cc_rank(complexity)
            print(f"    Complexity Rank: {rank}")
            # print(f"    lineno: {block.lineno}")
            print(f"    endline: {block.endline}")
            print(f"    col_offset: {block.col_offset}\n")
        if i <= 0:
            i += 2
        avg_complexity = avg_complexity / i
        print(f"\nAVG Complexity: {avg_complexity:.2f}")
        print(f"Total Rank: {cc_rank(int(avg_complexity + i // 10))}")

    async def execute_function_test(self, module_name: str, function_name: str,
                                    function_data: dict, test_kwargs: dict,
                                    profiler: cProfile.Profile) -> tuple[bool, str, dict, float]:
        start_time = time.time()
        with profile_section(profiler, hasattr(self, 'enable_profiling') and self.enable_profiling):
            try:
                result = await self.a_run_function(
                    (module_name, function_name),
                    tb_run_function_with_state=function_data.get('state'),
                    **test_kwargs
                )

                if not isinstance(result, Result):
                    result = Result.ok(result)

                success = result.info.exec_code == 0
                execution_time = time.time() - start_time
                return success, str(result), test_kwargs, execution_time
            except Exception as e:
                execution_time = time.time() - start_time
                return False, str(e), test_kwargs, execution_time

    async def process_function(self, module_name: str, function_name: str,
                               function_data: dict, profiler: cProfile.Profile) -> tuple[str, ModuleInfo]:
        start_time = time.time()
        info = ModuleInfo()

        with profile_section(profiler, hasattr(self, 'enable_profiling') and self.enable_profiling):
            if not isinstance(function_data, dict):
                return function_name, info

            test = function_data.get('do_test')
            info.coverage[0] += 1

            if test is False:
                return function_name, info

            params = function_data.get('params')
            sig = function_data.get('signature')
            samples = function_data.get('samples')

            test_kwargs_list = [{}] if params is None else (
                samples if samples is not None else generate_test_cases(sig=sig)
            )

            info.coverage[1] += 1

            # Create tasks for all test cases
            tasks = [
                self.execute_function_test(module_name, function_name, function_data, test_kwargs, profiler)
                for test_kwargs in test_kwargs_list
            ]

            # Execute all tests concurrently
            results = await asyncio.gather(*tasks)

            total_execution_time = 0
            for success, result_str, test_kwargs, execution_time in results:
                info.functions_run += 1
                total_execution_time += execution_time

                if success:
                    info.functions_sug += 1
                    info.calls[function_name] = [test_kwargs, result_str]
                else:
                    info.functions_sug += 1
                    info.error += 1
                    info.callse[function_name] = [test_kwargs, result_str]

            info.execution_time = time.time() - start_time
            return function_name, info

    async def process_module(self, module_name: str, functions: dict,
                             f_query: str, profiler: cProfile.Profile) -> tuple[str, ModuleInfo]:
        start_time = time.time()

        with profile_section(profiler, hasattr(self, 'enable_profiling') and self.enable_profiling):
            async with asyncio.Semaphore(mp.cpu_count()):
                tasks = [
                    self.process_function(module_name, fname, fdata, profiler)
                    for fname, fdata in functions.items()
                    if fname.startswith(f_query)
                ]

                if not tasks:
                    return module_name, ModuleInfo()

                results = await asyncio.gather(*tasks)

                # Combine results from all functions in the module
                combined_info = ModuleInfo()
                total_execution_time = 0

                for _, info in results:
                    combined_info.functions_run += info.functions_run
                    combined_info.functions_fatal_error += info.functions_fatal_error
                    combined_info.error += info.error
                    combined_info.functions_sug += info.functions_sug
                    combined_info.calls.update(info.calls)
                    combined_info.callse.update(info.callse)
                    combined_info.coverage[0] += info.coverage[0]
                    combined_info.coverage[1] += info.coverage[1]
                    total_execution_time += info.execution_time

                combined_info.execution_time = time.time() - start_time
                return module_name, combined_info

    async def execute_all_functions(self, m_query='', f_query='', enable_profiling=True):
        """
        Execute all functions with parallel processing and optional profiling.

        Args:
            m_query (str): Module name query filter
            f_query (str): Function name query filter
            enable_profiling (bool): Enable detailed profiling information
        """
        print("Executing all functions in parallel" + (" with profiling" if enable_profiling else ""))

        start_time = time.time()
        stats = ExecutionStats()
        items = list(self.functions.items()).copy()

        # Set up profiling
        self.enable_profiling = enable_profiling
        profiler = cProfile.Profile()

        with profile_section(profiler, enable_profiling):
            # Filter modules based on query
            filtered_modules = [
                (mname, mfuncs) for mname, mfuncs in items
                if mname.startswith(m_query)
            ]

            stats.modular_run = len(filtered_modules)

            # Process all modules concurrently
            async with asyncio.Semaphore(mp.cpu_count()):
                tasks = [
                    self.process_module(mname, mfuncs, f_query, profiler)
                    for mname, mfuncs in filtered_modules
                ]

                results = await asyncio.gather(*tasks)

            # Combine results and calculate statistics
            for module_name, info in results:
                if info.functions_run == info.functions_sug:
                    stats.modular_sug += 1
                else:
                    stats.modular_fatal_error += 1

                stats.errors += info.error

                # Calculate coverage
                coverage = (info.coverage[1] / info.coverage[0]) if info.coverage[0] > 0 else 0
                stats.coverage.append(f"{module_name}:{coverage:.2f}\n")

                # Store module info
                stats.__dict__[module_name] = info

            # Calculate total coverage
            total_coverage = (
                sum(float(t.split(":")[-1]) for t in stats.coverage) / len(stats.coverage)
                if stats.coverage else 0
            )

            stats.total_execution_time = time.time() - start_time

            # Generate profiling stats if enabled
            if enable_profiling:
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats()
                stats.profiling_data = {
                    'detailed_stats': s.getvalue(),
                    'total_time': stats.total_execution_time,
                    'function_count': stats.modular_run,
                    'successful_functions': stats.modular_sug
                }

            print(
                f"\n{stats.modular_run=}"
                f"\n{stats.modular_sug=}"
                f"\n{stats.modular_fatal_error=}"
                f"\n{total_coverage=}"
                f"\nTotal execution time: {stats.total_execution_time:.2f}s"
            )

            if enable_profiling:
                print("\nProfiling Summary:")
                print(f"{'=' * 50}")
                print("Top 10 time-consuming functions:")
                ps.print_stats(10)

            analyzed_data = analyze_data(stats.__dict__)
            return Result.ok(data=stats.__dict__, data_info=analyzed_data)


import asyncio
import base64
import inspect
import json
import traceback
from collections.abc import AsyncGenerator, Callable
from typing import Any, TypeVar

T = TypeVar('T')


class SSEGenerator:
    """
    Production-ready SSE generator that converts any data source to
    properly formatted Server-Sent Events compatible with browsers.
    """

    @staticmethod
    def format_sse_event(data: Any) -> str:
        """Format any data as a proper SSE event message."""
        # Already formatted as SSE
        if isinstance(data, str) and (data.startswith('data:') or data.startswith('event:')) and '\n\n' in data:
            return data

        # Handle bytes (binary data)
        if isinstance(data, bytes):
            try:
                # Try to decode as UTF-8 first
                data = data.decode('utf-8')
            except UnicodeDecodeError:
                # Binary data, encode as base64
                b64_data = base64.b64encode(data).decode('utf-8')
                return f"event: binary\ndata: {b64_data}\n\n"

        # Convert objects to JSON
        if not isinstance(data, str):
            try:
                data = json.dumps(data)
            except Exception:
                data = str(data)

        # Handle JSON data with special event formatting
        if data.strip().startswith('{'):
            try:
                json_data = json.loads(data)
                if isinstance(json_data, dict) and 'event' in json_data:
                    event_type = json_data['event']
                    event_id = json_data.get('id', '')

                    sse = f"event: {event_type}\n"
                    if event_id:
                        sse += f"id: {event_id}\n"
                    sse += f"data: {data}\n\n"
                    return sse
                else:
                    # Regular JSON without event
                    return f"data: {data}\n\n"
            except json.JSONDecodeError:
                # Not valid JSON, treat as text
                return f"data: {data}\n\n"
        else:
            # Plain text
            return f"data: {data}\n\n"

    @classmethod
    async def wrap_sync_generator(cls, generator):
        """Convert a synchronous generator to an async generator."""
        for item in generator:
            yield item
            # Allow other tasks to run
            await asyncio.sleep(0)

    @classmethod
    async def create_sse_stream(
        cls,
        source,
        cleanup_func: Callable[[], None] | Callable[[], T] | Callable[[], AsyncGenerator[T, None]] | None = None
    ) -> AsyncGenerator[str, None]:
        """
        Convert any source to a properly formatted SSE stream.

        Args:
            source: Can be async generator, sync generator, or iterable
            cleanup_func: Optional function to call when the stream ends or is cancelled.
                          Can be a synchronous function, async function, or async generator.

        Yields:
            Properly formatted SSE messages
        """
        # Send stream start event
        yield cls.format_sse_event({"event": "stream_start", "id": "0"})

        try:
            # Handle different types of sources
            if inspect.isasyncgen(source):
                # Source is already an async generator
                async for item in source:
                    yield cls.format_sse_event(item)
            elif inspect.isgenerator(source) or hasattr(source, '__iter__'):
                # Source is a sync generator or iterable
                async for item in cls.wrap_sync_generator(source):
                    yield cls.format_sse_event(item)
            else:
                # Single item
                yield cls.format_sse_event(source)
        except asyncio.CancelledError:
            # Client disconnected
            yield cls.format_sse_event({"event": "cancelled", "id": "cancelled"})
            raise
        except Exception as e:
            # Error in stream
            error_info = {
                "event": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            yield cls.format_sse_event(error_info)
        finally:
            # Always send end event
            yield cls.format_sse_event({"event": "stream_end", "id": "final"})

            # Execute cleanup function if provided
            if cleanup_func:
                try:
                    if asyncio.iscoroutinefunction(cleanup_func):
                        # Async function
                        await cleanup_func()
                    elif inspect.isasyncgen(cleanup_func):
                        # Async generator
                        async for _ in cleanup_func():
                            pass  # Exhaust the generator to ensure cleanup completes
                    else:
                        # Synchronous function
                        cleanup_func()
                except Exception as e:
                    # Log cleanup errors but don't propagate them to client
                    error_info = {
                        "event": "cleanup_error",
                        "message": str(e),
                        "traceback": traceback.format_exc()
                    }
                    # We can't yield here as the stream is already closing
                    # Instead, log the error
                    print(f"SSE cleanup error: {error_info}", flush=True)

