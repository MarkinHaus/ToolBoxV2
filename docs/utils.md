# ToolBoxV2: The `App` Class

The `App` class is the central singleton instance in ToolBoxV2, responsible for managing the application's lifecycle, configuration, module loading, and core functionalities. It's typically accessed via the `get_app()` utility function.

## Initialization

The `App` instance is initialized with a `prefix` and `AppArgs` (command-line arguments).

```python
from toolboxv2 import App, AppArgs, get_app

# Example: Initialize or get the App instance
# The prefix helps differentiate multiple App instances if needed,
# and is often used in directory naming.
args = AppArgs().default() # Or parsed from sys.argv in __main__.py
app_instance = get_app(prefix="my_app_instance", args=args)

# Accessing key attributes:
print(f"App ID: {app_instance.id}")
print(f"Version: {app_instance.version}")
print(f"Start Directory: {app_instance.start_dir}")
print(f"Data Directory: {app_instance.data_dir}")
print(f"Config Directory: {app_instance.config_dir}")
print(f"Debug Mode: {app_instance.debug}")
```

### Key Initialization Steps:

1.  **System & Paths:**
    *   Determines the operating system (`system_flag`).
    *   Sets the `start_dir` to the application's root directory.
    *   Resolves the `prefix`:
        *   If no prefix is provided, it attempts to load the last used prefix from `.data/last-app-prefix.txt`.
        *   If a prefix is provided, it's saved to this file for future use.
    *   Constructs the `app_id` (e.g., `prefix-hostname`).
    *   Sets up `data_dir`, `config_dir`, and `info_dir` based on the `app_id` (e.g., `./.data/prefix-hostname/`).
    *   Sets up `appdata` directory (OS-specific application data folder).

2.  **Logging:**
    *   Initializes a logger (`app.logger`). The logging level and output (terminal/file) can vary based on the `prefix` (e.g., "test", "live", "debug") and the `--debug` CLI argument.

3.  **Configuration:**
    *   Loads application configuration using `FileHandler` from a file typically named `app_id.config` in the `config_dir`.
    *   Defines default configuration `keys` and `defaults` (e.g., for macros, helpers, debug status).

4.  **Core Attributes:**
    *   `version`: ToolBoxV2 version, read from `pyproject.toml`.
    *   `debug`: Boolean, controlled by CLI args and config.
    *   `dev_modi`: Boolean, development mode status from config.
    *   `functions`: A dictionary to store registered functions from modules.
    *   `modules`: A dictionary to store loaded module objects.
    *   `interface_type`: Default `ToolBoxInterfaces.native`.
    *   `alive`: Boolean, controls the main application loop.
    *   `args_sto`: Stores the parsed `AppArgs`.
    *   `loop`: The asyncio event loop (initialized later or if already running).
    *   `session`: A `Session` object for managing user/remote session state.

5.  **Conditional Actions (based on `AppArgs`):**
    *   `args.init`: If true, adds `start_dir` to `sys.path`.
    *   The `__main__.py` script handles other arguments like `--update`, `--get-version`, etc., by calling `App` methods or other utilities.

## Core Functionalities

### Module Management

*   **`load_mod(mod_name: str, spec='app', mlm='I', **kwargs)` / `save_load(modname, spec='app')`:**
    *   Loads a module into the application.
    *   `spec` (specification): Used to namespace or categorize the module instance (e.g., 'app' for general, or a specific session ID).
    *   Supports different loading mechanisms (`mlm`):
        *   `'I'`: In-place load (imports the Python module directly). This is the default.
        *   `'C'`: Copies the module file to a runtime directory before loading (less common).
    *   Handles `ModuleNotFoundError` by attempting to guide the user (e.g., install via `CloudM` or `pip`).
    *   Registers the module's exported functions and/or its `Tools` class instance.
    *   Can reload modules if they are already loaded.
    ```python
    # Load the 'MyModule'
    my_module_instance = app_instance.load_mod("MyModule")

    # Or if it's a Tool-based module:
    # my_tool_instance = app_instance.load_mod("MyToolModule")
    ```

*   **`get_mod(name: str, spec='app') -> ModuleType | MainToolType`:**
    *   Retrieves a loaded module instance. If the module isn't loaded, it attempts to load it.
    ```python
    db_mod = app_instance.get_mod("DB")
    if db_mod:
        db_mod.some_db_function()
    ```

*   **`remove_mod(mod_name: str, spec='app', delete=True)` / `a_remove_mod(...)` (async):**
    *   Unloads a module, calling its `on_exit` functions if defined.
    *   `delete=True` removes it completely from the `functions` registry.

*   **`reload_mod(mod_name: str, spec='app', ...)`:**
    *   Reloads an existing module. Useful for development.

*   **`watch_mod(mod_name: str, ...)`:**
    *   Monitors a module's source file(s) for changes and automatically reloads it.
    ```python
    # In development, watch 'MyDevModule' for changes
    app_instance.watch_mod("MyDevModule")
    ```

*   **`load_all_mods_in_file(working_dir="mods")` / `a_load_all_mods_in_file(...)` (async):**
    *   Scans the specified directory (default `./mods/`) and loads all valid Python modules found.
# To [isaa](./isaa.md)
### Function Registration and Execution

*   **`@app.tb(...)` Decorator (via `_create_decorator`):**
    *   The primary way functions are registered with ToolBoxV2. See `example_mod.md` for details on usage.
    *   This decorator populates the `app.functions` dictionary.

*   **`get_function(name: Enum | tuple, metadata=False, state=True, specification='app', ...)`:**
    *   Retrieves a registered function.
    *   `name`: Can be an Enum (from `all_functions_enums.py`) or a `(module_name, function_name)` tuple.
    *   `metadata=True`: Returns a tuple `(function_data_dict, callable_function)`.
    *   `state=True`: Returns a stateful version of the function (bound to its module instance if applicable).
    *   `state=False`: Returns the raw, stateless function.
    *   `specification`: The context/instance spec to get the function for.

*   **`run_any(mod_function_name, ..., get_results=False, **kwargs)` / `a_run_any(...)` (async):**
    *   Executes a registered function by its name (Enum or tuple).
    *   Handles argument passing, stateful/stateless execution, and error wrapping into a `Result` object.
    *   `get_results=True`: Returns the `Result` object itself.
    *   `get_results=False` (default): Returns the `data` payload from the `Result` object if successful.
    *   Automatically handles running the function's pre/post compute hooks and caching if configured via `@app.tb`.
    ```python
    # Synchronous execution
    result_data = app_instance.run_any(("MyModule", "my_function"), arg1="hello")
    full_result_obj = app_instance.run_any(("MyModule", "my_function"), arg1="hello", get_results=True)

    # Asynchronous execution
    async_result_data = await app_instance.a_run_any(("MyAsyncModule", "my_async_function"))
    ```

*   **`run_http(mod_function_name, function_name=None, method="GET", ...)` (async):**
    *   Executes a function on a remote ToolBoxV2 instance via HTTP, using the app's `session` object.

### Application Lifecycle

*   **`exit()` / `a_exit()` (async):**
    *   Gracefully shuts down the application.
    *   Calls `on_exit` functions for all loaded modules.
    *   Saves configuration.
    *   Stops the main application loop (`alive = False`).
    *   Cleans up threads and the event loop if applicable.

### Utilities

*   **`print(text, *args, **kwargs)` / `sprint(text, *args, **kwargs)`:**
    *   Styled print functions, prepending `System$[app_id]:`. `sprint` is often used for more verbose/system-level messages and can be silenced.
*   **`debug_rains(e: Exception)`:** If `app.debug` is true, prints a full traceback and re-raises the exception.
*   **`set_flows(r: dict)` / `run_flows(name: str, **kwargs)`:** Manages and executes predefined application flows (sequences of operations).
*   **`get_username()` / `set_username(username: str)`:** Manages the application's user identity.
*   **`save_autocompletion_dict()` / `get_autocompletion_dict()`:** Saves/loads a dictionary of modules and their functions for autocompletion features.
*   **`save_registry_as_enums(directory: str, filename: str)`:** Generates an `all_functions_enums.py`-like file from the currently registered functions.
*   **`execute_all_functions(...)` / `a_execute_all_functions(...)` (async):**
    *   Runs all registered testable functions (marked with `test=True` in `@app.tb` or having `samples`).
    *   Useful for integration testing and profiling.
    *   Can filter by module (`m_query`) and function (`f_query`).
    *   Supports profiling via `cProfile`.
*   **`run_bg_task(task: Callable)`:**
    *   Runs a synchronous or asynchronous task in a separate background thread with its own event loop. Ensures proper handling of nested asyncio operations within the task.

## Session Management (`app.session`)

The `app.session` attribute holds an instance of the `Session` class (from `toolboxv2.utils.system.session`). It's used for:
*   Authenticating with a remote ToolBoxV2 service (e.g., SimpleCore Hub).
*   Making authenticated HTTP requests (`session.fetch`, `session.upload_file`, `session.download_file`).
*   Manages JWT claims and private key authentication.

```python
# Example: Making an authenticated API call
# Assumes app.session is already authenticated
response_data = await app_instance.session.fetch("/api/MyRemoteModule/get_info")
json_payload = await response_data.json()
```

---
```

### 2. `cli.md` - Documenting the Command Line Interface

This should explain how to use the `tb` (or `python -m toolboxv2`) command-line tool, detailing its arguments and their effects.

```markdown
# ToolBoxV2: Command Line Interface (CLI)

ToolBoxV2 provides a command-line interface (CLI) for managing and running applications. It's typically invoked as `tb` (if installed globally or via an alias) or `python -m toolboxv2`.

## General Usage

```bash
python -m toolboxv2 [options] [sub-commands]
# or
tb [options] [sub-commands]
```

The CLI script (`__main__.py`) performs the following main steps:
1.  Parses command-line arguments.
2.  Initializes the `App` instance via `setup_app()` (which calls `get_app()`).
3.  Handles various options to:
    *   Manage application data and configuration.
    *   Control application modes (background, proxy, debug).
    *   Load modules and manage their state.
    *   Run tests or profilers.
    *   Execute specific application flows or commands.

## Key CLI Arguments

The following are some of the primary arguments available. Use `tb -h` or `python -m toolboxv2 -h` for a full list.

*   **Instance and Mode:**
    *   `-init [name]`: Initializes ToolBoxV2 with a specific instance name (default: `main`).
    *   `-n, --name <name>`: Specifies an ID for the ToolBox instance (default: `main`). This affects data/config directory names.
    *   `-m, --modi <mode>`: Starts a ToolBoxV2 interface/flow (e.g., `cli`, `bg`, or custom flows). Default is usually "cli".
    *   `--debug`: Starts the application in debug mode (more verbose logging, potentially different behavior).
    *   `--remote`: Starts in remote mode, often for connecting to a proxy or another instance.
    *   `-bg, --background-application`: Starts an interface in the background as a detached process.
    *   `-bgr, --background-application-runner`: Runs the background application logic in the current terminal (for daemons).
    *   `-fg, --live-application`: Starts a proxy interface, connecting to a background daemon.
    *   `--kill`: Kills the currently running local ToolBoxV2 instance (matching the `-m <mode>` and `-n <name>`).

*   **Module and Version Management:**
    *   `-l, --load-all-mod-in-files`: Loads all modules found in the `mods/` directory on startup.
    *   `-sfe, --save-function-enums-in-file`: Generates/updates the `all_functions_enums.py` file based on loaded modules. Often used with `-l`.
    *   `-v, --get-version`: Prints the version of ToolBoxV2 and all loaded modules.
    *   `-i, --install <name>`: Installs a module or interface (likely via `CloudM` module).
    *   `-r, --remove <name>`: Uninstalls a module or interface.
    *   `-u, --update <name_or_main>`: Updates a module/interface or the core ToolBoxV2 (`main`).

*   **Development and Testing:**
    *   `--test`: Runs all unit tests (typically discovers and runs tests in the `tests/` directory).
    *   `--profiler`: Runs all registered testable functions and profiles their execution using `cProfile`.
    *   `--ipy`: Starts an IPython session with the ToolBoxV2 app pre-loaded. Provides magic commands like `%tb save/loadX/load/open/inject`.

*   **Service Management (`--sm`):**
    *   Provides a sub-menu for managing ToolBoxV2 as a system service (Linux/systemd or Windows Startup).
    *   Options: Init, Start/Stop/Restart, Status, Uninstall, Show/Hide console window (Windows).

*   **Log Management (`--lm`):**
    *   Provides a sub-menu for managing log files (e.g., removing or unstyling logs by date/level).

*   **Data and Configuration Management:**
    *   `--delete-config-all`: Deletes *all* configuration files. **Use with caution!**
    *   `--delete-data-all`: Deletes *all* data folders. **Use with caution!**
    *   `--delete-config`: Deletes the configuration file for the *named* instance.
    *   `--delete-data`: Deletes the data folder for the *named* instance.

*   **Network Configuration (for interfaces):**
    *   `-p, --port <port>`: Specifies the port for an interface (default: `5000` or `6587` for background).
    *   `-w, --host <host>`: Specifies the host for an interface (default: `0.0.0.0`).

*   **Direct Command Execution:**
    *   `-c, --command <module_name> <function_name> [arg1 arg2 ...]` (can be repeated): Executes a specific function.
    *   `--kwargs <key=value>` (can be repeated): Provides keyword arguments for commands specified with `-c`.

*   **Conda Integration:**
    *   `conda [conda_args...]`: Special argument to pass commands directly to a `conda_runner.py` script (e.g., `tb conda env list`).

*   **API Runner:**
    *   `api [api_args...]`: Special argument to invoke `cli_api_runner.py`, likely for direct API interactions or testing.

*   **GUI:**
    *   `gui`: Starts the GUI version of ToolBoxV2 (imports and runs `toolboxv2.__gui__.start`).

## CLI Execution Flow (`__main__.py`)

1.  **Argument Parsing:** `parse_args()` uses `argparse` to define and parse all CLI arguments.
2.  **App Setup (`setup_app()`):**
    *   Initializes the `App` instance using `get_app()` with the parsed arguments and name.
    *   Sets up PID file for the current process.
    *   Optionally silences `app.sprint` if not in debug/verbose mode.
    *   Loads all modules if `-l` is specified.
    *   Handles `--update` logic.
3.  **Background/Live Application Handling:**
    *   If `-bgr`: Initializes `DaemonApp`.
    *   If `-bg`: Starts the application as a detached background process using `subprocess.Popen`.
    *   If `-fg` (live-application): Attempts to connect to a background daemon using `ProxyApp`.
4.  **Action Dispatching:** Based on the parsed arguments, it performs actions like:
    *   Module installation (`--install`).
    *   Log management (`--lm`).
    *   Service management (`--sm`).
    *   Saving function enums (`-sfe`).
    *   Printing versions (`-v`).
    *   Running the profiler (`--profiler`).
    *   Running flows based on `--modi`.
    *   Handling Docker commands (`--docker`).
    *   Killing an existing instance (`--kill`).
    *   Executing direct commands (`-c`).
5.  **Cleanup:** Removes the PID file and calls `app.a_exit()` before exiting.

## Example CLI Usage

```bash
# Get version information
python -m toolboxv2 -v

# Load all modules and save function enums
python -m toolboxv2 -l -sfe

# Run a specific function in MyModule
python -m toolboxv2 -c MyModule my_function arg_value --kwargs param_name=kwarg_value

# Start the application with a custom flow named 'web_server' in debug mode
python -m toolboxv2 -m web_server --debug -n my_web_instance

# Start a background daemon for the 'bg_processing' flow
python -m toolboxv2 -m bg_processing -bg -n background_processor

# Connect to the background daemon with a live proxy application
python -m toolboxv2 -m cli -fg -n background_processor

# Kill the 'web_server' modi instance named 'my_web_instance'
python -m toolboxv2 -m web_server --kill -n my_web_instance
```

---
```

### 3. `example_mod.md` - Documenting Module Creation

This needs to be updated to accurately reflect the `@app.tb(...)` decorator from `toolbox.py` and the `Result` and `RequestData` classes from `types.py`.

```markdown
# ToolBoxV2: Creating Modules

ToolBoxV2 modules are Python files or packages that extend the framework's functionality. They can define simple functions, stateful tools (classes inheriting from `MainTool`), or API endpoints.

## Basic Module Structure

A typical ToolBoxV2 module (`.py` file) includes:

1.  **Imports:** Necessary libraries and ToolBoxV2 components.
2.  **Module Metadata (Optional but Recommended):**
    *   `Name` (or `name`): A string defining the module's canonical name.
    *   `version`: A string for the module's version (e.g., "1.0.0").
3.  **Function/Class Definitions:** The core logic of your module.
4.  **Exporting Functions:** Functions are made available to ToolBoxV2 using the `@export` decorator (which is an alias for `app.tb`).

## The `@export` Decorator (`app.tb`)

The `@export` decorator is the primary mechanism for registering functions and configuring their behavior within ToolBoxV2. It's obtained from an `App` instance.

```python
from toolboxv2 import get_app, App, Result, RequestData, MainTool
from toolboxv2.utils.system.types import ToolBoxInterfaces # For specific interface types
from typing import Optional, Dict, Any, List
import asyncio

# Get the application instance (singleton)
# The 'prefix' for get_app here is often the module's own name,
# though the decorator will use its 'mod_name' parameter.
app = get_app("MyModule")
export = app.tb # Alias the decorator for convenience

# --- Module Metadata (Best Practice) ---
Name = "MyModule"  # Used by the decorator if mod_name is not specified
version = "1.0.1"

# --- Example Functions ---

@export(mod_name=Name, version=version, helper="A simple greeting function.")
def greet(name: str) -> str:
    """Returns a greeting message."""
    return f"Hello, {name} from MyModule!"

@export(mod_name=Name, version=version, row=True, helper="Returns raw data without Result wrapping.")
def get_raw_data() -> dict:
    """Demonstrates returning raw data."""
    return {"key": "value", "number": 123}

@export(mod_name=Name, version=version, initial=True, helper="Runs when the module is first loaded.")
def on_module_load():
    """Initialization logic for this module."""
    app.print(f"{Name} module has been loaded and initialized!")
    # return Result.ok(info="MyModule initialized successfully") # Optional: return a Result

@export(mod_name=Name, version=version, exit_f=True, helper="Runs when the application is shutting down.")
async def on_module_exit():
    """Cleanup logic for this module."""
    await asyncio.sleep(0.1) # Simulate async cleanup
    app.print(f"{Name} module is cleaning up.")
    # return Result.ok(info="MyModule cleaned up.") # Optional

@export(mod_name=Name, version=version, api=True, api_methods=['GET'], request_as_kwarg=True,
        helper="An example API endpoint.")
async def my_api_endpoint(request: Optional[RequestData] = None) -> Result:
    """
    Handles GET requests to /api/MyModule/my_api_endpoint.
    Accesses request details if provided.
    """
    if request:
        app.print(f"API request received: {request.request.method} {request.request.path}")
        app.print(f"Query Params: {request.request.query_params}")
        app.print(f"User from session: {request.session.user_name}")
    return Result.json(data={"message": "API call successful!", "module_version": version})

@export(mod_name=Name, version=version, memory_cache=True, memory_cache_ttl=60)
def expensive_calculation(param: int) -> int:
    """
    An example of a function whose results will be cached in memory for 60 seconds.
    """
    app.print(f"Performing expensive calculation for {param}...")
    time.sleep(2) # Simulate work
    return param * param

# Example of a more complex function using App instance and returning a Result
@export(mod_name=Name, version=version)
def process_data_with_app(app_instance: App, data_id: int) -> Result:
    """
    This function automatically receives the 'App' instance if its first parameter is type-hinted as 'App'.
    This is determined by the 'state=True' logic in the decorator if 'app' is the first param.
    Alternatively, use state=False for stateless functions.
    """
    if not isinstance(app_instance, App): # Should always be App if first param is 'app'
        return Result.default_internal_error("App instance not correctly passed.")

    # Use app_instance for logging, accessing config, other modules, etc.
    app_instance.logger.info(f"Processing data_id: {data_id} in {Name}")
    if data_id < 0:
        return Result.default_user_error(info="Data ID cannot be negative.")
    return Result.ok(data={"processed_id": data_id, "status": "completed"})

```

### `@export` Decorator Parameters:

*   `name` (str, optional): The name to register the function under. Defaults to the function's actual name.
*   `mod_name` (str): The name of the module this function belongs to. If not provided, it tries to infer from `func.__module__` or a global `Name` in the module.
*   `version` (str, optional): Version string for this function/feature. Combined with the app's version.
*   `helper` (str, optional): A docstring or help text for the function.
*   `api` (bool, default `False`): If `True`, exposes this function as an HTTP API endpoint.
    *   The URL pattern is typically `/api/<mod_name>/<func_name>`.
    *   For streaming, `/sse/<mod_name>/<func_name>`.
*   `api_methods` (List[str], optional): Specifies allowed HTTP methods (e.g., `['GET', 'POST']`). Defaults to `['AUTO']` (GET if no body params, POST if body params).
*   `request_as_kwarg` (bool, default `False`): If `True` and `api=True`, the function will receive a `request: RequestData` keyword argument if it's defined in its signature.
*   `row` (bool, default `False`): If `True`, the function's raw return value is used directly. If `False` (default), the return value is automatically wrapped in a `Result.ok()` object if it's not already a `Result` or `ApiResult`.
*   `initial` (bool, default `False`): If `True`, this function is added to the module's "on_start" list and is called when the module is loaded by the `App` instance (if the module instance is a `MainTool` or similar, or if called directly).
*   `exit_f` (bool, default `False`): If `True`, this function is added to the module's "on_exit" list and is called when the `App` instance is shutting down or the module is removed.
*   `state` (bool, optional):
    *   If `None` (default): Automatically determined. If the first parameter of the decorated function is named `self` or `app` (and type-hinted as `App`), `state` is considered `True`. Otherwise `False`.
    *   If `True`: The function is considered stateful. If its first parameter is `self`, it's assumed to be a method of a class instance (e.g., a `MainTool` subclass). If `app`, the `App` instance is passed.
    *   If `False`: The function is treated as stateless.
*   `test` (bool, default `True`): Marks the function as testable. Used by `app.execute_all_functions()`.
*   `samples` (List[Dict[str, Any]], optional): A list of sample keyword arguments to be used when testing the function with `app.execute_all_functions()`.
*   `memory_cache` (bool, default `False`): Enables in-memory caching for the function's results.
*   `memory_cache_ttl` (int, default `300`): Time-to-live in seconds for memory cache entries.
*   `memory_cache_max_size` (int, default `100`): Max number of entries in the memory cache.
*   `file_cache` (bool, default `False`): Enables file-based caching for the function's results. (Stored in `app.data_dir/cache/...`).
*   `restrict_in_virtual_mode` (bool, default `False`): If `True`, restricts function in certain virtualized/proxied modes.
*   `level` (int, default `-1`): A general-purpose level or priority for the function.
*   `pre_compute` (Callable, optional): A function `(func, *args, **kwargs) -> (args, kwargs)` called before the main function executes. It can modify args/kwargs.
*   `post_compute` (Callable, optional): A function `(result, func, *args, **kwargs) -> result` called after the main function executes. It can modify the result.
*   `interface` (ToolBoxInterfaces | str, optional): Specifies the intended interface type (e.g., `ToolBoxInterfaces.cli`, `ToolBoxInterfaces.api`). Defaults to "tb".

### `Result` and `ApiResult` Objects

*   Modules should typically return `Result` objects (or `ApiResult` for API endpoints) to provide standardized responses including success/error status, data payload, and informational messages.
*   The `toolboxv2.utils.system.types.Result` class provides factory methods:
    *   `Result.ok(data=..., info=...)`
    *   `Result.json(data=..., info=...)` (for `api=True` functions)
    *   `Result.html(data=..., info=...)`
    *   `Result.text(text_data=..., info=...)`
    *   `Result.binary(data=..., content_type=..., download_name=...)`
    *   `Result.redirect(url=..., status_code=...)`
    *   `Result.stream(stream_generator=..., info=..., cleanup_func=...)` (for SSE)
    *   `Result.default_user_error(info=..., exec_code=...)`
    *   `Result.default_internal_error(info=..., exec_code=...)`
    *   `Result.custom_error(data=..., info=..., exec_code=...)`
*   The `Result` object has a `task(background_task_callable)` method to schedule a background task to run after the main function returns.

### `RequestData` Object

*   For API functions (`api=True`) with `request_as_kwarg=True`, if the function signature includes `request: Optional[RequestData] = None`, it will receive an instance of `toolboxv2.utils.system.types.RequestData`.
*   `RequestData` provides access to:
    *   `request.method`, `request.path`
    *   `request.headers` (an instance of `Headers`, e.g., `request.headers.user_agent`, `request.headers.hx_request`)
    *   `request.query_params` (dict)
    *   `request.form_data` (dict, if applicable)
    *   `request.body` (parsed JSON if `content_type` is `application/json`, otherwise raw bytes/str)
    *   `session.SiID`, `session.user_name`, `session.level` (from the current user's session)

## Creating a `MainTool`-based Module

For more complex, stateful modules, you can create a class that inherits from `toolboxv2.utils.system.main_tool.MainTool`.

```python
from toolboxv2 import get_app, App, Result, MainTool
from toolboxv2.utils.system.types import ToolBoxError

app = get_app("MyToolModule")
export = app.tb

Name = "MyToolModule"
version = "0.5.0"

class Tools(MainTool): # The class must be named 'Tools' for auto-detection by older App versions
                      # or ensure your module file directly uses @export on methods if not named Tools.
    # Or, you can export methods directly from any class:
    # class MyCustomTool(MainTool):
    #    @export(...)
    #    def my_method(self, ...): ...

    async def __ainit__(self): # Asynchronous initialization
        # self.app is automatically available
        # self.name, self.version, self.logger are set by MainTool's __await__
        await super().__ainit__(name=Name, v=version, tool={
            "process_item": self.process_item, # For older compatibility if functions were in 'tools' dict
            "get_status": self.get_status
        })
        self.internal_state = "initialized"
        self.app.print(f"{self.name} (Tool) has been initialized with state: {self.internal_state}")

    @export(mod_name=Name, version=version) # Decorate methods to export them
    def process_item(self, item_id: int, details: str) -> Result:
        # 'self' provides access to app, logger, name, version, config
        self.app.logger.info(f"{self.name} processing item: {item_id} - {details}")
        self.internal_state = f"last_processed_{item_id}"
        if item_id == 0:
            return self.return_result( # Helper from MainTool
                error=ToolBoxError.input_error,
                exec_code=1, # Custom error code
                help_text="Item ID cannot be zero.",
                data_info="Validation failed"
            )
        return Result.ok(data={"item_id": item_id, "status": "processed by tool"})

    @export(mod_name=Name, version=version)
    async def get_status(self) -> str: # Example async method
        await asyncio.sleep(0.01)
        return f"Tool {self.name} current state: {self.internal_state}"

    async def on_exit(self): # Not automatically called unless also decorated or part of a convention
        self.app.print(f"Tool {self.name} is shutting down its internal state.")
        # Perform cleanup

# To ensure on_exit is called by the App framework:
@export(mod_name=Name, version=version, exit_f=True)
async def custom_tool_exit_function(app_instance: App):
    tool_instance = app_instance.get_mod(Name)
    if tool_instance and hasattr(tool_instance, 'on_exit') and callable(tool_instance.on_exit):
        await tool_instance.on_exit()

```

**Key aspects of `MainTool`:**
*   **Asynchronous Initialization:** Use `async def __ainit__(self)` for setup. The `MainTool` itself is awaitable, and `__ainit__` is called when the instance is first awaited (e.g., by `app.load_mod` or `app.get_mod`).
*   **`self.app`:** The `App` instance is available as `self.app`.
*   **`self.name`, `self.version`, `self.logger`:** These are automatically set up.
*   **`self.return_result(...)`:** A helper method for creating `Result` objects.
*   Methods intended to be called via `app.run_any` should be decorated with `@export`.

## Steps to Create a Valid Toolboxv2 Module:

1.  **Define Module Structure:** Organize your code with imports, metadata, and function/class definitions.
2.  **Clarify Dependencies:** Import necessary libraries. Handle missing optional dependencies gracefully if needed.
3.  **Export Functions/Methods:** Use the `@export(...)` decorator (e.g., `app.tb(...)`) to mark functions/methods that ToolBoxV2 should recognize.
    *   Provide `mod_name` and `version`.
    *   Use other parameters (`api`, `row`, `initial`, `exit_f`, `memory_cache`, etc.) as needed.
    *   Ensure clear signatures and document parameters/return types (Python type hints are highly recommended).
4.  **Documentation and Versioning:** Document your module and its functions. Use semantic versioning.
5.  **Testing:** Test your module thoroughly, including how it integrates with the ToolBoxV2 app (`app.run_any`, `app.get_mod`, etc.). Use the `test=True` and `samples` parameters in `@export` to facilitate automated testing via `app.execute_all_functions()`.

# To [isaa](./isaa.md)
