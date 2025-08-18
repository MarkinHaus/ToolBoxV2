# ToolBoxV2 Module Creation Guide

This guide provides a comprehensive overview of how to create, structure, and integrate new modules within the ToolBoxV2 framework. It is intended for both human developers and AI agents.

## 1. Core Concepts

A ToolBoxV2 module is a self-contained unit of functionality that plugs into the main application. It can expose functions to the system, provide API endpoints, and render user interfaces.

### Key Components of a Module:

*   **Module File**: A Python file (e.g., `MyModule.py`) located in the `toolboxv2/mods/` directory.
*   **`get_app` and `export`**: The entry point for connecting your module to the ToolBoxV2 application instance.
*   **`@export` Decorator**: The primary mechanism for registering functions and defining their behavior (e.g., as API endpoints, lifecycle hooks, etc.).
*   **`Tools` Class (Optional but Recommended)**: A class that inherits from `MainTool` to organize your module's logic, state, and lifecycle methods (`on_start`, `on_exit`).
*   **`RequestData` and `Result` Objects**: Standardized Pydantic models for handling incoming requests and formatting outgoing responses, ensuring consistency across the framework.

## 2. Module Structure: A Boilerplate

Here is a standard boilerplate for a new module file (e.g., `toolboxv2/mods/MyNewModule.py`):

```python
# toolboxv2/mods/MyNewModule.py

from toolboxv2 import App, Result, RequestData, get_app, MainTool
from typing import Dict, Optional

# -- Constants ---
MOD_NAME = "MyNewModule"
VERSION = "1.0.0"

# -- Module Export ---
# This makes the @export decorator available for this module.
export = get_app(f"mods.{MOD_NAME}").tb

# -- Main Logic Class (Recommended) ---
class Tools(MainTool):
    def __init__(self, app: App):
        self.app = app
        self.name = MOD_NAME
        self.version = VERSION
        # You can define CLI tools here if needed
        self.tools = {
            "all": [["show_version", "Displays the module version"]],
            "name": self.name,
            "show_version": self.show_version,
        }
        super().__init__(
            load=self.on_start, # Corresponds to @export(initial=True)
            v=self.version,
            tool=self.tools,
            name=self.name,
            on_exit=self.on_exit # Corresponds to @export(exit_f=True)
        )

    def on_start(self):
        """Called when the module is loaded."""
        self.app.logger.info(f"{self.name} v{self.version} initialized.")
        # Example: Registering a UI component with another module
        self.app.run_any(("CloudM", "add_ui"),
                         name=self.name,
                         title=self.name,
                         path=f"/api/{self.name}/ui",
                         description="A description of my module's UI.",
                         auth=True
                         )

    def on_exit(self):
        """Called when the application is shutting down."""
        self.app.logger.info(f"Closing {self.name}. Goodbye!")

    def show_version(self):
        return self.version

# -- API Endpoints & Functions ---

@export(mod_name=MOD_NAME, name="ui", api=True, api_methods=["GET"])
async def get_main_ui(self) -> Result:
    """Serves the main HTML UI for the module."""
    # The 'self' here will be the instance of the Tools class
    html_content = "<h1>Welcome to MyNewModule!</h1>"
    return Result.html(data=html_content)

@export(mod_name=MOD_NAME, name="get_data", api=True, request_as_kwarg=True)
async def get_some_data(self, request: RequestData) -> Result:
    """An example API endpoint to fetch data."""
    user = await self.app.run_any(("WidgetsProvider", "get_user_from_request"), request=request)
    user_name = user.name if user else "Guest"
    return Result.json(data={"message": f"Hello, {user_name}!", "module": self.name})

```

## 3. The `@export` Decorator

The `@export` decorator is the most critical part of creating a module. It tells ToolBoxV2 how to treat your function. Here are the key parameters:

*   `mod_name` (str): **Required**. The name of your module. Must match `MOD_NAME`.
*   `name` (str): The name to expose the function under. If omitted, the Python function name is used.
*   `api` (bool): If `True`, the function becomes an HTTP API endpoint accessible at `/api/MOD_NAME/function_name`.
*   `api_methods` (List[str]): A list of allowed HTTP methods (e.g., `["GET", "POST"]`).
*   `request_as_kwarg` (bool): If `True`, the `RequestData` object will be passed as a keyword argument to your function.
*   `initial` (bool): If `True`, the function is an initialization hook and runs when the module is first loaded.
*   `exit_f` (bool): If `True`, the function is a cleanup hook and runs when the application exits.
*   `row` (bool): If `True`, the function's raw return value is sent as the response body, bypassing the `Result` object wrapper. Useful for serving files or raw text.
*   `level` (int): An integer indicating the privilege level required to execute the function.

## 4. Handling Requests and Responses

### The `RequestData` Object

When a function is an API endpoint (`api=True`) and uses `request_as_kwarg=True`, it receives a `RequestData` object. This object contains all the information about the incoming HTTP request:

*   `request.method`: The HTTP method (e.g., 'GET', 'POST').
*   `request.path`: The request path.
*   `request.headers`: A Pydantic model of the request headers.
*   `request.query_params`: A dictionary of URL query parameters.
*   `request.form_data`: A dictionary of data from a submitted form.
*   `request.session`: A Pydantic model containing user session information, if the user is authenticated.

### The `Result` Object

API functions should almost always return a `Result` object. This standardizes responses and error handling.

**Success Responses:**

*   `Result.ok(data, info)`: A generic success response.
*   `Result.json(data, info)`: For JSON API responses. Sets the `Content-Type` header to `application/json`.
*   `Result.html(data, info)`: For serving HTML content.
*   `Result.file(data, filename)`: For sending files to the user for download.
*   `Result.sse(stream_generator)`: For Server-Sent Events (event streaming).

**Error Responses:**

*   `Result.default_user_error(info, exec_code)`: For client-side errors (e.g., bad input). Typically returns a 4xx status code.
*   `Result.default_internal_error(info, exec_code)`: For server-side errors. Typically returns a 5xx status code.

## 5. Frontend Integration with `tbjs`

Modules often have a corresponding frontend component. The `tbjs` framework is designed to interact seamlessly with ToolBoxV2 modules.

### Making API Calls from the Frontend

Use the `TB.api.request` function in your JavaScript to call your module's backend functions.

```javascript
// In your frontend JavaScript file

async function fetchDataFromMyModule() {
    try {
        // Calls the 'get_data' function in the 'MyNewModule' module
        const response = await TB.api.request('MyNewModule', 'get_data');

        if (response.error === "none") {
            const data = response.get(); // Helper to get response.result.data
            console.log("Data from backend:", data.message);
            document.getElementById('my-element').innerText = data.message;
        } else {
            TB.ui.Toast.showError(`Error: ${response.info.help_text}`);
        }
    } catch (error) {
        TB.logger.error("Network or API request failed", error);
        TB.ui.Toast.showError("Failed to connect to the server.");
    }
}
```

### Serving a UI

Your module can serve its entire UI as an HTML string from an API endpoint. This is a common pattern for "widgets" or self-contained applications.

1.  **Backend (`MyNewModule.py`):** Create an endpoint that returns HTML.

    ```python
    @export(mod_name=MOD_NAME, name="ui", api=True)
    async def get_main_ui(self) -> Result:
        html_content = """
            <div>
                <h1>My Module's UI</h1>
                <button id="my-button">Fetch Data</button>
                <p id="my-element"></p>
                <script unSave="true">
                    document.getElementById('my-button').addEventListener('click', async () => {
                        const response = await TB.api.request('MyNewModule', 'get_data');
                        if (response.get()) {
                            document.getElementById('my-element').innerText = response.get().message;
                        }
                    });
                </script>
            </div>
        """
        return Result.html(data=html_content)
    ```

2.  **Integration (`on_start`)**: In your module's `on_start` method, register this UI with the `CloudM` module so it appears in the main application menu.

    ```python
    def on_start(self):
        self.app.run_any(("CloudM", "add_ui"),
                         name=self.name,
                         title="My Awesome Module",
                         path=f"/api/{self.name}/ui",
                         description="This is a module I built.",
                         auth=True # Requires user to be logged in
                         )
    ```

## 6. Inter-Module Communication

Modules can call functions in other modules using `app.run_any()`.

```python
# In MyNewModule.py

@export(mod_name=MOD_NAME, name="process_and_store", api=True, request_as_kwarg=True)
async def process_and_store(self, request: RequestData) -> Result:
    # 1. Get the user from the request using the WidgetsProvider module
    user = await self.app.run_any(("WidgetsProvider", "get_user_from_request"), request=request)
    if not user:
        return Result.default_user_error("Authentication required.")

    # 2. Use the ISAA module to analyze some text
    analysis = await self.app.run_any(("isaa", "mini_task_completion"),
                                     mini_task="Summarize this text.",
                                     user_task="ToolBoxV2 is a modular framework...")

    # 3. Save the result to the user's file storage using FileWidget
    storage = await self.app.run_any(("FileWidget", "get_blob_storage"), request=request)
    # ... (code to save 'analysis' to blob storage) ...

    return Result.ok(info="Data processed and saved.")
```

This pattern allows for powerful, decoupled architectures where modules specialize in one area (AI, files, UI) and collaborate to build complex applications.
