# ToolBoxV2 Developer Guide

Based on the provided documentation, here's a comprehensive guide on how to use the ToolBoxV2 framework for building applications.

## Introduction

ToolBoxV2 is a Python framework that provides a structured approach to building applications with standardized request handling and response formatting. It consists of two main components:

1. **RequestData Classes** - For handling HTTP requests with strong typing
2. **Result Class** - For standardized response handling and error management

## Setting Up Your Application

### Creating a Module

Start by initializing your application module:

```python
from toolboxv2 import get_app, App, RequestData, Result
from typing import Dict, Optional

# Define your module
MOD_NAME = "YOUR_MODULE_NAME"
version = "1.0"
export = get_app("{MODULE-NAME.SUB-MODULE}").tb
```

### Registering Functions

Use the `@export` decorator to register functions within your module:

```python
@export(mod_name=MOD_NAME, version=version)
def your_function():
    # Function logic here
    return Result.ok(data="Success")
```

## Function Types

### Standard System Functions

```python
# Basic function with App parameter
@export(mod_name=MOD_NAME, version=version, row=True)
def system_function(app: App):
    # Implementation
    return "Raw return value"  # Will be returned as-is because row=True

# Function without App parameter
@export(mod_name=MOD_NAME, version=version)
def function_without_app():
    # Implementation
    return Result.ok(data="Success")

# Function with arguments
@export(mod_name=MOD_NAME, version=version)
def function_with_args(name: str) -> Result:
    # Implementation
    return Result.ok(data=name)

# Function returning raw data
@export(mod_name=MOD_NAME, version=version, row=True)
def function_with_args_kwargs(name: str, nickname: Optional[str]=None) -> str:
    if nickname is None:
        nickname = ""
    return name + nickname  # Returned as raw string
```

### Async Functions

```python
@export(mod_name=MOD_NAME, version=version, row=True)
async def async_function(app: App):
    # Async implementation
    result = await some_async_operation()
    return result
```

### API Endpoints

```python
# API endpoint with request parameter
@export(mod_name=MOD_NAME, api=True, version="1.0", request_as_kwarg=True)
async def get_data(request: Optional[RequestData]=None):
    if request:
        query_params = request.query_params
        # Process query parameters
    return Result.json(data={"status": "success"})

# API endpoint with App and Request parameters
@export(mod_name=MOD_NAME, api=True, version="1.0", request_as_kwarg=True)
async def get_user_data(app, request: Optional[RequestData]=None):
    # Implementation using app and request
    return Result.ok(data={"user": "data"})

# API endpoint with specific HTTP methods
@export(mod_name=MOD_NAME, api=True, version="1.0", api_methods=['PUT', 'POST'])
async def update_data(app, data: Dict):
    # Process the JSON data received in the request body
    return Result.ok(data=data)

# API endpoint handling form data
@export(mod_name=MOD_NAME, api=True, version="1.0", api_methods=['PUT', 'POST'])
async def submit_form(app, form_data: Dict):
    # Process form data
    return Result.ok(data=form_data)
```

## Working with Request Data

### Accessing Request Information

```python
@export(mod_name=MOD_NAME, api=True, version="1.0", request_as_kwarg=True)
async def process_request(request: Optional[RequestData]=None):
    if request:
        # Access method and path
        method = request.method
        path = request.path

        # Access headers
        user_agent = request.headers.user_agent
        content_type = request.headers.content_type
        custom_header = request.headers.extra_headers.get('x-custom-header')

        # Access query parameters
        query_params = request.query_params
        search_term = query_params.get('search')

        # Access form data or JSON body
        if request.form_data:
            form_values = request.form_data

        if request.body and request.content_type == 'application/json':
            json_data = request.body

    return Result.ok(data="Request processed")
```

### Accessing Session Information

```python
@export(mod_name=MOD_NAME, api=True, version="1.0", request_as_kwarg=True)
async def get_user_session(request: Optional[RequestData]=None):
    if request and hasattr(request, 'session'):
        # Access session data
        session_id = request.session.SiID
        user_name = request.session.user_name
        session_level = request.session.level

        # Access custom session data
        custom_data = request.session.extra_data.get('custom_key')

    return Result.ok(data={"user": user_name})
```

## Working with Results

### Creating Different Types of Responses

```python
@export(mod_name=MOD_NAME, api=True, version="1.0")
async def response_examples(app):
    # Choose the appropriate response type based on your needs

    # 1. Standard success response
    return Result.ok(
        data={"key": "value"},
        info="Operation completed successfully"
    )

    # 2. JSON response
    return Result.json(
        data={"status": "online", "version": "1.0"},
        info="API status retrieved"
    )

    # 3. HTML response
    return Result.html(
        data="<html><body><h1>Welcome</h1></body></html>",
        info="Page rendered"
    )

    # 4. Text response
    return Result.text(
        text_data="Plain text content",
        content_type="text/plain"
    )

    # 5. Binary file response
    return Result.binary(
        data=binary_data,
        content_type="application/pdf",
        download_name="report.pdf"
    )

    # 6. Redirect response
    return Result.redirect(
        url="/dashboard",
        status_code=302
    )
```

### Error Handling

```python
@export(mod_name=MOD_NAME, version=version)
def process_with_validation(user_input):
    # Validate input
    if not user_input:
        return Result.default_user_error(
            info="Empty input is not allowed",
            exec_code=400
        )

    # Process valid input
    try:
        processed_data = process_data(user_input)
        return Result.ok(data=processed_data)
    except Exception as e:
        return Result.default_internal_error(
            info=f"Processing error: {str(e)}",
            exec_code=500
        )
```

### Using lazy_return for Simplified Error Handling

```python
@export(mod_name=MOD_NAME, version=version)
def validate_and_process(data):
    # Validate data
    validation_result = validate_data(data)

    # If validation fails, return the error
    # If validation succeeds, return the processed data
    return validation_result.lazy_return(
        'user',  # Use user error if validation fails
        data={"processed": True, "original": data}  # Return this if successful
    )
```

### Streaming Responses

```python
@export(mod_name=MOD_NAME, api=True, version="1.0")
async def stream_data():
    async def generator():
        for i in range(10):
            yield {"chunk": i}
            await asyncio.sleep(0.5)

    async def cleanup():
        # Cleanup resources when the stream closes
        print("Stream closed, performing cleanup")

    return Result.stream(
        stream_generator=generator(),
        info="Streaming data chunks",
        cleanup_func=cleanup
    )
```

## Advanced Features

### Caching

```python
# Memory caching
@export(mod_name=MOD_NAME, version=version, memory_cache=True,
        memory_cache_max_size=100, memory_cache_ttl=300)
def cached_function(key):
    # Expensive operation here
    return Result.ok(data=compute_expensive_data(key))

# File caching
@export(mod_name=MOD_NAME, version=version, file_cache=True)
def file_cached_function(key):
    # Expensive operation here
    return Result.ok(data=compute_expensive_data(key))
```

### Background Functions

```python
# Memory caching
@export(mod_name=MOD_NAME, version=version)
def function_with_log_running_bg_call():
    # Expensive operation here
    def sync_bg_function():
        print("running in gb")
        compute_expensive_function()

    return Result.ok(data="Starting processing").task(sync_bg_function)

# File caching
@export(mod_name=MOD_NAME, version=version)
async def function_with_log_running_bg_call():
    # Expensive operation here
    async def bg_function():
        print("running in gb")
        await compute_expensive_function()
    return Result.ok(data="Starting processing").task(bg_function())

```

### Lifecycle Management

```python
# Initialization function
@export(mod_name=MOD_NAME, version=version, initial=True)
def initialize_module(app: App):
    # Called when the module is loaded
    print(f"Initializing {MOD_NAME} module")
    # Set up resources, connections, etc.
    return Result.ok(info="Module initialized")

# Exit function
@export(mod_name=MOD_NAME, version=version, exit_f=True)
def cleanup_module(app: App):
    # Called when the application is shutting down
    print(f"Cleaning up {MOD_NAME} module")
    # Release resources, close connections, etc.
    return Result.ok(info="Module cleaned up")
```

### Pre/Post Compute Functions

```python
def log_before_execution(func, *args, **kwargs):
    print(f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
    return args, kwargs

def log_after_execution(result, func, *args, **kwargs):
    print(f"Function {func.__name__} returned: {result}")
    return result

@export(mod_name=MOD_NAME, version=version,
        pre_compute=log_before_execution,
        post_compute=log_after_execution)
def monitored_function(name):
    # Function logic
    return Result.ok(data=f"Hello, {name}!")
```

## URL Patterns for API Endpoints

API endpoints are accessible using the following URL patterns:

- Regular API: `/api/MOD_NAME/{function_name}?param1=value1&param2=value2`
- Server-Sent Events (streaming): `/sse/MOD_NAME/{function_name}?param1=value1&param2=value2`
