# toolboxv2 API Reference

This section provides an API reference for key components directly available from the `toolboxv2` package.

## Core Application & Tooling

::: toolboxv2.AppType
handler: python
options:
show_root_heading: true # Keep the class name heading
show_source: true       # Keep the source link
# --- SELECTIVE MEMBER DOCUMENTATION ---
members:
# --- Core Attributes (inspired by AppType) ---
- prefix
        - id
        - start_dir
        - data_dir
        - config_dir
        - logger
        - version
        - loop
        - interface_type
        - alive
        - args_sto
        # --- Key Configuration/Data Structures ---
        - keys
        - defaults
        - flows
        # --- Initialization ---
        - __init__
        # --- Core Lifecycle & Control ---
        - exit
        - a_exit
        - idle         # Sync idle loop
        - a_idle       # Async idle loop
        - debug        # Property for debug status
        - set_logger   # Method to configure logging
        # --- Module Management ---
        - load_mod
        - init_module    # Often used async variant/entry point
        - remove_mod
        - a_remove_mod
        - reload_mod
        - watch_mod      # Useful dev feature
        - get_mod        # Access loaded modules
        - remove_all_modules
        - a_remove_all_modules
        # --- Function Execution ---
        - run_function
        - a_run_function
        - run_any        # Flexible runner
        - a_run_any      # Flexible async runner
        - run_a_from_sync # Utility for async calls
        - get_function   # Retrieve registered functions
        # --- Decorators (Primary User Interaction) ---
        - tb             # The main decorator for toolbox functions
        # --- Introspection & Testing ---
        - print_functions
        - execute_all_functions # For running tests/suites
        # --- Web/UI related ---
        - web_context
        # --- Utilities ---
        - print          # Static print helper
        - sprint         # Static print helper

::: toolboxv2.MainTool
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: toolboxv2.get_app
    handler: python
    options:
      show_root_heading: true
      show_source: true

## System Utilities & Configuration

::: toolboxv2.FileHandler
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: toolboxv2.utils
    handler: python
    options:
      show_root_heading: true
      heading_level: 3
      show_bases: false
      show_submodules: true

::: toolboxv2.show_console
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Logging

::: toolboxv2.get_logger
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: toolboxv2.setup_logging
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Styling & Console Output

::: toolboxv2.Style
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: toolboxv2.Spinner
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: toolboxv2.remove_styles
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Data Types & Structures

::: toolboxv2.AppArgs
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: toolboxv2.Result
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: toolboxv2.ApiResult
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: toolboxv2.RequestData
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Security

::: toolboxv2.Code
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Modules & Flows

::: toolboxv2.mods
    handler: python
    options:
      show_root_heading: true
      heading_level: 3
      show_submodules: true

::: toolboxv2.flows_dict
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: toolboxv2.TBEF
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Other Exposed Items

::: toolboxv2.ToolBox_over
    handler: python
    options:
      show_root_heading: true
      show_source: true
