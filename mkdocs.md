site_name: ToolBoxV2 Docs

nav:
  - Home: index.md
  - Toolbox: toolboxv2.md  # relative to the docs/ folder

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          setup_commands:
            - import sys
            - sys.path.insert(0, "toolboxv2")  # ensures GitHub runner resolves correctly
