site_name: ToolBoxV2 Docs

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [toolboxv2]
nav:
  - Home: index.md
  - Toolbox: toolboxv2.md
