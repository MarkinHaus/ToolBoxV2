# Feature Management System

**Status**: ✅ Implemented  
**Version**: 0.1.25  
**Date**: 2025-01-29

## Summary

Modulares Feature-System für ToolBoxV2 mit automatischer Dependency-Installation.

## Usage

```bash
# List all features
tb manifest list

# Enable feature (auto-installs dependencies via pip/uv)
tb manifest enable web

# Disable feature  
tb manifest disable desktop

# Show feature details
tb manifest files isaa
```

## Features

| Name | Default | Dependencies |
|------|---------|--------------|
| core | ✅ enabled | - |
| cli | ✅ enabled | prompt-toolkit, ipython |
| web | ❌ disabled | starlette, uvicorn, httpx |
| desktop | ❌ disabled | PyQt6 |
| isaa | ❌ disabled | openai, litellm |
| exotic | ❌ disabled | scipy, matplotlib |

## Installation Groups

```bash
pip install toolboxv2[cli]        # CLI only
pip install toolboxv2[web]        # Web workers
pip install toolboxv2[desktop]    # PyQt6 GUI
pip install toolboxv2[isaa]       # AI/LLM
pip install toolboxv2[exotic]     # Scientific
pip install toolboxv2[all]        # Everything
pip install toolboxv2[production] # CLI + Web
```

## Python API

```python
from toolboxv2 import _feature_enabled

if _feature_enabled("web"):
    from toolboxv2.workers import start_workers

if _feature_enabled("isaa"):
    from toolboxv2.mods.isaa import Agent
```

## Files

```
toolboxv2/
├── features/
│   ├── core/feature.yaml      # immutable
│   ├── cli/feature.yaml
│   ├── web/feature.yaml
│   ├── desktop/feature.yaml
│   ├── isaa/feature.yaml
│   └── exotic/feature.yaml
├── utils/system/feature_manager.py
├── utils/manifest/schema.py   # FeatureSpec
└── utils/clis/manifest_cli.py # CLI commands
```

## Checklist

- [x] FeatureSpec in schema.py
- [x] FeaturesConfig in schema.py
- [x] features/ directory with YAMLs
- [x] FeatureManager (Singleton metaclass)
- [x] CLI: list, enable, disable, files
- [x] __main__.py integration
- [x] __init__.py: _feature_enabled()
- [x] pyproject.toml optional-dependencies
- [ ] Registry download (CloudM)
- [ ] Registry upload (CloudM)
