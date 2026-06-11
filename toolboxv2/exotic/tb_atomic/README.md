# tb_atomic

Import single **ToolBoxV2** components — and only their minimal, transitive
in-tree dependencies — from remote sources (gist / GitHub raw / your
`registry.simplecore.app`), **without installing the full ToolBoxV2**.

`from toolboxv2.mods.isaa.memory_layer import MemoryLayer` just works, even on
a machine that has never seen ToolBoxV2, by fetching the needed `.py` files at
import time, hash-pinning them, and caching them to disk.

## How it works

Two cooperating layers, installed via a `sys.meta_path` finder:

| Layer | Purpose |
|-------|---------|
| **Remote modules** | Real source pulled over HTTP, optionally `sha256`-pinned, cached under `~/.tb_atomic_cache/`. |
| **Stubs** | Lightweight stand-ins for heavy top-level symbols (e.g. `toolboxv2.get_logger`) so importing a leaf module does **not** drag in the whole framework. |

Intermediate packages (`toolboxv2`, `toolboxv2.mods`, …) are synthesised as
empty namespace packages on demand. **Stubs take precedence over remote
sources** of the same name — so the heavy top-level `toolboxv2/__init__.py` is
never executed when you stub it.

## Install

Directly from the gist (a gist is a git repo):

```bash
pip install "git+https://gist.github.com/<USER>/<GIST_ID>.git"
```

Pin a specific revision:

```bash
pip install "git+https://gist.github.com/<USER>/<GIST_ID>.git@<COMMIT_SHA>"
```

Only dependency is `requests`.

## Quickstart

```python
import tb_atomic

# Load the manifest that ships with ToolBoxV2 — registers every module + stub.
tb_atomic.load_manifest(
    "https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/main/toolboxv2/atomic.json"
)

# Now import as if ToolBoxV2 were installed.
from toolboxv2.mods.isaa.memory_layer import MemoryLayer
```

### Without a manifest (manual)

```python
import tb_atomic, logging

# Top-level stub instead of the heavy real __init__.
tb_atomic.register_stub("toolboxv2", get_logger=logging.getLogger)

RAW = "https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/main"
tb_atomic.register("toolboxv2.utils.types",  f"{RAW}/toolboxv2/utils/types.py")
tb_atomic.register("toolboxv2.utils.logger", f"{RAW}/toolboxv2/utils/logger.py")
tb_atomic.register(
    "toolboxv2.mods.isaa.memory_layer",
    f"{RAW}/toolboxv2/mods/isaa/memory_layer.py",
    sha256="abc123…",  # optional integrity pin
)

from toolboxv2.mods.isaa.memory_layer import MemoryLayer
```

## Generating the manifest (`atomic.json`)

Run the AST crawler **once** on the ToolBoxV2 checkout. It statically resolves
every transitive `toolboxv2.*` import from an entry file — no execution, no
markup in the source files required — and records each module's path + sha256.

```bash
tb-atomic crawl toolboxv2/mods/isaa/memory_layer.py \
    --root toolboxv2 \
    --top toolboxv2 \
    --base https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/main \
    --version 0.4.2 \
    -o atomic.json
```

Then add the stubs section by hand (the symbols you want substituted instead
of fetched):

```json
{
  "version": "0.4.2",
  "base_url": "https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/main",
  "modules": {
    "toolboxv2.mods.isaa.memory_layer": {
      "path": "toolboxv2/mods/isaa/memory_layer.py",
      "sha256": "…"
    },
    "toolboxv2.utils.types":  { "path": "toolboxv2/utils/types.py",  "sha256": "…" },
    "toolboxv2.utils.logger": { "path": "toolboxv2/utils/logger.py", "sha256": "…" }
  },
  "stubs": {
    "toolboxv2": ["get_logger"]
  }
}
```

Commit `atomic.json` to the repo; let CI regenerate it on each release.

> Stub defaults: `get_logger` maps to `logging.getLogger`. Any other stubbed
> name raises a clear `ImportError` on use until you register a real
> replacement via `tb_atomic.register_stub(...)`.

## Caching & overrides

The cache lives under `~/.tb_atomic_cache/` (override with
`tb_atomic.configure(cache_dir=...)` or the `TB_ATOMIC_CACHE` env path you set
yourself).

```python
# Re-fetch a module on next import (drop its cached copy now):
tb_atomic.register(name, url, force=True)
tb_atomic.invalidate("toolboxv2.utils.types")

# Refresh the whole manifest:
tb_atomic.load_manifest(url, force=True)

# Auto-stale after N seconds:
tb_atomic.register(name, url, ttl=3600)

# Version-pinned: bumping `version` in register/manifest forces a re-fetch
# even if a cached copy exists.
tb_atomic.register(name, url, version="0.4.3")

# Clear everything (or specific modules):
tb_atomic.clear_cache()
tb_atomic.clear_cache("toolboxv2.utils.types")
```

CLI:

```bash
tb-atomic clear-cache                       # all
tb-atomic clear-cache toolboxv2.utils.types # one
```

## Use as a uv script (PEP 723)

`uv` resolves inline script metadata, so a standalone script can declare
`tb-atomic` (from the gist) as a dependency and run with zero manual setup:

```python
# memory_demo.py
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "tb-atomic @ git+https://gist.github.com/<USER>/<GIST_ID>.git",
# ]
# ///
import tb_atomic

tb_atomic.load_manifest(
    "https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/main/toolboxv2/atomic.json"
)

from toolboxv2.mods.isaa.memory_layer import MemoryLayer

print("Loaded:", MemoryLayer)
```

Run it — uv creates an ephemeral environment, installs `tb-atomic` from the
gist, and executes:

```bash
uv run memory_demo.py
```

Add a one-off dependency without editing the script:

```bash
uv run --with "git+https://gist.github.com/<USER>/<GIST_ID>.git" memory_demo.py
```

## API

| Function | Purpose |
|----------|---------|
| `register(name, url, *, sha256=None, version=None, ttl=0, is_package=False, force=False)` | Register a remote module source. |
| `register_stub(name, **attrs)` | Register a lightweight stand-in module. |
| `load_manifest(url_or_path, *, force=False, prefetch=False)` | Register everything from an `atomic.json`. |
| `invalidate(name)` | Drop one module's cache. |
| `clear_cache(*names)` | Clear all or specific modules. |
| `configure(cache_dir=...)` | Override the cache directory. |

## Limitations

- C extensions and data files are not handled — pure-Python in-tree modules only.
- A stubbed symbol must actually be substitutable; truly framework-bound objects
  should raise rather than silently no-op (that is the default behaviour).
- Integrity pinning is opt-in: provide `sha256` for anything you don't fully control.
