# tb_atomic

Dev/build tool that packs a **single ToolBoxV2 component** — plus only its
minimal transitive in-tree dependencies — into a **standalone, install-free**
Python distribution. The packed module imports on a machine that has never seen
ToolBoxV2, with no runtime import hooks and no network fetch.

This is a rewrite. The old `tb_atomic` was a *runtime fetcher* (a
`sys.meta_path` finder pulling `.py` files over HTTP at import time). That is
gone. `tb_atomic` is now a pure **packer**: it produces real wheels at build
time, and a separate script publishes them.

---

## What it does (and does not)

**Does:** statically crawl an entry file's load-time imports, vendor that exact
set of source files, rewrite the `toolboxv2.` namespace to a neutral `tbv.`,
split shared base modules into a reusable `tbv-core`, and emit ready-to-build
package directories.

**Does not:** execute any ToolBoxV2 code, run a live framework, or guarantee
that *every* runtime code path works. It guarantees the module **imports
standalone** and that load-time behaviour matches. Calls that reach back into
un-packed framework code raise a clear `ImportError` at call time (see
[Limitations](#limitations)).

---

## Pipeline

```
tb_atomic pack <entry.py>
   │
   ├─ 1. crawl        load-time transitive in-tree closure (relative + absolute)
   ├─ 2. split        core (shared base)  vs  leaf (entry-specific)
   ├─ 3. rewrite      toolboxv2.*  ->  tbv.*   (+ generate tbv/_shim.py)
   ├─ 4. deps         classify external deps into 3 buckets
   ├─ 5. emit         tbv-core/  +  <leaf>/   (PEP420 namespace, pyproject each)
   └─ 6. (gate)       run import-closed TB tests against the packed copy
```

### 1. Crawl — load-time only

The crawler resolves both **absolute** (`from toolboxv2.x import Y`) and
**relative** (`from ..utils import Y`) imports. Crucially it follows **only
load-time imports** — imports that execute when the module is imported.

Imports nested inside a **function body** are *deferred*: they are **not**
vendored (they would drag in most of the framework). Imports at module level —
including inside a top-level `try: / except ImportError:` or a `class` body —
**are** load-time and are vendored.

This is what keeps the pack "atomic". A naive full-static crawl of a real ISAA
module pulls in ~130 modules (most of ToolBoxV2); the load-time closure is
typically a handful.

### 2. Core / leaf split

Modules whose name is a **re-export source in `toolboxv2/__init__.py`** (e.g.
`utils.system.types`, `utils.extras.Style`, `utils.security.cryp`) go into the
shared **`tbv-core`** package. Everything else goes into the **leaf** package.

The split rule is an allowlist seeded automatically from `__init__.py` — the
file already declares `symbol -> source module`, so no list is maintained by
hand.

### 3. Namespace rewrite + shim

All `toolboxv2` references are rewritten to a neutral shared namespace `tbv`,
so multiple packed modules co-install and never clash with a real ToolBoxV2:

| Original | Rewritten |
|---|---|
| `from toolboxv2.mods.x import Y` | `from tbv.mods.x import Y` |
| `from toolboxv2 import Result` | `from tbv._shim import Result` |
| `import toolboxv2` | `import tbv._shim as toolboxv2` |

`tbv` is a **PEP 420 namespace package** — there is **no `__init__.py`**
anywhere in the tree, so `tbv-core` and any number of leaves can each contribute
files to `tbv.*` without file collisions, and shared classes keep a single
identity (`tbv.mods.isaa.base.VectorStores.types.Chunk` is the same class for
every dependent).

Top-level symbols are mirrored in a generated **`tbv/_shim.py`**:

- `get_logger` → real (`logging.getLogger`)
- self-contained classes whose source module was packed (`Result`, `Style`,
  `Spinner`, `Code`, …) → re-exported from the vendored module
- framework-bound symbols (`App`, `get_app`, `flows_dict`, `ToolBox_over`) and
  any symbol whose source module was **not** packed → a stub that raises a clear
  `ImportError` on use.

### 4. External dependencies — 3 buckets

Third-party (non-stdlib, non-ToolBoxV2) imports are classified, not guessed:

| Bucket | Rule | Goes to |
|---|---|---|
| **hard** | unguarded, module-level | `[project.dependencies]` |
| **optional** | inside `try/except ImportError` (anywhere) | `optional-dependencies.optional` |
| **full** | everything, incl. lazy in-function imports | `optional-dependencies.full` |

Import names are mapped to PyPI distribution names (`faiss` → `faiss-cpu`,
`cv2` → `opencv-python`, `yaml` → `pyyaml`, …). The test gate installs
`.[full]` so behaviour matches the original even for lazily-imported libraries.

### 5. Emit + dedup (D-mid hybrid)

`tbv-core` is a **monotonic superset**. State lives in
`tbv-core/.tbv_core.json` (`{version, modules: {module: sha256}}`). On each
pack:

- new core modules are added; existing ones are kept (never shrinks);
- if a module's content hash changed, the patch version is bumped and a
  `WARNING` is printed (it is shared — verify back-compat for dependents);
- leaves pin `tbv-core>=<current version>`.

So installing several packed modules pulls `tbv-core` exactly once.

### 6. Test gate

`select_tests()` picks the ToolBoxV2 tests that are **import-closed** over the
packed set — every in-tree module the test imports is inside the closure. Those
tests get the same `toolboxv2.`→`tbv.` rewrite and run against the standalone
copy in a clean environment.

Many TB tests pull in `App` / `get_app` and are therefore **not** isolatable;
the gate reports and skips them rather than pretending they passed.

---

## Usage — packing

```bash
# from a ToolBoxV2 checkout
python tb_atomic.py pack toolboxv2/utils/workers/fast_tb.py \
    --root . \
    -o dist_atomic \
    [--name tbv-fast-tb] \
    [--version 0.1.0]
```

Output:

```
dist_atomic/
├─ tbv-core/                 # shared base (only if the entry needs core symbols)
│  ├─ tbv/_shim.py
│  ├─ tbv/utils/extras/Style.py
│  ├─ .tbv_core.json
│  └─ pyproject.toml
└─ tbv-fast-tb/              # the leaf
   ├─ tbv/utils/workers/fast_tb.py
   └─ pyproject.toml
```

Updating a packed module = re-run `pack` against the current source. The core
merges forward; the leaf is regenerated.

---

## Usage — publishing

`publish_tb_atomic.py` operates on the **packed output directory** and builds /
uploads **core-first** (leaves depend on it).

```bash
python publish_tb_atomic.py dist_atomic [targets] [flags]
```

**Targets** (combinable; default `--testpypi` if none given):

| Flag | Effect |
|---|---|
| `--testpypi` | wheel + sdist → TestPyPI |
| `--pypi` (`--prod`) | wheel + sdist → production PyPI |
| `--registry` | zipped package → TB Registry (via `RegistryClient`) |

**Flags:** `--check` (build + `twine check`, no upload), `--build-only`,
`--no-clean`, `--packages a,b` (publish a subset).

**Auth** (environment):

```bash
# PyPI / TestPyPI
export TWINE_PASSWORD=<token>           # username is forced to __token__

# TB Registry
export TB_REGISTRY_TOKEN=<jwt>
export REGISTRY_BASE_URL=https://registry.simplecore.app   # optional override
```

PyPI upload uses `twine`. Registry upload imports `RegistryClient` from the
local `toolboxv2` checkout, checks the token belongs to a **verified
publisher**, then `create_package` (idempotent — ignores "already exists") →
`upload_version` with a zipped package and sha256.

Examples:

```bash
python publish_tb_atomic.py dist_atomic --check            # dry: build + check
python publish_tb_atomic.py dist_atomic                     # -> TestPyPI
python publish_tb_atomic.py dist_atomic --pypi --registry   # both, core first
```

---

## Validated

Against a real `github.com/MarkinHaus/ToolBoxV2` checkout, in a fresh venv with
`toolboxv2` **absent**:

- `tbv.mods.isaa.base.hybrid_memory` imports standalone; live FAISS
  `add_embeddings` + `search` roundtrip returns the correct nearest neighbour.
- `tbv.utils.workers.fast_tb` imports standalone; `FastTB` route registration
  works.
- Shim: `get_logger` real; `get_app()` raises `ImportError`.
- A real TB test (`test_styles.py`), selected by the gate and rewritten,
  passes against the packed copy (6 passed).
- Core merge: 2 → 8 modules across packs bumped `tbv-core` 0.1.0 → 0.1.1;
  leaf pinned `tbv-core>=0.1.1`.
- Publisher: all packages build wheel+sdist and pass `twine check` core-first;
  registry zip/metadata/order verified mechanically.

---

## Limitations

Honest list of what this does **not** do:

- **Load-time scope only.** Code paths that lazily import un-packed framework
  modules at *call* time will raise `ImportError` (by design — those need a full
  ToolBoxV2). The pack guarantees import + load-time behaviour, not 100% runtime
  coverage.
- **Intermediate `__init__` side effects are dropped.** Parent packages
  (`toolboxv2.mods.isaa…`) are synthesised as empty PEP420 namespaces. If a real
  `__init__.py` did registration or re-exports, that behaviour is not carried
  over.
- **Dependency classification is heuristic.** A module-level guarded import
  lands in `optional`; a lazy in-function import lands only in `full`. If a lazy
  import is actually mandatory, promote it to `dependencies` by hand.
- **Core breaking-change detection is hash + warning only.** It detects *that* a
  shared module changed, not whether the change is API-breaking. Version pins
  are `>=`; a genuinely incompatible change needs a manual major bump and
  narrowed pins.
- **Live registry upload is untested end-to-end** here — it needs a real
  verified-publisher token and a reachable registry. The code path matches the
  `RegistryClient` API; the zip/metadata/ordering are verified.
- **No `crawl` / `update` CLI subcommands yet** — only `pack`. Updating is
  re-running `pack`.
- **Pure-Python in-tree modules only.** C extensions and data files are not
  handled.

---

## Files

| File | Purpose |
|---|---|
| `tb_atomic.py` | the packer (`pack` CLI + crawl/rewrite/split/gate functions) |
| `publish_tb_atomic.py` | build & upload packed dists to PyPI and/or TB Registry |
