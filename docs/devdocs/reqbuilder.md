# Requirements Builder (`utils/extras/reqbuilder.py`)

> **File:** `toolboxv2/utils/extras/reqbuilder.py` (~36 Zeilen)
> **Typ:** Reference
> ⚠️ **Currently a stub** — function exists but prints "Not Implemented".

## API

| Function | Signature | Status |
|----------|-----------|--------|
| `generate_requirements` | `(folder: str, output_file: str)` | ❌ Stub — prints "Not Implemented" |
| `run_pipeline` | `(folder: str, output_file: str)` | ❌ Stub |

### Planned Implementation (commented out)

```python
# Würde pipreqs nutzen um Imports zu scannen und requirements.txt zu generieren
from pipreqs.pipreqs import get_all_imports
imports = set(get_all_imports(folder))
imports.remove('toolboxv2')
with open(output_file, "w") as f:
    f.write("\n".join(imports))
```

## Related

- [RegistryClient](registry_client.md) — mod packaging
