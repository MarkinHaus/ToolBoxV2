# State System (`utils/system/state_system.py`)

> **File:** `toolboxv2/utils/system/state_system.py` (~422 Zeilen)
> **Typ:** Reference
> Auto-Update System — lädt neue `tb`-Executable von GitHub Releases herunter.

## Why This Matters

Wenn ein User `tb update` ausführt, nutzt ToolBoxV2 `state_system.py` um:
1. Aktuelle OS/Architektur zu erkennen (Windows/Linux/Mac, x64/arm64)
2. Neueste Release-Version von GitHub zu finden
3. Die passende `.zip` herunterzuladen und zu entpacken
4. Die alte Executable zu ersetzen

## API Reference

| Function | Signature | Description |
|----------|-----------|-------------|
| `detect_os_and_arch` | `() → (os_name, arch_name)` | Detects `windows`/`linux`/`darwin` + `amd64`/`arm64` |
| `query_executable_url` | `(version=None) → (url, version, filename)` | Query GitHub API for latest release |
| `find_highest_zip_version` | `(path) → str?` | Find highest version number in local zips |
| `download_executable` | `(url, file_name) → str?` | Download with streaming, `chmod +x` on Unix |
| `get_state_from_app` | `(app) → Dict` | Current version + platform state |
| `download_executable` | `(url, filename)` | Streams download in 8KB chunks |

## How-to: Update ToolBoxV2

```bash
# Via CLI
tb update
# → Detects OS, downloads latest, replaces executable

# Check current state
tb status
# → Version: 0.1.28, Platform: windows/amd64
```

```python
from toolboxv2.utils.system.state_system import detect_os_and_arch, query_executable_url

os_name, arch = detect_os_and_arch()  # → ("windows", "amd64")
url, version, filename = query_executable_url()
print(f"Latest: {version}, Download: {url}")
```

## Related

- [Core Types](types.md) — `AppType` calls state functions
