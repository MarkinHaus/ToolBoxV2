# Quickstart — ToolBoxV2

> Get started with ToolBoxV2 in under 5 minutes.

## Install

### Linux / macOS

```bash
curl -sSL https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/main/install.sh | bash
```

<!-- verified: installer.sh::main -->

### Windows

```powershell
irm https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/main/install.ps1 | iex
```

<!-- verified: installer.ps1::main -->

### pip (all platforms)

```bash
pip install ToolBoxV2
```

### git (developer mode)

```bash
git clone https://github.com/MarkinHaus/ToolBoxV2.git
cd ToolBoxV2
pip install -e .
```

---

## First Run

On first `tb` call you will be prompted to choose a **profile**:

| Profile | Behavior | Use Case |
|---------|----------|----------|
| `consumer` | Launches GUI | App/Mod users |
| `homelab` | Interactive dashboard | Local multi-mod setup |
| `server` | ASCII status overview, then exit | Infrastructure management |
| `business` | 3-line health summary, then exit | Quick health checks |
| `developer` | Interactive dashboard + dev hints | Core/Mod development |

Profile defaults: `homelab` (index 2).

<!-- verified: schema.py::ProfileType -->
<!-- verified: first_run.py::PROFILES -->

---

## Verify Installation

```bash
tb --version
tb manifest validate
```

<!-- verified: __init__.py::__version__ -->

---

## Next Steps

- [Configuration Guide](../docs/configuration.md)
- [Module Management](../docs/mods.md)
- [Feature System](../docs/feature_system.md)
- [ISAA Integration](../docs/isaa.md)