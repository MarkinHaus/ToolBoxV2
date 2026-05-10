# Quickstart — ToolBoxV2

> Up and running in under 5 minutes.

---

## 1 — Install

### Linux / macOS

```bash
curl -fsSL https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.sh | bash
```

### Windows

```powershell
irm "https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.ps1" | tee tbInstaller.ps1 | % { & ([scriptblock]::Create($_)) }
```

### pip (all platforms)

```bash
pip install ToolBoxV2
```

### Source (developer mode)

```bash
git clone https://github.com/MarkinHaus/ToolBoxV2.git
cd ToolBoxV2
uv sync        # or: pip install -e .
```

---

## 2 — First Run

```bash
tb
```

On first launch, choose your **profile**:

| # | Profile     | Behavior                              | Use Case                    |
|---|-------------|---------------------------------------|-----------------------------|
| 1 | `consumer`  | Launches GUI                          | App / Mod users             |
| 2 | `homelab`   | Interactive dashboard                 | Local multi-mod setup       |
| 3 | `server`    | ASCII status overview, then exit      | Infrastructure management   |
| 4 | `business`  | 3-line health summary, then exit      | Quick health checks         |
| 5 | `developer` | Interactive dashboard + dev hints     | Core / Mod development      |

Default: `homelab` (press Enter to confirm).

---

## 3 — Verify

```bash
tb --version
tb status
tb manifest validate
```

---

## Next Steps

| Goal | Command |
|------|---------|
| Configure TB | `tb manifest init` |
| Install a module | `tb install <mod>` |
| Start web workers | `tb workers start` |
| View all commands | `tb --help` |

- [Full Installation Guide](installation.md)
- [Configuration Guide](../configuration.md)
- [Module Management](../mods.md)
- [ISAA Integration](../isaa.md)
