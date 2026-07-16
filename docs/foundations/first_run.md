# First Run — ToolBoxV2 Onboarding

> What happens when you run `tb` for the first time.

## What Happens on First `tb`

1. **Profile Check**: `_get_profile()` reads `app.profile` from `tb-manifest.yaml`.
   - If profile is `None` or manifest missing → triggers first-run.
2. **Profile Selection**: `run_first_run()` displays the profile picker.
3. **Profile Persisted**: Choice written to manifest via `cmd_set()`.
4. **Config Wizard**: Offered to configure database, workers, auth.

```python
# __main__.py::main_helper()
from toolboxv2.__main__ import _get_profile
profile = _get_profile()
if profile is None:
    from toolboxv2.utils.clis.first_run import run_first_run
    profile = run_first_run()
```

<!-- verified: __main__.py::_get_profile -->
<!-- verified: __main__.py::main_helper -->

---

## Profile Selection

| # | Key | Label | Description |
|---|-----|-------|-------------|
| 1 | `consumer` | 👤 Consumer | App/Mod users — launches GUI |
| 2 | `homelab` | 🏠 Homelab | Local multi-mod setup — interactive dashboard |
| 3 | `server` | 🖥️ Server | Infrastructure — ASCII overview, exit |
| 4 | `business` | 💼 Business | Quick status — 3-line health summary, exit |
| 5 | `developer` | 🛠️ Developer | Core/Mod dev — dashboard + hints |

Default: `homelab` (index 2).

```python
PROFILES = {
    "consumer":  ("👤 Consumer",  "Ich nutze eine App / eine Mod. Einfach starten."),
    "homelab":   ("🏠 Homelab",   "Ich betreibe mehrere Mods, Features, Flows lokal."),
    "server":    ("🖥️  Server",    "Ich manage ein verteiltes System / IT-Infrastruktur."),
    "business":  ("💼 Business",  "Ich brauche einen schnellen Gesundheitsstatus."),
    "developer": ("🛠️  Developer", "Ich entwickle Mods, Features oder den Core."),
}
```


<!-- verified: first_run.py::PROFILES -->
<!-- verified: schema.py::ProfileType -->

---

## Change Profile Later

### Via CLI

```bash
tb manifest set app.profile <profile_name>
```

### Via Code

```python
from toolboxv2.utils.manifest import ManifestLoader, ProfileType
from toolboxv2 import tb_root_dir

loader = ManifestLoader(tb_root_dir)
manifest = loader.load()
manifest.app.profile = ProfileType.DEVELOPER
loader.save(manifest)
```

<!-- verified: schema.py::AppConfig -->

---

## Config Wizard

After profile selection you are asked:

```
Run config wizard now? [Y/n]:
```

The wizard (`run_config_wizard()`) configures:

- **Database mode**: LC (local JSON), LR (local Redis), RR (remote Redis), CB (MinIO)
- **Workers**: HTTP/WebSocket port configuration
- **Auth provider**: Custom, Local, or None
- **Paths**: data/, config/, logs/ directories

```python
# Invoked from first_run.py
if run_wiz != "n":
    from toolboxv2.utils.clis.config_wizard import run_config_wizard
    run_config_wizard(tb_root_dir)
```

<!-- verified: first_run.py::run_first_run -->
<!-- verified: schema.py::DatabaseMode -->

---

## Profile → Runner Mapping

After first-run, `main_helper()` maps profile to runner:

| Profile | Runner Called |
|---------|--------------|
| `consumer` | `gui` |
| `server` | `_run_server_overview()` → exit |
| `business` | `_run_business_overview()` → exit |
| `homelab`, `developer` | `default` (interactive dashboard) |

```python
if profile == "consumer":
    runner_name = "gui"
elif profile == "server":
    _run_server_overview()
    return
elif profile == "business":
    _run_business_overview()
    return
else:
    runner_name = "default"
```

<!-- verified: __main__.py::_run_server_overview -->
<!-- verified: __main__.py::_run_business_overview -->
