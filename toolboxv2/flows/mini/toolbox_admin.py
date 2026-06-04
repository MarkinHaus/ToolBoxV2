# toolboxv2/flows/toolbox_admin.py
# Toolbox Admin Flow - Agent-powered CLI for full Toolbox management

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle

from toolboxv2 import App, Result, get_app, tb_root_dir
from toolboxv2.utils.extras.Style import Style, cls, Spinner

NAME = "toolbox_admin"
ICON = "admin_panel_settings"
AUTH = False

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """# ToolBox Admin Agent v2.0

Du bist der ToolBox-Administrator-Agent "tb-admin".
Du hast VOLLEN Zugriff auf das ToolBoxV2-System.

## Kernprinzipien

1. **Quellcode > Docs**: Die Dokumentation enthaelt teilweise veraltete/grobe Infos.
   Nutze `docs_read` und `docs_lookup` als Orientierung, aber VERIFIZIERE immer
   gegen den tatsaechlichen Quellcode via `docs_lookup(include_code=True)` oder `shell`.
   Cross-Referenzen im Code sind Ground Truth.

2. **Manifest ist die Single Source of Truth** fuer Konfiguration.
   Aenderungen immer ueber `manifest_set` machen — das synct automatisch `.env`.

3. **Streaming**: Du arbeitest im Streaming-Modus. Der User sieht jeden Schritt live.

## Architektur-Wissen

### Worker-System
- Python workers verbinden sich DIREKT zu nginx (kein Zwischenlayer)
- `cli_worker_manager.py` verwaltet nginx-Config (upstream ports, health checks)
- Workers werden ueber `tb workers` CLI gestartet/verwaltet
- Typen: `http_worker`, `ws_worker`, `broker`, `event`
- Jeder Worker bekommt einen pre-allocated upstream Port in nginx

### GUI / Server
- `tb gui` startet die Desktop/Web GUI (CustomTkinter oder Web)
- `tb browser` startet den Browser-Client
- Server-Config laeuft ueber das Manifest (`nginx` section)
- `tb services` verwaltet Background-Services (auto-start via service manager)

### ISAA (Agent Framework)
- Agents werden ueber `AgentBuilder` erstellt und bei ISAA registriert
- Tools werden ueber `builder.add_tool()` oder `tool_mgr.register_cli_tool()` hinzugefuegt
- Agent-Config: fast_model, complex_model, max_iterations, history_length
- Sessions sind persistent (session_id)
- `a_stream_verbose` fuer Live-Output, `a_run` fuer Silent-Execution

### TB CLI - Vollstaendige Runner-Liste
Core:     user, run, db, workers, services, registry, build, mods, flow
Network:  http_worker, ws_worker, broker, event, p2p, LiveSync
Utility:  login, logout, status, session, manifest, access, obs
Dev:      venv, mcp, gui, browser, docksh, docker-image
Special:  llm-gateway, fl, jsx, ytss
Default:  default (interactive dashboard)

Jeder Runner wird via `tb <runner>` aufgerufen. `tb <runner> --help` zeigt Optionen.

### Flows & Mods
- Flows: async Funktionen in `toolboxv2/flows/<name>.py` mit `NAME`, `run(app, args)`
- Mods: Module in `toolboxv2/mods/<name>/` mit `@export` dekorierte Funktionen
- Templates on-demand via CloudM erstellen
- Testen via `shell` mit `python -m unittest discover` oder direkt `tb run --test`

## Verfuegbare Tools

### Dokumentation
- `docs_read(query, section_id, file_path, tags, max_results)` - Docs durchsuchen
- `docs_lookup(name, element_type, file_path, language, include_code)` - Code-Elemente finden
- `docs_suggestions()` - Verbesserungsvorschlaege fuer Docs
- `docs_sync()` - Index mit Filesystem synchronisieren

### Manifest / Konfiguration
- `manifest_show(section, as_json)` - Manifest anzeigen
- `manifest_get(key)` - Einzelnen Wert lesen (dotted key: "database.mode")
- `manifest_set(key, value)` - Wert setzen (synct .env automatisch)
- `manifest_validate()` - Manifest validieren
- `manifest_apply(dry_run)` - Sub-Configs generieren
- `manifest_feature(action, feature_name)` - Features enable/disable/list/files

### Toolbox Core
- `toolbox_execute(module_name, function_name, kwargs)` - Beliebige Mod-Funktion
- `toolbox_list_mods(module_name)` - Module und Funktionen
- `toolbox_status()` - Systemstatus
- `flow_manage(action, name, content)` - Flows verwalten (list/read/create/run)
- `cloudm_action(action, **kwargs)` - CloudM (Nutzer/Ordner/LiveSync)

### System
- `shell` - Shell-Befehle ausfuehren
- `memory_recall` / `memory_save` - Langzeit-Gedaechtnis

## Arbeitsweise

1. Bei Config-Fragen: IMMER `manifest_show` oder `manifest_get` zuerst
2. Bei Code-Fragen: `docs_lookup(include_code=True)` → verifiziere im Quellcode
3. Bei Docs-Fragen: `docs_read` als Startpunkt, dann Cross-Check mit Code
4. Bei mehrstufigen Aufgaben: Schritte AUSFUEHREN, nicht nur beschreiben
5. Vor destruktiven Aktionen: `think()` nutzen
6. Deutsch wenn User deutsch spricht
"""


# ---------------------------------------------------------------------------
# Tool Builders
# ---------------------------------------------------------------------------


def _build_docs_tools(app):
    """Build DocsSystem tools for the admin agent."""
    tools = []
    _docs_system = None

    def _get_docs():
        nonlocal _docs_system
        if _docs_system is None:
            from toolboxv2.utils.extras.mkdocs import create_docs_system
            _docs_system = create_docs_system(
                project_root=str(tb_root_dir.parent),
                include_dirs=["toolboxv2", "flows", "mods", "utils", "docs"],
            )
        return _docs_system

    async def docs_read(
        query: str = None,
        section_id: str = None,
        file_path: str = None,
        tags: str = None,
        max_results: int = 25,
    ) -> str:
        """Search and read documentation sections."""
        try:
            ds = _get_docs()
            await ds.initialize()
            tag_list = [t.strip() for t in tags.split(",")] if tags else None
            result = await ds.read(
                query=query,
                section_id=section_id,
                file_path=file_path,
                tags=tag_list,
                max_results=max_results,
            )
            return json.dumps(result, ensure_ascii=False, default=str, indent=2)
        except Exception as e:
            return f"Docs read error: {e}"

    async def docs_lookup(
        name: str = None,
        element_type: str = None,
        file_path: str = None,
        language: str = None,
        include_code: bool = False,
        max_results: int = 25,
    ) -> str:
        """Look up code elements (classes, functions, methods) across Python/JS/TS."""
        try:
            ds = _get_docs()
            await ds.initialize()
            result = await ds.lookup_code(
                name=name,
                element_type=element_type,
                file_path=file_path,
                language=language,
                include_code=include_code,
                max_results=max_results,
            )
            return json.dumps(result, ensure_ascii=False, default=str, indent=2)
        except Exception as e:
            return f"Docs lookup error: {e}"

    async def docs_suggestions(max_suggestions: int = 20) -> str:
        """Get documentation improvement suggestions."""
        try:
            ds = _get_docs()
            await ds.initialize()
            result = await ds.get_suggestions(max_suggestions=max_suggestions)
            return json.dumps(result, ensure_ascii=False, default=str, indent=2)
        except Exception as e:
            return f"Docs suggestions error: {e}"

    async def docs_sync() -> str:
        """Sync documentation index with filesystem/git changes."""
        try:
            ds = _get_docs()
            await ds.initialize()
            result = await ds.sync()
            return json.dumps(result, ensure_ascii=False, default=str, indent=2)
        except Exception as e:
            return f"Docs sync error: {e}"

    async def docs_inventory(
        focus_dirs: str = "",
        max_classes_per_file: int = 5,
        max_methods_per_class: int = 3,
        include_functions: bool = True,
    ) -> str:
        """Generate project inventory: what files, classes, and functions exist."""
        try:
            ds = _get_docs()
            await ds.initialize()
            result = await ds.generate_inventory(
                focus_dirs=focus_dirs.split(",") if focus_dirs else None,
                max_classes_per_file=max_classes_per_file,
                max_methods_per_class=max_methods_per_class,
                include_functions=include_functions,
                format_type="markdown",
            )
            return result.get("content", json.dumps(result, ensure_ascii=False, default=str))
        except Exception as e:
            return f"Inventory error: {e}"

    async def docs_relationship_map(
        focus_dirs: str = "",
        focus_classes: str = "",
        max_nodes: int = 40,
        format_type: str = "markdown",
    ) -> str:
        """Generate relationship map showing how components connect (Mermaid diagram)."""
        try:
            ds = _get_docs()
            await ds.initialize()
            result = await ds.generate_relationship_map(
                focus_dirs=focus_dirs.split(",") if focus_dirs else None,
                focus_classes=focus_classes.split(",") if focus_classes else None,
                max_nodes=max_nodes,
                format_type=format_type,
            )
            return result.get("content", json.dumps(result, ensure_ascii=False, default=str))
        except Exception as e:
            return f"Relationship map error: {e}"

    async def docs_export_docmap(
        output_path: str = "",
        format_type: str = "html",
        focus_dirs: str = "",
        title: str = "",
    ) -> str:
        """Export complete DocMap (inventory + relationships) as HTML or Markdown file."""
        try:
            ds = _get_docs()
            await ds.initialize()
            result = await ds.export_docmap(
                output_path=output_path or None,
                format_type=format_type,
                focus_dirs=focus_dirs.split(",") if focus_dirs else None,
                title=title or None,
            )
            if output_path:
                return f"DocMap exported to {result.get('output_path', output_path)}"
            return result.get("content", "Export failed — no content generated")
        except Exception as e:
            return f"DocMap export error: {e}"



    tools.extend([
        (docs_read, "docs_read",
         "Dokumentation durchsuchen (query, section_id, file_path, tags, max_results)",
         ["docs", "read"]),
        (docs_lookup, "docs_lookup",
         "Code-Elemente nachschlagen (name, element_type, file_path, language, include_code)",
         ["docs", "code"]),
        (docs_suggestions, "docs_suggestions",
         "Docs-Verbesserungsvorschlaege abrufen",
         ["docs", "read"]),
        (docs_sync, "docs_sync",
         "Docs-Index mit Filesystem synchronisieren",
         ["docs", "write"]),
        (docs_inventory, "docs_inventory",
         "Projekt-Inventar generieren: Dateien, Klassen, Funktionen "
         "(focus_dirs, max_classes_per_file, max_methods_per_class, include_functions)",
         ["docs", "read"]),

        (docs_relationship_map, "docs_relationship_map",
         "Relationship-Map generieren: Mermaid-Diagramm zeigt Vererbung, Nutzung, Imports "
         "(focus_dirs, focus_classes, max_nodes, format_type)",
         ["docs", "read"]),

        (docs_export_docmap, "docs_export_docmap",
         "Komplette DocMap exportieren als HTML oder Markdown "
         "(output_path, format_type, focus_dirs, title)",
         ["docs", "write"]),
    ])
    return tools


def _build_manifest_tools(app):
    """Build manifest management tools."""
    tools = []
    manifest_root = tb_root_dir

    async def manifest_show(section: str = None, as_json: bool = False) -> str:
        """Show manifest content, optionally a specific section."""
        try:
            from toolboxv2.utils.manifest import ManifestLoader
            loader = ManifestLoader(manifest_root)
            if not loader.exists():
                return "No tb-manifest.yaml found. Use manifest_apply or `tb manifest init`."

            if as_json or section:
                manifest = loader.load()
                if section:
                    section_data = getattr(manifest, section, None)
                    if section_data is None:
                        return f"Unknown section: {section}. Available: app, database, workers, nginx, auth, paths, isaa, features"
                    return json.dumps(section_data.model_dump(), indent=2, default=str, ensure_ascii=False)
                return json.dumps(manifest.model_dump(), indent=2, default=str, ensure_ascii=False)
            else:
                with open(loader.manifest_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            return f"Manifest show error: {e}"

    async def manifest_get(key: str) -> str:
        """Get a single manifest value by dotted key (e.g. 'database.mode')."""
        try:
            import yaml
            from toolboxv2.utils.manifest import ManifestLoader
            loader = ManifestLoader(manifest_root)
            if not loader.exists():
                return "No tb-manifest.yaml found."
            with open(loader.manifest_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            node = data
            for k in key.split("."):
                if not isinstance(node, dict) or k not in node:
                    return f"Key not found: {key}"
                node = node[k]
            return f"{key} = {node!r}"
        except Exception as e:
            return f"Manifest get error: {e}"

    async def manifest_set(key: str, value: str) -> str:
        """Set a manifest value (auto-syncs to .env if mapping exists)."""
        try:
            import yaml
            from toolboxv2.utils.manifest import ManifestLoader
            loader = ManifestLoader(manifest_root)
            if not loader.exists():
                return "No tb-manifest.yaml found."

            with open(loader.manifest_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Coerce value
            def _coerce(v):
                if v.lower() in ("true", "false"):
                    return v.lower() == "true"
                try:
                    return int(v)
                except ValueError:
                    pass
                try:
                    return float(v)
                except ValueError:
                    pass
                return v

            keys = key.split(".")
            node = data
            for k in keys[:-1]:
                if k not in node or not isinstance(node[k], dict):
                    node[k] = {}
                node = node[k]

            old_val = node.get(keys[-1], "<unset>")
            new_val = _coerce(value)
            node[keys[-1]] = new_val

            with open(loader.manifest_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            result = f"{key}: {old_val!r} -> {new_val!r}"

            # Sync .env
            _MANIFEST_TO_ENV = {
                "app.profile": "TB_PROFILE",
                "app.profiling": "PROFILING",
                "database.mode": "TB_DB_MODE",
                "app.debug": "TB_DEBUG",
                "app.log_level": "TOOLBOX_LOGGING_LEVEL",
                "app.environment": "TB_ENVIRONMENT",
                "auth.jwt.secret": "TB_JWT_SECRET",
                "auth.session.cookie_secret": "TB_COOKIE_SECRET",
                "database.minio.endpoint": "MINIO_ENDPOINT",
                "database.minio.access_key": "MINIO_ACCESS_KEY",
                "database.minio.secret_key": "MINIO_SECRET_KEY",
                "database.redis.url": "DB_CONNECTION_URI",
                "nginx.server_name": "TB_NGINX_SERVER_NAME",
                "paths.data_dir": "TB_DATA_DIR",
                "isaa.self_agent.max_iteration": "DEFAULT_MAX_ITERATIONS",
                "isaa.self_agent.fast_model": "FASTMODEL",
                "isaa.self_agent.complex_model": "COMPLEXMODEL",
            }
            env_key = _MANIFEST_TO_ENV.get(key)
            if env_key:
                env_path = manifest_root.parent / ".env"
                lines, found = [], False
                if env_path.exists():
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        if line.startswith(f"{env_key}="):
                            lines.append(f"{env_key}={value}")
                            found = True
                        else:
                            lines.append(line)
                if not found:
                    lines.append(f"{env_key}={value}")
                env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                result += f" | .env synced: {env_key}={value}"

            return result
        except Exception as e:
            return f"Manifest set error: {e}"

    async def manifest_validate() -> str:
        """Validate manifest schema and semantics."""
        try:
            from toolboxv2.utils.manifest import ManifestLoader
            loader = ManifestLoader(manifest_root)
            if not loader.exists():
                return "No tb-manifest.yaml found."
            manifest = loader.load()
            is_valid, errors = loader.validate()
            if is_valid:
                return "Validation PASSED (schema + semantics)"
            return "Validation FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        except Exception as e:
            return f"Schema validation FAILED: {e}"

    async def manifest_apply(dry_run: bool = False) -> str:
        """Generate sub-configs from manifest."""
        try:
            from toolboxv2.utils.manifest import ManifestLoader, ConfigConverter
            loader = ManifestLoader(manifest_root)
            if not loader.exists():
                return "No tb-manifest.yaml found."
            manifest = loader.load()
            converter = ConfigConverter(manifest, manifest_root)
            if dry_run:
                return "Dry run - would generate:\n" + "\n".join([
                    f"  - {manifest_root / '.config.yaml'}",
                    f"  - {manifest_root / 'bin' / 'config.toml'}",
                    f"  - {manifest_root / '.info' / 'services.json'}",
                ] + ([f"  - {manifest_root / 'nginx.conf'}"] if manifest.nginx.enabled else []))
            generated = converter.apply_all()
            return "Generated:\n" + "\n".join(f"  ✓ {p}" for p in generated)
        except Exception as e:
            return f"Manifest apply error: {e}"

    async def manifest_feature(action: str = "list", feature_name: str = None) -> str:
        """Manage features: list, enable, disable, files."""
        try:
            from toolboxv2.utils.system.feature_manager import FeatureManager
            features_dir = manifest_root / "features"
            fm = FeatureManager(features_dir=str(features_dir.name))

            if action == "list":
                if not fm.features:
                    return "No features found."
                lines = ["Feature Status:"]
                for fname, fspec in fm.features.items():
                    status = "ENABLED" if fspec.enabled else "DISABLED"
                    lines.append(f"  [{status}] {fname} v{fspec.version} - {fspec.description[:60]}")
                return "\n".join(lines)

            if not feature_name:
                return "feature_name required for enable/disable/files"

            feature_name = feature_name.lower()
            if feature_name not in fm.features:
                return f"Unknown feature: {feature_name}. Available: {', '.join(fm.features.keys())}"

            if action == "enable":
                feature = fm.features[feature_name]
                # Check requirements
                for req in (feature.requires or []):
                    if not fm.is_enabled(req):
                        return f"Required feature '{req}' is not enabled. Enable it first."
                success = fm.enable(feature_name)
                if success:
                    fm.save_to_manifest(feature_name)
                    return f"Feature '{feature_name}' enabled + manifest updated"
                return f"Failed to enable '{feature_name}'"

            elif action == "disable":
                success = fm.disable(feature_name)
                if success:
                    fm.save_to_manifest(feature_name)
                    return f"Feature '{feature_name}' disabled + manifest updated"
                return f"Failed to disable '{feature_name}'"

            elif action == "files":
                feature = fm.features[feature_name]
                parts = [f"{feature_name} v{feature.version}: {feature.description}"]
                if feature.files:
                    parts.append("Files: " + ", ".join(feature.files))
                if feature.imports:
                    parts.append("Imports: " + ", ".join(feature.imports))
                if feature.dependencies:
                    parts.append("Dependencies: " + ", ".join(feature.dependencies))
                if feature.requires:
                    parts.append("Requires: " + ", ".join(feature.requires))
                return "\n".join(parts)

            return f"Unknown action: {action}. Use list/enable/disable/files"
        except Exception as e:
            return f"Feature management error: {e}"

    tools.extend([
        (manifest_show, "manifest_show",
         "Manifest anzeigen (section, as_json). Sections: app, database, workers, nginx, auth, paths, isaa, features",
         ["manifest", "read"]),
        (manifest_get, "manifest_get",
         "Einzelnen Manifest-Wert lesen (dotted key: 'database.mode', 'isaa.self_agent.fast_model')",
         ["manifest", "read"]),
        (manifest_set, "manifest_set",
         "Manifest-Wert setzen und .env automatisch syncen (key, value)",
         ["manifest", "write"]),
        (manifest_validate, "manifest_validate",
         "Manifest Schema + Semantik validieren",
         ["manifest", "read"]),
        (manifest_apply, "manifest_apply",
         "Sub-Configs aus Manifest generieren (dry_run=True fuer Preview)",
         ["manifest", "write"]),
        (manifest_feature, "manifest_feature",
         "Features verwalten: action=list|enable|disable|files, feature_name=...",
         ["manifest", "write"]),
    ])
    return tools


def _build_toolbox_tools(isaa, app):
    """Build core toolbox tools for the admin agent."""
    tools = []

    async def toolbox_execute(module_name: str, function_name: str, kwargs: dict = None) -> str:
        """Execute any registered mod function."""
        kwargs = kwargs or {}
        try:
            from toolboxv2.utils.system.all_functions_enums import get_function_enum
            func_enum = get_function_enum(module_name, function_name)
            if func_enum is None:
                return f"Function not found: {module_name}.{function_name}"
            result = app.run_any(func_enum, **kwargs)
            if hasattr(result, "get") and callable(result.get):
                return json.dumps(result, ensure_ascii=False, default=str, indent=2)
            return str(result) if result is not None else "Done (None)"
        except Exception as e:
            return f"Error in {module_name}.{function_name}: {e}"

    async def toolbox_list_mods(module_name: str = None) -> str:
        """List loaded modules and their exported functions."""
        try:
            if module_name:
                funcs = app.get_all_functions_from(module_name)
                if funcs is None:
                    return f"Module not found: {module_name}"
                lines = [f"## {module_name}"]
                if isinstance(funcs, dict):
                    for f_name in sorted(funcs.keys()):
                        lines.append(f"  - {f_name}")
                else:
                    lines.append(str(funcs))
                return "\n".join(lines)
            else:
                mod_list = app.alive_mod_names if hasattr(app, "alive_mod_names") else []
                lines = ["## Loaded Modules"]
                for m in sorted(mod_list):
                    try:
                        funcs = app.get_all_functions_from(m)
                        count = len(funcs) if funcs and isinstance(funcs, dict) else 0
                        lines.append(f"  - {m} ({count} functions)")
                    except Exception:
                        lines.append(f"  - {m}")
                return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    async def toolbox_status() -> str:
        """Show current system status."""
        try:
            info = {
                "app_name": getattr(app, "app_name", "unknown"),
                "version": getattr(app, "version", "unknown"),
                "mods": list(app.alive_mod_names) if hasattr(app, "alive_mod_names") else [],
                "flows": list(app.flows.keys()) if hasattr(app, "flows") and app.flows else [],
                "data_dir": getattr(app, "data_dir", "unknown"),
                "debug": getattr(app, "debug", False),
            }
            return json.dumps(info, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            return f"Error: {e}"

    async def flow_manage(action: str = "list", name: str = None, content: str = None) -> str:
        """Manage flows: list, read, create, run."""
        try:
            if action == "list":
                flows = app.flows if hasattr(app, "flows") and app.flows else {}
                lines = ["## Available Flows"]
                for fname in sorted(flows.keys()):
                    lines.append(f"  - {fname}")
                return "\n".join(lines)
            elif action == "read" and name:
                flows_dir = Path(__file__).parent
                flow_file = flows_dir / f"{name}.py"
                if flow_file.exists():
                    return flow_file.read_text(encoding="utf-8")
                return f"Flow not found: {name}"
            elif action == "create" and name and content:
                target = Path("/global/flows") / f"{name}.py"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return f"Flow created: {target}"
            elif action == "run" and name:
                result = await app.run_flows(name)
                return f"Flow result: {result}"
            else:
                return "Usage: action=list|read|create|run, name=..., content=..."
        except Exception as e:
            return f"Error: {e}"

    async def cloudm_action(action: str = "list_users", **kwargs) -> str:
        """CloudM actions: list_users, list_folders, create_mod_template, create_flow_template."""
        try:
            cloudm = app.get_mod("CloudM")
            if cloudm is None:
                return "CloudM module not loaded."
            if action == "list_users":
                result = await cloudm.list_all_users()
                return json.dumps(result, ensure_ascii=False, default=str, indent=2)
            elif action == "list_folders":
                result = await cloudm.list_folders()
                return json.dumps(result, ensure_ascii=False, default=str, indent=2)
            else:
                return f"Unknown action: {action}"
        except Exception as e:
            return f"CloudM error: {e}"

    tools.extend([
        (toolbox_execute, "toolbox_execute",
         "Beliebige Mod-Funktion ausfuehren (module_name, function_name, kwargs)",
         ["toolbox", "execute"]),
        (toolbox_list_mods, "toolbox_list_mods",
         "Module und Funktionen anzeigen (module_name=None fuer alle)",
         ["toolbox", "read"]),
        (toolbox_status, "toolbox_status",
         "Systemstatus: mods, flows, config",
         ["toolbox", "read"]),
        (flow_manage, "flow_manage",
         "Flows verwalten: list/read/create/run",
         ["toolbox", "flow"]),
        (cloudm_action, "cloudm_action",
         "CloudM: list_users, list_folders, create_mod_template, create_flow_template",
         ["cloud", "read"]),
    ])
    return tools


def _build_style():
    return PTStyle.from_dict({
        "prompt": "#22d3ee bold",
        "": "#e2e8f0",
    })


# ---------------------------------------------------------------------------
# Streaming Output Helper
# ---------------------------------------------------------------------------

async def _print_stream(agent, text: str, session_id: str = "admin"):
    """Run agent with a_stream_verbose and print chunks live."""
    if False:
        res = await agent.a_run(
            query=text,
            session_id=session_id,
            human_online=True
        )

        print()
        print(res)
        return

    try:
        async for chunk in agent.a_stream_verbose(
            query=text,
            session_id=session_id,
            human_online=True,
            max_iterations=50
        ):
            if chunk:
                print(chunk, end="", flush=True)
        print()  # final newline
    except Exception as e:
        print(Style.RED(f"\nAgent error: {e}"))
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------------

async def run(app: App, args=None):
    """Main entry point for the toolbox_admin flow."""
    cls()
    print(Style.CYAN("╔═════════════════════════════════╗"))
    print(Style.CYAN("║    ToolBox v2.0 Admin Agent     ║"))
    print(Style.CYAN("║     · Full System Access ·      ║"))
    print(Style.CYAN("╚═════════════════════════════════╝"))
    print()

    # --- ISAA init ---
    isaa = app.get_mod("isaa")
    if isaa is None:
        print(Style.RED("ERROR: ISAA module not loaded!"))
        print("Start with: python -m toolboxv2 -m isaa -f toolbox_admin")
        return

    print(Style.YELLOW("Initializing ISAA..."))
    await isaa.init_isaa(name="tb_admin")

    print(Style.YELLOW("Building Admin Agent..."))
    builder = isaa.get_agent_builder(
        "tb_admin",
        add_base_tools=True,
        with_dangerous_shell=True,
    )

    builder.with_stream(True)

    # --- Register all tool groups ---
    for func, name, desc, cats in _build_toolbox_tools(isaa, app):
        builder.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})

    for func, name, desc, cats in _build_docs_tools(app):
        builder.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})

    for func, name, desc, cats in _build_manifest_tools(app):
        builder.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})

    # --- System prompt ---
    builder.config.system_message = SYSTEM_PROMPT

    await isaa.register_agent(builder)
    agent = await isaa.get_agent("tb_admin")
    agent.tool_manager.register_cli_tool("tb", executable="uv", executable_args=["run"],
                                    flags={"system_tool_by_name": True},
                                    cli_tool_executable="tb",
                                    category="system")
    print(Style.GREEN(f"Agent ready: {agent.amd.name}"))
    print(Style.GREEN(f"  Fast Model:    {agent.amd.fast_llm_model}"))
    print(Style.GREEN(f"  Complex Model: {agent.amd.complex_llm_model}"))
    print()

    # --- Init DocsSystem in background ---
    print(Style.YELLOW("Initializing DocsSystem index..."))
    try:
        from toolboxv2.utils.extras.mkdocs import create_docs_system
        with Spinner("DocsSystem index init",symbols="a"):
            ds = create_docs_system(
                project_root=str(tb_root_dir.parent),
                include_dirs=["toolboxv2", "flows", "mods", "utils", "docs"],
            )
        init_result = await ds.initialize(show_tqdm=True)
        print(Style.GREEN(
            f"  Docs: {init_result.get('sections', 0)} sections, "
            f"{init_result.get('elements', 0)} code elements "
            f"({init_result.get('status', '?')}, {init_result.get('time_ms', 0):.0f}ms)"
        ))
    except Exception as e:
        print(Style.YELLOW(f"  Docs init skipped: {e}"))

    print()
    print(Style.GREY("Commands: /status  /mods  /flows  /manifest  /docs  /help  exit"))
    print(Style.GREY("Everything else goes to the agent (streaming)."))
    print()

    # --- REPL ---
    history_path = Path(app.data_dir) / ".toolbox_admin_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    completions = [
        "/status", "/mods", "/flows", "/manifest", "/docs", "/help",
        "exit", "quit", "/quit",
    ]
    session = PromptSession(
        history=FileHistory(str(history_path)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=FuzzyCompleter(WordCompleter(completions, ignore_case=True)),
        style=_build_style(),
    )

    while True:
        try:
            user_input = await session.prompt_async("tb-admin> ")
        except (EOFError, KeyboardInterrupt):
            print(Style.YELLOW("\nBye!"))
            break

        text = user_input.strip()
        if not text:
            continue

        if text.lower() in ("exit", "quit", "/quit"):
            print(Style.YELLOW("Bye!"))
            break

        # --- Quick commands (bypass agent) ---
        if text == "/status":
            for func, name, *_ in _build_toolbox_tools(isaa, app):
                if name == "toolbox_status":
                    print(await func())
                    break
            continue

        if text == "/mods":
            for func, name, *_ in _build_toolbox_tools(isaa, app):
                if name == "toolbox_list_mods":
                    print(await func())
                    break
            continue

        if text == "/flows":
            for func, name, *_ in _build_toolbox_tools(isaa, app):
                if name == "flow_manage":
                    print(await func(action="list"))
                    break
            continue

        if text == "/manifest":
            for func, name, *_ in _build_manifest_tools(app):
                if name == "manifest_show":
                    print(await func())
                    break
            continue

        if text == "/docs":
            try:
                from toolboxv2.utils.extras.mkdocs import create_docs_system
                ds = create_docs_system(project_root=str(tb_root_dir.parent))
                info = await ds.initialize()
                print(json.dumps(info, indent=2, default=str))
            except Exception as e:
                print(Style.RED(f"Docs error: {e}"))
            continue

        if text == "/help":
            print(Style.CYAN("ToolBox Admin v2.0"))
            print("  /status    - System status")
            print("  /mods      - Loaded modules")
            print("  /flows     - Available flows")
            print("  /manifest  - Show manifest")
            print("  /docs      - Docs index info")
            print("  /help      - This help")
            print("  exit       - Quit")
            print()
            print("Agent examples (streaming):")
            print('  "Zeige alle CloudM Funktionen"')
            print('  "Setze database.mode auf sqlite"')
            print('  "Welche Features sind aktiviert?"')
            print('  "Erstelle einen neuen Flow fuer X"')
            print('  "Suche in den Docs nach LiveSync"')
            print('  "Lookup die AgentBuilder Klasse mit Code"')
            continue

        # --- Agent streaming ---
        await _print_stream(agent, text)

    # --- Cleanup ---
    print(Style.YELLOW("Saving agent state..."))
    try:
        await isaa.on_exit()
    except Exception:
        pass
    print(Style.GREEN("Auf Wiedersehen!"))


if __name__ == "__main__":
    asyncio.run(run(get_app()))
