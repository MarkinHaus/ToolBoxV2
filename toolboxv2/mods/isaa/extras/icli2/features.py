"""Features und Skills der icli 2.0: SimpleFeatureManager samt den sieben kleinen Ladern.

Wird in IsaaBackend eingemischt. Alle toolboxv2-Importe passieren erst in der
jeweiligen Methode bzw. im Lader (das Paket ist beim Start der UI nicht zwingend geladen).

Vorlage (1:1 uebernommen, nur ohne print_status-Ausgabe, die das TUI stoeren wuerde):
  flows/isaa/icli.py  SimpleFeatureManager, load_desktop_auto_feature, load_web_auto_feature,
                      load_isis_feature, load_full_web_auto_feature, load_coder_toolkit,
                      load_chain_toolkit, load_execute
Skills: SkillsManager (base/Agent/skills.py) ueber agent._get_execution_engine().skills_manager.
Die grossen Lader (docs, autodoc, autotest, autofix) sind bewusst nicht enthalten.
"""
from __future__ import annotations

import asyncio
import sys


class SimpleFeatureManager:
    def __init__(self):
        self.agent_ref = None
        self.features = {}

    def list_features(self):
        return list(self.features.keys())

    async def enable(self, feature):
        if feature in self.features:
            self.features[feature]["is_enabled"] = True
            if self.features[feature]["activation_f"]:
                res = self.features[feature]["activation_f"](self.agent_ref)
                if asyncio.iscoroutine(res):
                    await res

    def is_enabled(self, feature):
        return self.features.get(feature, {"is_enabled": False})["is_enabled"]

    def disable(self, feature):
        if feature in self.features:
            self.features[feature]["is_enabled"] = False
            if self.features[feature]["deactivation_f"]:
                self.features[feature]["deactivation_f"](self.agent_ref)

    def add_feature(self, feature, activation_f=None, deactivation_f=None):
        self.features[feature] = {"is_enabled": False, "activation_f": activation_f,
                                  "deactivation_f": deactivation_f}

    def set_agent(self, agent):
        self.agent_ref = agent


# ============================ Lader (Importe lazy) ============================
def load_desktop_auto_feature(fm):
    def enable(agent):
        from toolboxv2.mods.isaa.extras.destop_auto import register_enhanced_tools
        kit, tools = register_enhanced_tools()
        agent.add_tools(tools)

    def disable(agent):
        from toolboxv2.mods.isaa.extras.destop_auto import register_enhanced_tools
        kit, tools = register_enhanced_tools()
        agent.remove_tools(tools)
    fm.add_feature("desktop_auto", activation_f=enable, deactivation_f=disable)


def load_web_auto_feature(fm):
    from toolboxv2.mods.isaa.extras.web_helper.tooklit import get_minimal_tools
    tools_set = [None]
    toolkit, tools_set[0] = get_minimal_tools(out_of_process=True, headless=True)

    async def enable(agent):
        await toolkit.start_browser()
        agent.add_tools(tools_set[0])

    async def disable(agent):
        await toolkit.stop_browser()
        agent.remove_tools(tools_set[0])
    fm.add_feature("mini_web_auto", activation_f=enable, deactivation_f=disable)


def load_isis_feature(fm):
    from toolboxv2.mods.isaa.extras.toolkit.isis_toolkit import IsisToolkit
    toolkit = IsisToolkit(out_of_process=True, headless=True)
    tools_set = [None]

    async def enable(agent):
        await toolkit.start()
        tools_set[0] = toolkit.get_tools()
        agent.add_tools(tools_set[0])

    async def disable(agent):
        await toolkit.stop()
        agent.remove_tools(tools_set[0])
    fm.add_feature("isis_auto", activation_f=enable, deactivation_f=disable)


def load_full_web_auto_feature(fm):
    from toolboxv2.mods.isaa.extras.web_helper.tooklit import WebAgentToolkit
    toolkit = WebAgentToolkit(out_of_process=True, headless=True)
    tools_set = [None]

    async def enable(agent):
        from toolboxv2 import tb_root_dir
        if sys.platform == "win32":
            asyncio.set_event_loop(asyncio.ProactorEventLoop())
        await toolkit.start_browser()
        tools_set[0] = toolkit.get_tools()
        agent.add_tools(tools_set[0])
        session = await agent.session_manager.get_or_create(agent.active_session)
        session.vfs.add_system_file(str(tb_root_dir.parent / "docs" / "mods" / "isaa" / "web_toolkit.md"),
                                    "web_toolkit.md", open_state=True)

    async def disable(agent):
        await toolkit.stop_browser()
        agent.remove_tools(tools_set[0])
        session = await agent.session_manager.get_or_create(agent.active_session)
        session.vfs.remove_system_file("web_toolkit.md")
    fm.add_feature("full_web_auto", activation_f=enable, deactivation_f=disable)


def load_coder_toolkit(fm):
    from toolboxv2.mods.isaa.CodingAgent.coder_toolset import coder_register_flow_tools
    pool = [None]

    def enable(agent):
        from toolboxv2 import init_cwd
        _pool, tools = coder_register_flow_tools(agent, str(init_cwd))
        pool[0] = _pool
        agent.add_tools(tools)

    def disable(agent):
        from toolboxv2 import init_cwd
        _pool, tools = coder_register_flow_tools(agent, init_cwd)
        agent.remove_tools(tools)
    fm.add_feature("coder", activation_f=enable, deactivation_f=disable)


def load_chain_toolkit(fm):
    from toolboxv2.mods.isaa.base.chain.chain_tools import create_chain_tools
    tools_set = [None]
    agent_registry, coder_registry, format_registry = {}, {}, {}

    def enable(agent):
        tools_set[0] = create_chain_tools(agent, agent_registry=agent_registry,
                                          coder_registry=coder_registry, format_registry=format_registry)
        agent.add_tools(tools_set[0])

    def disable(agent):
        agent.remove_tools(tools_set[0])
    fm.add_feature("chain", activation_f=enable, deactivation_f=disable)


def load_execute(fm):
    from toolboxv2.mods.isaa.base.Agent.executors import register_code_exec_tools
    tools_set = [None]

    def enable(agent):
        tools_set[0] = register_code_exec_tools(agent)

    def disable(agent):
        for t in (tools_set[0] or []):
            agent.remove_tool(t["name"] if isinstance(t, dict) else t)
        tools_set[0] = None
    fm.add_feature("execute", activation_f=enable, deactivation_f=disable)


def load_docs_feature(fm):
    """
    Documentation Feature - Integrates mkdocs-based docs system.

    When active in toolboxv2 directory: Uses tb_root_dir.parent/docs as doc dir.
    When active in another directory: Prompts user to select docs directory.
    """
    docs_system = [None]  # Mutable container for the docs system instance
    docs_tools = [None]   # Mutable container for the tool list
    _TOOL_HEALTH_EXTENSIONS = {
        "docs_reader": {
            "live_test_inputs": [
                {
                    "query": "_probe_health_check_query",
                    "max_results": 1
                }
            ],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": (
                    "Das Tool muss ein Dictionary mit 'count' und 'time_ms' (oder 'sections') zurückgeben."
                )
            },
            "cleanup_func": None
        },
        "docs_writer": {
            "live_test_inputs": [
                {
                    # Negative Testing: Wir prüfen das Routing und Error-Handling,
                    # ohne echte Dateien zu schreiben.
                    "action": "_probe_invalid_action"
                }
            ],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": (
                    "Da die Action ungültig ist, muss ein Dict mit dem Key 'error' "
                    "und einer entsprechenden Fehlermeldung zurückgegeben werden."
                )
            },
            "cleanup_func": None
        },
        "docs_lookup": {
            "live_test_inputs": [
                {
                    "name": "_probe_SystemTest",
                    "max_results": 1
                }
            ],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Muss ein Dict mit 'results' (Liste) und 'count' zurückgeben."
            },
            "cleanup_func": None
        },
        "docs_sync": {
            "live_test_inputs": [{}],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Muss ein Dict mit 'changes_detected' und 'files_updated' zurückgeben."
            },
            "cleanup_func": None
        },
        "docs_init": {
            "live_test_inputs": [
                {
                    # Wichtig: force_rebuild=False für einen schnellen, sicheren Test!
                    "force_rebuild": False,
                    "show_tqdm": False
                }
            ],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Muss ein Dict mit 'status' (z.B. 'loaded') und 'time_ms' zurückgeben."
            },
            "cleanup_func": None
        },
        "get_task_context": {
            "live_test_inputs": [
                {
                    "files": [],
                    "intent": "_probe_health_check_intent"
                }
            ],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Muss ein Dict mit 'result' und 'meta' zurückgeben."
            },
            "cleanup_func": None
        },
        "docs_inventory": {
            "live_test_inputs": [
                {
                    "focus_dirs": "",
                    "max_classes_per_file": 3,
                    "max_methods_per_class": 2,
                    "include_functions": True,
                }
            ],
            "result_contract": {
                "expected_type": str,
                "semantic_check_hint": "Muss Markdown-String mit '# Project Inventory' zurueckgeben.",
            },
            "cleanup_func": None,
        },
        "docs_relationship_map": {
            "live_test_inputs": [
                {
                    "focus_dirs": "",
                    "focus_classes": "",
                    "max_nodes": 20,
                    "format_type": "markdown",
                }
            ],
            "result_contract": {
                "expected_type": str,
                "semantic_check_hint": "Muss Markdown-String mit mermaid code block zurueckgeben.",
            },
            "cleanup_func": None,
        },
        "docs_export_docmap": {
            "live_test_inputs": [
                {
                    "output_path": "",
                    "format_type": "md",
                    "focus_dirs": "",
                    "title": "_probe_health_check",
                }
            ],
            "result_contract": {
                "expected_type": str,
                "semantic_check_hint": "Muss DocMap Markdown content zurueckgeben.",
            },
            "cleanup_func": None,
        },
    }
    async def enable(agent):
        """Enable documentation system and add tools to agent."""
        try:
            # Determine docs directory based on current working directory
            current_dir = init_cwd
            project_root = tb_root_dir

            # Check if we're in toolboxv2 or its parent
            if current_dir == project_root or current_dir == project_root.parent or project_root in current_dir.parents:
                docs_dir = project_root.parent / "docs"
                c_print(HTML(f"<style fg='ansicyan'>📚 Auto-detected docs dir:</style> <style fg='ansigreen'>{docs_dir}</style>"))
            else:
                # Prompt user for docs directory
                docs_input = tb_root_dir
                docs_dir = Path(docs_input).expanduser().resolve()

            if not docs_dir.exists():
                c_print(HTML(f"<style fg='ansired'>⚠️ Docs directory not found: {docs_dir}</style>"))
                c_print(HTML("<style fg='ansiyellow'>Creating minimal docs structure...</style>"))
                docs_dir.mkdir(parents=True, exist_ok=True)

            # Create docs system instance
            system = DocsSystem(
                project_root=project_root,
                docs_root=docs_dir,
                include_dirs=["toolboxv2", "flows", "mods", "utils", "docs", "src"]
            )

            _get_docs = lambda :system

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

            # Initialize (load existing index or build new one)
            result = await system.initialize()
            c_print(HTML(f"<style fg='ansigreen'>✓ Docs initialized:</style> {result['sections']} sections, {result['elements']} elements"))

            docs_system[0] = system

            # Prepare tool list with proper descriptions
            tools = [
                {
                    "tool_func": system.read,
                    "name": "docs_reader",
                    "description": "Durchsucht die Dokumentation nach relevanten Abschnitten basierend auf einer Suchanfrage. Nützlich um schnell Informationen aus der Dokumentation zu finden ohne manuell zu suchen. Gibt strukturierte Ergebnisse mit Relevanz-Scores zurück.",
                    "category": ["docs", "read", "search"],
                    "flags": {}
                },
                {
                    "tool_func": system.write,
                    "name": "docs_writer",
                    "description": "Schreibt neue Dokumentations-Dateien oder aktualisiert existierende. Unterstützt Markdown-Formatierung und speichert im docs-Verzeichnis. Erstellt automatisch die nötige Verzeichnisstruktur wenn nötig. Ideal um Ergebnisse und Erkenntnisse zu dokumentieren.",
                    "category": ["docs", "write"],
                    "flags": {}
                },
                {
                    "tool_func": system.lookup_code,
                    "name": "docs_lookup",
                    "description": "Sucht nach Code-Elementen (Klassen, Funktionen, Module, etc.) im gesamten Codebase. Gibt Definitionen, Signaturen und Docstrings zurück. Nützlich um Implementierungsdetails schnell zu verstehen ohne durch Dateien navigieren zu müssen.",
                    "category": ["docs", "code", "search"],
                    "flags": {}
                },
                {
                    "tool_func": system.sync,
                    "name": "docs_sync",
                    "description": "Synchronisiert Änderungen aus der VFS zurück zum Dateisystem. Stellt sicher dass Änderungen persistent gespeichert werden. Sollte regelmäßig aufgerufen werden nachdem Änderungen vorgenommen wurden.",
                    "category": ["docs", "sync"],
                    "flags": {}
                },
                {
                    "tool_func": system.initialize,
                    "name": "docs_init",
                    "description": "Baut den Dokumentations-Index neu auf. Nützlich wenn neue Dateien hinzugefügt wurden oder der Index veraltet ist. Indiziert Markdown-, Python- und JavaScript/TypeScript-Dateien aus den konfigurierten include_dirs.",
                    "category": ["docs", "index"],
                    "flags": {}
                },
                {
                    "tool_func": system.get_task_context,
                    "name": "get_task_context",
                    "description": "Generiert Kontext-Informationen für Aufgaben wie relevante Dateien, Klassen und Dokumentation basierend auf einer Aufgabenbeschreibung. Hilft dem Agent die Aufgabe besser zu verstehen und die richtigen Ressourcen zu finden.",
                    "category": ["docs", "context"],
                    "flags": {}
                },
                {
                    "tool_func": docs_inventory,
                    "name": "docs_inventory",
                    "description": (
                        "Generiert ein kompaktes Projekt-Inventar: welche Dateien, Klassen und "
                        "Funktionen existieren im Projekt. Gruppiert nach Dateien mit den wichtigsten "
                        "Klassen und deren meistgenutzten Methoden. Beantwortet: 'Was ist alles da?'"
                    ),
                    "category": ["docs", "read"],
                    "flags": {},
                },
                {
                    "tool_func": docs_relationship_map,
                    "name": "docs_relationship_map",
                    "description": (
                        "Generiert eine Beziehungskarte als Mermaid-Diagramm. Zeigt Vererbung, "
                        "Komposition und Cross-File-Referenzen zwischen Klassen. "
                        "Beantwortet: 'Wie spielt Komponente A mit B zusammen?'"
                    ),
                    "category": ["docs", "read"],
                    "flags": {},
                },
                {
                    "tool_func": docs_export_docmap,
                    "name": "docs_export_docmap",
                    "description": (
                        "Exportiert eine vollstaendige DocMap (Inventar + Beziehungen) als "
                        "standalone HTML (mit Neo-Brutalism Paper Style, Mermaid-Rendering, Dark Mode) "
                        "oder als LLM-optimiertes Markdown. Kombiniert Inventar und Relationship Map."
                    ),
                    "category": ["docs", "write"],
                    "flags": {},
                },
            ]

            for tool in tools:
                if tool["name"] in _TOOL_HEALTH_EXTENSIONS:
                    tool.update(_TOOL_HEALTH_EXTENSIONS[tool["name"]])

            docs_tools[0] = tools
            agent.add_tools(tools)

            print_status("Documentation feature enabled.", "success")

        except Exception as e:
            c_print(HTML(f"<style fg='ansired'>✗ Failed to enable docs: {e}</style>"))
            import traceback
            c_print(traceback.format_exc())

    def disable(agent):
        """Disable documentation system."""
        try:
            if docs_tools[0]:
                agent.remove_tools(docs_tools[0])
            docs_system[0] = None
            docs_tools[0] = None
            print_status("Documentation feature disabled.", "success")
        except Exception as e:
            c_print(HTML(f"<style fg='ansired'>✗ Failed to disable docs: {e}</style>"))

    fm.add_feature("docs", activation_f=enable, deactivation_f=disable)

LOADERS = (load_docs_feature, load_desktop_auto_feature, load_web_auto_feature, load_full_web_auto_feature,
           load_coder_toolkit, load_chain_toolkit, load_execute, load_isis_feature)


class FeatureMixin:
    def _features_init(self) -> None:
        """Idempotent: Manager anlegen, Lader registrieren (jeder einzeln abgesichert)."""
        if getattr(self, "_fm", None) is not None:
            return
        self._fm = SimpleFeatureManager()
        self._skill_context: dict[str, str] = {}
        for loader in LOADERS:
            try:
                loader(self._fm)
            except Exception as e:  # fehlender Import darf die anderen Lader nicht stoppen
                try:
                    self.app.logger.warning(f"Feature-Lader {loader.__name__} uebersprungen: {e}")
                except Exception:
                    pass

    def _features_active(self) -> list[str]:
        """Namen der aktiven Features (Kopfzeile) - wirft nie."""
        try:
            self._features_init()
            return [f for f in self._fm.list_features() if self._fm.is_enabled(f)]
        except Exception:
            return []

    async def _rows_features(self) -> tuple[str, list[dict]]:
        self._features_init()
        rows = []
        for name in self._fm.list_features():
            on = self._fm.is_enabled(name)
            rows.append({"value": name, "label": ("✓ " if on else "· ") + name, "hint": "an" if on else "aus"})
        return "Features", rows

    async def _do_feature(self, name: str) -> dict:
        self._features_init()
        if name not in self._fm.features:
            raise ValueError(f"Unbekanntes Feature: {name}")
        self._fm.set_agent(await self._agent())
        if self._fm.is_enabled(name):
            # fm.disable() wuerde async-Deaktivierer nicht abwarten -> hier selbst behandeln
            feat = self._fm.features[name]
            feat["is_enabled"] = False
            if feat["deactivation_f"]:
                res = feat["deactivation_f"](self._fm.agent_ref)
                if asyncio.iscoroutine(res):
                    await res
            state = "aus"
        else:
            try:
                await self._fm.enable(name)
            except Exception:
                self._fm.features[name]["is_enabled"] = False
                raise
            state = "an"
        return {"ready": await self._ready(), "text": f"Feature {name} {state}"}

    # ---------------------------------------------------------------- Skills
    async def _skills_manager(self):
        return (await self._agent())._get_execution_engine().skills_manager

    async def _rows_skills(self) -> tuple[str, list[dict]]:
        self._features_init()
        sm = await self._skills_manager()
        rows = []
        for sid, sk in sm.skills.items():
            on = sid in self._skill_context
            trig = ", ".join(getattr(sk, "triggers", None) or [])
            desc = (getattr(sk, "description", "") or "").strip().splitlines()
            hint = (trig or (desc[0] if desc else ""))[:60]
            rows.append({"value": sid, "label": ("✓ " if on else "· ") + (getattr(sk, "name", None) or sid),
                         "hint": hint})
        return f"Skills: {self.agent}", rows

    async def _do_skill(self, name: str) -> dict:
        self._features_init()
        sm = await self._skills_manager()
        sk = sm.skills.get(name)
        if sk is None:
            raise ValueError(f"Skill nicht gefunden: {name}")
        label = getattr(sk, "name", None) or name
        if name in self._skill_context:
            self._skill_context.pop(name, None)
            try:
                sk.confidence = 0.0
            except Exception:
                pass
            return {"text": f"Skill {label} aus"}
        self._skill_context[name] = getattr(sk, "instruction", "") or ""
        try:
            sk.confidence = 0.95
        except Exception:
            pass
        return {"text": f"Skill {label} an - Agent hat den vollen Kontext"}

    def _skill_wrap(self, query: str) -> str:
        """Aktive Skills der Anfrage voranstellen (Format wie icli.py ~14505) - wirft nie."""
        try:
            ctx = getattr(self, "_skill_context", None)
            if not ctx:
                return query
            blocks = "\n\n".join(f"[ACTIVE SKILL: {sid}]\n{instr}" for sid, instr in ctx.items())
            return f"{blocks}\n\n---\nUser request:\n{query}"
        except Exception:
            return query
