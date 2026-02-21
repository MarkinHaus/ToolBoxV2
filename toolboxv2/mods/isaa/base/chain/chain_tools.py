"""
Chain Tools Module - Agent-Driven Chain Management

3 Tools für den FlowAgent:
- create_validate_chain: Chain erstellen & validieren (DSL → gespeichert, default: unsafe)
- run_chain: Chain ausführen (nur wenn accepted)
- list_auto_get_fitting: Chains auflisten & passende für Task finden

DSL Format:
    Sequential:     step >> step >> step
    Parallel:       (step + step)
    Error Handler:  (step | fallback)
    Conditional:    IS(key==value) >> true_step % false_step

Pre-Blocks (vor der Chain-Definition):
    model:Name(field: type, ...)  → Inline Pydantic-Klasse für CF()
    def:name(data) -> expression  → Inline Custom-Funktion (Python-Ausdruck)

Step Types:
    tool:name(arg="val")          → Registriertes Agent-Tool
    @agent("focus instruction")   → Agent-Aufruf mit Chain-Kontext
    @coder("focus instruction")   → CoderAgent (gleicher Name = gleicher Kontext)
    def:name(data) -> expression  → Inline Custom-Funktion (Python-Ausdruck)
    CF(ModelName) - "key"         → Pydantic Format + Extraktion
    CF(ModelName) - "key[n]"      → Format + Parallel-Extraktion

Sicherheit:
    Chains sind IMMER 'unsafe' bis manuell akzeptiert (einmalig pro Chain).
    Custom-Funktionen (def:) werden in restricted exec ausgeführt.

Author: ToolBoxV2
"""

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

from toolboxv2 import get_app
from toolboxv2.mods.isaa.base.Agent.chain import (
    CF,
    IS,
    Chain,
    ChainBase,
    ConditionalChain,
    ErrorHandlingChain,
    Function,
    ParallelChain,
)


# =============================================================================
# CHAIN STORAGE
# =============================================================================


@dataclass
class StoredChain:
    """Eine gespeicherte Chain mit Metadaten."""

    id: str
    name: str
    dsl: str
    description: str
    accepted: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_run: str | None = None
    run_count: int = 0
    tags: list[str] = field(default_factory=list)
    # Validierung
    is_valid: bool = False
    validation_errors: list[str] = field(default_factory=list)
    # Welche custom functions sind definiert
    custom_functions: list[str] = field(default_factory=list)
    # Welche tools werden referenziert
    referenced_tools: list[str] = field(default_factory=list)
    # Ob agents verwendet werden
    uses_agents: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "dsl": self.dsl,
            "description": self.description,
            "accepted": self.accepted,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "run_count": self.run_count,
            "tags": self.tags,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "custom_functions": self.custom_functions,
            "referenced_tools": self.referenced_tools,
            "uses_agents": self.uses_agents,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StoredChain":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ChainStore:
    """Persistenter Speicher für Chains."""

    def __init__(self, store_path: str | Path):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._index_file = self.store_path / "_index.json"
        self._chains: dict[str, StoredChain] = {}
        self._load_index()

    def _load_index(self):
        if self._index_file.exists():
            try:
                with open(self._index_file, encoding="utf-8") as f:
                    data = json.load(f)
                self._chains = {
                    k: StoredChain.from_dict(v) for k, v in data.items()
                }
            except Exception:
                self._chains = {}

    def _save_index(self):
        with open(self._index_file, "w", encoding="utf-8") as f:
            json.dump(
                {k: v.to_dict() for k, v in self._chains.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def save(self, chain: StoredChain):
        self._chains[chain.id] = chain
        self._save_index()

    def get(self, chain_id: str) -> StoredChain | None:
        return self._chains.get(chain_id)

    def get_by_name(self, name: str) -> StoredChain | None:
        for c in self._chains.values():
            if c.name == name:
                return c
        return None

    def list_all(self) -> list[StoredChain]:
        return list(self._chains.values())

    def delete(self, chain_id: str) -> bool:
        if chain_id in self._chains:
            del self._chains[chain_id]
            self._save_index()
            return True
        return False

    def accept(self, chain_id: str) -> bool:
        """Chain als sicher markieren (einmalig)."""
        chain = self._chains.get(chain_id)
        if chain:
            chain.accepted = True
            self._save_index()
            return True
        return False


# =============================================================================
# DSL PARSER
# =============================================================================


class ChainParseError(Exception):
    """Fehler beim Parsen einer Chain-DSL."""
    pass


@dataclass
class ParsedStep:
    """Ein geparster Step aus der DSL."""
    type: str  # "tool", "agent", "custom_func", "format", "condition"
    name: str
    args: dict[str, str] = field(default_factory=dict)
    raw: str = ""
    # Für custom functions
    func_body: str | None = None
    func_param: str | None = None
    # Für CF
    format_class_name: str | None = None
    extract_key: str | None = None
    is_parallel_extract: bool = False
    # Für agent
    focus_instruction: str | None = None
    # Für IS
    condition_key: str | None = None
    condition_value: str | None = None


class ChainDSLParser:
    """
    Parser für die Chain-DSL.

    Wandelt DSL-String in validierte ParsedSteps um.
    Baut KEINE Chain-Objekte direkt — das macht der ChainBuilder.
    """

    # Regex für Step-Typen
    RE_TOOL = re.compile(
        r'^tool:(\w+)\(([^)]*)\)$'
    )
    RE_AGENT = re.compile(
        r'^@(\w+)\("([^"]*)"\)$'
    )
    RE_AGENT_SIMPLE = re.compile(
        r'^@(\w+)$'
    )
    RE_CUSTOM_FUNC = re.compile(
        r'^def:(\w+)\((\w+)\)\s*->\s*(.+)$'
    )
    RE_FORMAT = re.compile(
        r'^CF\((\w+)\)(?:\s*-\s*"([^"]*)")?$'
    )
    RE_CONDITION = re.compile(
        r'^IS\("?(\w+)"?\s*==\s*"?([^")\s]+)"?\)$'
    )

    def __init__(self):
        self.custom_functions: dict[str, ParsedStep] = {}
        self.inline_models: dict[str, type] = {}  # model: Pre-Block → Pydantic Klassen
        self.errors: list[str] = []

    def parse(self, dsl: str) -> tuple[list[Any], list[str]]:
        """
        Parse DSL-String in eine strukturierte Repräsentation.

        Returns:
            (structure, errors) - structure ist eine nested list/dict Repräsentation
        """
        self.errors = []
        self.custom_functions = {}
        self.inline_models = {}

        lines = dsl.strip().split("\n")

        # Phase 1: Pre-Blocks extrahieren (müssen VOR der Chain definiert sein)
        chain_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("def:"):
                self._parse_custom_func_definition(line)
            elif line.startswith("model:"):
                self._parse_inline_model(line)
            else:
                chain_lines.append(line)

        if not chain_lines:
            self.errors.append("Keine Chain-Definition gefunden")
            return [], self.errors

        # Phase 2: Chain-Zeilen zusammenführen
        chain_str = " ".join(chain_lines)

        # Phase 3: Rekursiv parsen
        try:
            structure = self._parse_expression(chain_str.strip())
        except ChainParseError as e:
            self.errors.append(str(e))
            structure = []

        return structure, self.errors

    def _parse_custom_func_definition(self, line: str):
        """Parse eine def:name(param) -> body Zeile."""
        match = self.RE_CUSTOM_FUNC.match(line.strip())
        if match:
            name, param, body = match.groups()
            step = ParsedStep(
                type="custom_func_def",
                name=name,
                func_param=param,
                func_body=body.strip(),
                raw=line,
            )
            self.custom_functions[name] = step
        else:
            self.errors.append(f"Ungültige Custom-Function Syntax: {line}")

    # Regex: model:Name(field1: type1, field2: type2)
    RE_MODEL_DEF = re.compile(
        r'^model:(\w+)\(([^)]+)\)$'
    )

    # Erlaubte Typ-Mappings für inline Pydantic Models
    _TYPE_MAP = {
        "str": str, "string": str,
        "int": int, "integer": int,
        "float": float, "number": float,
        "bool": bool, "boolean": bool,
        "list": list, "dict": dict,
        "list[str]": list[str], "list[int]": list[int],
        "list[float]": list[float], "list[dict]": list[dict],
        "any": Any,
    }

    def _parse_inline_model(self, line: str):
        """
        Parse eine model:Name(field: type, ...) Zeile → Pydantic BaseModel.

        Syntax:
            model:SearchResult(title: str, url: str, score: float)
            model:TaskList(tasks: list[str], priority: int)
        """
        match = self.RE_MODEL_DEF.match(line.strip())
        if not match:
            self.errors.append(f"Ungültige Model-Syntax: {line}")
            return

        model_name, fields_str = match.groups()

        # Fields parsen: "field1: type1, field2: type2"
        annotations = {}
        for field_def in fields_str.split(","):
            field_def = field_def.strip()
            if ":" not in field_def:
                self.errors.append(
                    f"Model '{model_name}': Feld '{field_def}' braucht Typ (name: type)"
                )
                continue

            fname, ftype_str = field_def.split(":", 1)
            fname = fname.strip()
            ftype_str = ftype_str.strip().lower()

            ftype = self._TYPE_MAP.get(ftype_str)
            if ftype is None:
                # Optional[str] etc. → fallback zu str
                self.errors.append(
                    f"Model '{model_name}': Unbekannter Typ '{ftype_str}' für '{fname}', "
                    f"verwende str. Erlaubt: {', '.join(sorted(self._TYPE_MAP.keys()))}"
                )
                ftype = str

            annotations[fname] = ftype

        if not annotations:
            self.errors.append(f"Model '{model_name}': Keine gültigen Felder")
            return

        # Dynamisch Pydantic BaseModel erstellen
        from pydantic import BaseModel
        model_cls = type(model_name, (BaseModel,), {"__annotations__": annotations})
        self.inline_models[model_name] = model_cls

    def _parse_expression(self, expr: str) -> Any:
        """
        Rekursiver Expression-Parser.

        Precedence (niedrigste zuerst):
            1. % (Conditional false branch)
            2. >> (Sequential)
            3. | (Error handling)
            4. + (Parallel)
            5. Atom (einzelner Step oder Klammer-Gruppe)
        """
        expr = expr.strip()
        if not expr:
            raise ChainParseError("Leerer Ausdruck")

        # % - Conditional False Branch (niedrigste Precedence)
        parts = self._split_top_level(expr, "%")
        if len(parts) == 2:
            return {
                "type": "conditional_branches",
                "true": self._parse_expression(parts[0]),
                "false": self._parse_expression(parts[1]),
            }

        # >> - Sequential
        parts = self._split_top_level(expr, ">>")
        if len(parts) > 1:
            return {
                "type": "sequential",
                "steps": [self._parse_expression(p) for p in parts],
            }

        # | - Error Handling
        parts = self._split_top_level(expr, "|")
        if len(parts) == 2:
            return {
                "type": "error_handling",
                "primary": self._parse_expression(parts[0]),
                "fallback": self._parse_expression(parts[1]),
            }

        # + - Parallel
        parts = self._split_top_level(expr, "+")
        if len(parts) > 1:
            return {
                "type": "parallel",
                "branches": [self._parse_expression(p) for p in parts],
            }

        # Klammer-Gruppe
        if expr.startswith("(") and expr.endswith(")"):
            return self._parse_expression(expr[1:-1])

        # Atom - einzelner Step
        return self._parse_atom(expr)

    def _split_top_level(self, expr: str, operator: str) -> list[str]:
        """Split an operator, aber nur auf Top-Level (nicht in Klammern oder Strings)."""
        parts = []
        current = []
        depth = 0
        in_string = False
        string_char = None
        i = 0

        while i < len(expr):
            char = expr[i]

            if in_string:
                current.append(char)
                if char == string_char and (i == 0 or expr[i - 1] != "\\"):
                    in_string = False
            elif char in ('"', "'"):
                in_string = True
                string_char = char
                current.append(char)
            elif char == "(":
                depth += 1
                current.append(char)
            elif char == ")":
                depth -= 1
                current.append(char)
            elif depth == 0 and expr[i:i + len(operator)] == operator:
                parts.append("".join(current).strip())
                current = []
                i += len(operator)
                continue
            else:
                current.append(char)

            i += 1

        remaining = "".join(current).strip()
        if remaining:
            parts.append(remaining)

        return parts if len(parts) > 1 else [expr]

    def _parse_atom(self, atom: str) -> ParsedStep:
        """Parse einen einzelnen Step."""
        atom = atom.strip()

        # tool:name(args)
        match = self.RE_TOOL.match(atom)
        if match:
            name, args_str = match.groups()
            args = self._parse_tool_args(args_str)
            return ParsedStep(type="tool", name=name, args=args, raw=atom)

        # @agent("focus") oder @agent
        match = self.RE_AGENT.match(atom)
        if match:
            name, focus = match.groups()
            return ParsedStep(
                type="agent", name=name, focus_instruction=focus, raw=atom
            )
        match = self.RE_AGENT_SIMPLE.match(atom)
        if match:
            return ParsedStep(type="agent", name=match.group(1), raw=atom)

        # def:name — Referenz auf zuvor definierte Custom Function
        if atom.startswith("def:"):
            func_name = atom[4:].strip()
            if func_name in self.custom_functions:
                defn = self.custom_functions[func_name]
                return ParsedStep(
                    type="custom_func",
                    name=func_name,
                    func_param=defn.func_param,
                    func_body=defn.func_body,
                    raw=atom,
                )
            # Vielleicht inline definiert: def:name(param) -> body
            match = self.RE_CUSTOM_FUNC.match(atom)
            if match:
                name, param, body = match.groups()
                return ParsedStep(
                    type="custom_func",
                    name=name,
                    func_param=param,
                    func_body=body.strip(),
                    raw=atom,
                )
            self.errors.append(f"Unbekannte Custom-Function: {func_name}")
            return ParsedStep(type="error", name=func_name, raw=atom)

        # CF(ModelName) - "key"
        match = self.RE_FORMAT.match(atom)
        if match:
            class_name, key = match.groups()
            is_parallel = False
            if key and "[n]" in key:
                key = key.replace("[n]", "")
                is_parallel = True
            return ParsedStep(
                type="format",
                name=f"CF({class_name})",
                format_class_name=class_name,
                extract_key=key,
                is_parallel_extract=is_parallel,
                raw=atom,
            )

        # IS(key==value)
        match = self.RE_CONDITION.match(atom)
        if match:
            key, value = match.groups()
            return ParsedStep(
                type="condition",
                name=f"IS({key}=={value})",
                condition_key=key,
                condition_value=value,
                raw=atom,
            )

        self.errors.append(f"Unbekannter Step: '{atom}'")
        return ParsedStep(type="error", name=atom, raw=atom)

    @staticmethod
    def _parse_tool_args(args_str: str) -> dict[str, str]:
        """Parse Tool-Argumente: arg1="val", arg2="{prev}" """
        args = {}
        if not args_str.strip():
            return args

        # Einfacher Key=Value Parser
        pattern = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
        for match in pattern.finditer(args_str):
            args[match.group(1)] = match.group(2)

        return args


# =============================================================================
# DSL VALIDATOR
# =============================================================================


class ChainValidator:
    """Validiert eine geparste Chain gegen die verfügbaren Tools/Agents."""

    def __init__(
        self,
        available_tools: set[str] | None = None,
        available_agents: set[str] | None = None,
        available_format_classes: set[str] | None = None,
    ):
        self.available_tools = available_tools or set()
        self.available_agents = available_agents or set()
        self.available_format_classes = available_format_classes or set()

    def validate(self, structure: Any, dsl: str) -> tuple[bool, list[str], dict]:
        """
        Validiert die geparste Struktur.

        Returns:
            (is_valid, errors, metadata)
        """
        errors = []
        metadata = {
            "referenced_tools": [],
            "custom_functions": [],
            "uses_agents": False,
            "step_count": 0,
        }

        self._validate_recursive(structure, errors, metadata)

        # Check DSL Syntax (basic bracket matching)
        open_count = dsl.count("(")
        close_count = dsl.count(")")
        if open_count != close_count:
            errors.append(
                f"Klammer-Mismatch: {open_count} öffnende vs {close_count} schließende"
            )

        return len(errors) == 0, errors, metadata

    def _validate_recursive(self, node: Any, errors: list, metadata: dict):
        if node is None:
            return

        if isinstance(node, ParsedStep):
            metadata["step_count"] += 1
            self._validate_step(node, errors, metadata)

        elif isinstance(node, dict):
            node_type = node.get("type")

            if node_type == "sequential":
                for step in node.get("steps", []):
                    self._validate_recursive(step, errors, metadata)

            elif node_type == "parallel":
                branches = node.get("branches", [])
                if len(branches) < 2:
                    errors.append("Parallel braucht mindestens 2 Branches")
                for branch in branches:
                    self._validate_recursive(branch, errors, metadata)

            elif node_type == "error_handling":
                self._validate_recursive(node.get("primary"), errors, metadata)
                self._validate_recursive(node.get("fallback"), errors, metadata)

            elif node_type == "conditional_branches":
                self._validate_recursive(node.get("true"), errors, metadata)
                self._validate_recursive(node.get("false"), errors, metadata)

    def _validate_step(self, step: ParsedStep, errors: list, metadata: dict):
        if step.type == "error":
            errors.append(f"Parse-Fehler bei: '{step.raw}'")
            return

        if step.type == "tool":
            metadata["referenced_tools"].append(step.name)
            if self.available_tools and step.name not in self.available_tools:
                errors.append(
                    f"Tool '{step.name}' nicht verfügbar. "
                    f"Verfügbar: {', '.join(sorted(self.available_tools)[:10])}"
                )

        elif step.type == "agent":
            metadata["uses_agents"] = True
            if self.available_agents and step.name not in self.available_agents:
                errors.append(
                    f"Agent '{step.name}' nicht verfügbar. "
                    f"Verfügbar: {', '.join(sorted(self.available_agents))}"
                )

        elif step.type == "custom_func":
            metadata["custom_functions"].append(step.name)
            # Validiere, dass die Expression sicher ist (keine imports, exec, etc.)
            if step.func_body:
                unsafe = self._check_unsafe_expression(step.func_body)
                if unsafe:
                    errors.append(
                        f"Custom Function '{step.name}' enthält unsicheren Code: {unsafe}"
                    )

        elif step.type == "format":
            if (
                self.available_format_classes
                and step.format_class_name
                and step.format_class_name not in self.available_format_classes
            ):
                errors.append(
                    f"Format-Klasse '{step.format_class_name}' nicht registriert"
                )

    @staticmethod
    def _check_unsafe_expression(body: str) -> str | None:
        """Prüft ob ein Python-Ausdruck gefährlich ist."""
        dangerous = [
            "import ", "__import__", "exec(", "eval(", "compile(",
            "open(", "os.", "sys.", "subprocess", "shutil",
            "__class__", "__subclasses__", "__globals__",
            "breakpoint", "getattr(", "setattr(", "delattr(",
        ]
        body_lower = body.lower()
        for pattern in dangerous:
            if pattern.lower() in body_lower:
                return f"'{pattern}' nicht erlaubt"
        return None


# =============================================================================
# CHAIN BUILDER — Structure → Chain Objects
# =============================================================================


class ChainContextWrapper:
    """
    Wrapper der einen FlowAgent in Chain-Kontext ausführt.

    Injiziert System-Kontext der dem Agent mitteilt:
    - Er läuft in einer Chain
    - Was sein aktueller Step / Fokus ist
    - Er soll NUR diesen Step bearbeiten
    """

    CHAIN_CONTEXT_TEMPLATE = (
        "[CHAIN-EXECUTION MODE]\n"
        "Du wirst gerade als Step in einer automatisierten Chain ausgeführt.\n"
        "Chain: {chain_name}\n"
        "Dein Step: {step_index}\n"
        "Deine Aufgabe: {focus}\n\n"
        "WICHTIG:\n"
        "- Fokussiere dich NUR auf diese eine Aufgabe\n"
        "- Der vorherige Step hat dir folgende Daten übergeben (als Input)\n"
        "- Gib dein Ergebnis als final_answer zurück\n"
        "- Stelle KEINE Rückfragen — arbeite mit dem was du hast\n"
        "- Halte dich kurz und präzise"
    )

    def __init__(
        self,
        agent: Any,  # FlowAgent
        chain_name: str,
        step_index: int,
        focus_instruction: str = "",
    ):
        self.agent = agent
        self.chain_name = chain_name
        self.step_index = step_index
        self.focus = focus_instruction

    async def a_run(self, data: Any, **kwargs) -> str:
        """Führt den Agent mit Chain-Kontext aus."""
        context = self.CHAIN_CONTEXT_TEMPLATE.format(
            chain_name=self.chain_name,
            step_index=self.step_index,
            focus=self.focus or "Verarbeite den Input und liefere ein Ergebnis",
        )

        query = f"{context}\n\n--- INPUT DATA ---\n{data}"

        result = await self.agent.a_run(
            query=query,
            session_id=kwargs.get("session_id", f"chain_{self.chain_name}"),
            max_iterations=kwargs.get("max_iterations", 10),
        )
        return result

    async def a_format_class(self, pydantic_model, prompt, **kwargs):
        """Delegate format_class an den Agent."""
        return await self.agent.a_format_class(pydantic_model, prompt, **kwargs)


class CoderContextWrapper:
    """
    Wrapper der einen CoderAgent in Chain-Kontext ausführt.

    Gleicher Name in der DSL = gleiche CoderAgent-Instanz.
    → Behält: GitWorktree, ExecutionMemory, State (plan/done/current_file)
    → Jeder Step baut auf dem vorherigen auf (gleicher Branch, gleiche History)

    Nutzt CoderAgent.execute(task) → CoderResult(success, message, files, messages, ...)
    """

    CODER_CHAIN_PREFIX = (
        "[CHAIN-EXECUTION MODE — CODER STEP]\n"
        "Du bist Step {step_index} in Chain '{chain_name}'.\n"
        "Fokus: {focus}\n"
        "Bearbeite NUR diesen einen Auftrag. Keine Rückfragen.\n"
        "Wenn fertig, beende mit [DONE].\n"
        "---\n"
    )

    def __init__(
        self,
        coder: Any,  # CoderAgent Instanz
        chain_name: str,
        step_index: int,
        focus_instruction: str = "",
    ):
        self.coder = coder
        self.chain_name = chain_name
        self.step_index = step_index
        self.focus = focus_instruction

    async def a_run(self, data: Any, **kwargs) -> str:
        """Führt den CoderAgent mit Chain-Kontext aus.

        Der Coder behält seinen vollen Kontext (Worktree, Memory, State)
        über mehrere Chain-Steps hinweg wenn der gleiche Name verwendet wird.
        """
        prefix = self.CODER_CHAIN_PREFIX.format(
            chain_name=self.chain_name,
            step_index=self.step_index,
            focus=self.focus or "Bearbeite den Code-Task",
        )

        # Chain-Step in Coder-State injizieren (ohne bestehenden State zu zerstören)
        if hasattr(self.coder, "state"):
            self.coder.state["plan"] = [self.focus or "Chain-Step ausführen"]

        task = f"{prefix}{data}"
        result = await self.coder.execute(task)

        # CoderResult → String (CoderResult(success, message, [file_paths], messages, ...))
        if hasattr(result, "success"):
            # files sind 3. positional arg — Feld-Name kann variieren
            files = (
                getattr(result, "files_changed", None)
                or getattr(result, "files", None)
                or []
            )
            files_str = ", ".join(files) if files else "keine"
            tokens = getattr(result, "tokens_used", 0)

            if result.success:
                return (
                    f"[Coder OK] Step {self.step_index} abgeschlossen\n"
                    f"Dateien: {files_str}\n"
                    f"Tokens: {tokens}\n"
                    f"{result.message}"
                )
            else:
                return (
                    f"[Coder FEHLER] Step {self.step_index}\n"
                    f"Dateien (teilweise): {files_str}\n"
                    f"{result.message}"
                )

        return str(result)

    async def a_format_class(self, pydantic_model, prompt, **kwargs):
        """Delegate format_class an den inneren FlowAgent des Coders."""
        if hasattr(self.coder, "agent"):
            return await self.coder.agent.a_format_class(pydantic_model, prompt, **kwargs)
        raise NotImplementedError("CoderAgent hat keine a_format_class Methode")


class SafeCustomFunction:
    """
    Sicher ausführbare Custom-Function.

    Evaluiert einen Python-Ausdruck in einer eingeschränkten Umgebung.
    """

    # Erlaubte Built-ins für Custom Functions
    SAFE_BUILTINS = {
        "len": len, "str": str, "int": int, "float": float, "bool": bool,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "sorted": sorted, "reversed": reversed, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter, "range": range,
        "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
        "any": any, "all": all, "isinstance": isinstance, "type": type,
        "None": None, "True": True, "False": False,
        "json": json,  # json Modul erlauben
    }

    def __init__(self, name: str, param: str, body: str):
        self.name = name
        self.param = param
        self.body = body

    async def __call__(self, data: Any) -> Any:
        """Führt die Custom Function sicher aus."""
        safe_globals = {"__builtins__": self.SAFE_BUILTINS}
        safe_locals = {self.param: data}

        try:
            result = eval(self.body, safe_globals, safe_locals)  # noqa: S307
            return result
        except Exception as e:
            return f"[CustomFunc Error] {self.name}: {e}"


class ChainBuilder:
    """
    Baut Chain-Objekte aus geparster DSL-Struktur.

    Benötigt:
    - agent_registry: Dict[name → FlowAgent] für @agent Steps
    - coder_registry: Dict[name → CoderAgent] für @coder Steps (gleicher Name = gleiche Instanz)
    - tool_executor: Callable für tool: Steps (nutzt FlowAgent.arun_function)
    - format_registry: Dict[name → Pydantic BaseModel] für CF() Steps
    - inline_models: Dict[name → type] aus model: Pre-Blocks der DSL
    """

    def __init__(
        self,
        agent_registry: dict[str, Any] | None = None,
        coder_registry: dict[str, Any] | None = None,
        tool_executor: Callable | None = None,
        format_registry: dict[str, type] | None = None,
        inline_models: dict[str, type] | None = None,
        chain_name: str = "unnamed",
    ):
        self.agent_registry = agent_registry or {}
        self.coder_registry = coder_registry or {}
        self.tool_executor = tool_executor
        self.format_registry = format_registry or {}
        self.inline_models = inline_models or {}
        self.chain_name = chain_name
        self._step_counter = 0

    def build(self, structure: Any) -> ChainBase:
        """Baut eine Chain aus der geparsten Struktur."""
        self._step_counter = 0
        return self._build_recursive(structure)

    def _build_recursive(self, node: Any) -> ChainBase:
        if isinstance(node, ParsedStep):
            return self._build_step(node)

        if isinstance(node, dict):
            node_type = node.get("type")

            if node_type == "sequential":
                steps = node.get("steps", [])
                if not steps:
                    raise ChainParseError("Leere Sequential Chain")
                built = [self._build_recursive(s) for s in steps]
                # Manuell Chain bauen
                chain = Chain()
                chain.tasks = built
                return chain

            elif node_type == "parallel":
                branches = [
                    self._build_recursive(b) for b in node.get("branches", [])
                ]
                return ParallelChain(branches)

            elif node_type == "error_handling":
                primary = self._build_recursive(node["primary"])
                fallback = self._build_recursive(node["fallback"])
                return ErrorHandlingChain(primary, fallback)

            elif node_type == "conditional_branches":
                # Die true-Branch enthält IS() >> step, false-Branch ist der % Teil
                true_node = node["true"]
                false_node = node["false"]
                # Wir brauchen die Condition aus dem true_node
                # Wenn true_node sequential ist und erstes Element IS() ist
                condition = None
                true_chain = None
                if isinstance(true_node, dict) and true_node.get("type") == "sequential":
                    steps = true_node.get("steps", [])
                    if steps and isinstance(steps[0], ParsedStep) and steps[0].type == "condition":
                        cond_step = steps[0]
                        condition = IS(cond_step.condition_key, cond_step.condition_value)
                        remaining = steps[1:]
                        if len(remaining) == 1:
                            true_chain = self._build_recursive(remaining[0])
                        else:
                            true_chain = self._build_recursive({
                                "type": "sequential", "steps": remaining
                            })
                elif isinstance(true_node, ParsedStep) and true_node.type == "condition":
                    condition = IS(true_node.condition_key, true_node.condition_value)
                    true_chain = self._build_recursive(false_node)
                    return true_chain  # Kein echtes Conditional ohne true-Branch

                if condition is None:
                    # Fallback: Erster Step wird Condition
                    true_chain = self._build_recursive(true_node)
                    false_chain = self._build_recursive(false_node)
                    return ConditionalChain(None, true_chain, false_chain)

                false_chain = self._build_recursive(false_node)
                return ConditionalChain(condition, true_chain, false_chain)

        raise ChainParseError(f"Unbekannter Node-Typ: {type(node)}: {node}")

    def _build_step(self, step: ParsedStep) -> ChainBase:
        self._step_counter += 1

        if step.type == "tool":
            return self._build_tool_step(step)

        elif step.type == "agent":
            return self._build_agent_step(step)

        elif step.type == "custom_func":
            return self._build_custom_func_step(step)

        elif step.type == "format":
            return self._build_format_step(step)

        elif step.type == "condition":
            # Standalone condition — wird von conditional_branches verarbeitet
            # Hier als Marker zurückgeben
            return Function(lambda data, _key=step.condition_key, _val=step.condition_value: data)

        raise ChainParseError(f"Unbekannter Step-Typ: {step.type}")

    def _build_tool_step(self, step: ParsedStep) -> Function:
        """Baut einen Tool-Step der FlowAgent.arun_function nutzt."""
        tool_name = step.name
        static_args = step.args.copy()
        executor = self.tool_executor

        async def _execute_tool(data: Any) -> Any:
            if executor is None:
                return f"[Error] Kein Tool-Executor verfügbar für '{tool_name}'"

            # Args auflösen: {prev} → data, {input} → original input
            resolved_args = {}
            for k, v in static_args.items():
                if v == "{prev}":
                    resolved_args[k] = str(data)
                elif v == "{input}":
                    resolved_args[k] = str(data)
                else:
                    resolved_args[k] = v

            # Wenn keine args definiert, data als erstes Argument übergeben
            if not resolved_args:
                # Versuche Tool mit 'query' oder 'input' oder 'data' arg
                resolved_args = {"query": str(data)}

            try:
                result = await executor(tool_name, **resolved_args)
                return result
            except TypeError:
                # Fallback: versuche ohne query
                try:
                    resolved_args.pop("query", None)
                    resolved_args["data"] = str(data)
                    result = await executor(tool_name, **resolved_args)
                    return result
                except Exception as e:
                    return f"[Tool Error] {tool_name}: {e}"
            except Exception as e:
                return f"[Tool Error] {tool_name}: {e}"

        func = Function(_execute_tool)
        func.func_name = f"tool:{tool_name}"
        return func

    def _build_agent_step(self, step: ParsedStep) -> ChainContextWrapper:
        """Baut einen Agent- oder Coder-Step mit Chain-Kontext."""
        agent_name = step.name

        # Coder-Registry hat Priorität bei Namenskollision
        if agent_name in self.coder_registry:
            coder = self.coder_registry[agent_name]
            return CoderContextWrapper(
                coder=coder,
                chain_name=self.chain_name,
                step_index=self._step_counter,
                focus_instruction=step.focus_instruction or "",
            )

        if agent_name not in self.agent_registry:
            # Fallback: Gebe Fehlermeldung als Funktion zurück
            async def _missing_agent(data):
                return f"[Error] Agent/Coder '{agent_name}' nicht gefunden"
            func = Function(_missing_agent)
            func.func_name = f"@{agent_name}[MISSING]"
            return func

        agent = self.agent_registry[agent_name]
        return ChainContextWrapper(
            agent=agent,
            chain_name=self.chain_name,
            step_index=self._step_counter,
            focus_instruction=step.focus_instruction or "",
        )

    def _build_custom_func_step(self, step: ParsedStep) -> Function:
        """Baut einen Custom-Function Step."""
        safe_func = SafeCustomFunction(
            name=step.name,
            param=step.func_param or "data",
            body=step.func_body or "data",
        )

        func = Function(safe_func)
        func.func_name = f"def:{step.name}"
        return func

    def _build_format_step(self, step: ParsedStep) -> CF:
        """Baut einen CF Format Step. Prüft: format_registry → inline_models → Placeholder."""
        class_name = step.format_class_name

        if class_name in self.format_registry:
            format_class = self.format_registry[class_name]
        elif class_name in self.inline_models:
            # Aus model: Pre-Block in der DSL definiert
            format_class = self.inline_models[class_name]
        else:
            # Kein Model gefunden → Placeholder mit result: str
            from pydantic import BaseModel
            format_class = type(class_name, (BaseModel,), {"__annotations__": {"result": str}})

        cf = CF(format_class)
        if step.extract_key:
            if step.is_parallel_extract:
                cf.extract_key = step.extract_key
                cf.is_parallel_extraction = True
            else:
                cf.extract_key = step.extract_key

        return cf


# =============================================================================
# CHAIN TOOLS — Die 3 Agent-Tools
# =============================================================================


def generate_chain_id(name: str, dsl: str) -> str:
    """Generiert eine kurze, deterministische Chain-ID."""
    raw = f"{name}:{dsl}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def create_chain_tools(
    agent: Any,
    chain_store: ChainStore | None = None,
    agent_registry: dict[str, Any] | None = None,
    coder_registry: dict[str, Any] | None = None,
    format_registry: dict[str, type] | None = None,
) -> list[
    dict[str, Callable[[str, str, str, str], Coroutine[Any, Any, str]] | str | list[str] | dict[str, bool]] | dict[
        str, Callable[[str, str, bool], Coroutine[Any, Any, str]] | str | list[str] | dict[str, bool]] | dict[
        str, Callable[[str, bool], Coroutine[Any, Any, str]] | str | list[str] | dict[str, bool]]]:
    """
    Erstellt die 3 Chain-Management Tools für den FlowAgent.

    Args:
        agent: Der FlowAgent der die Tools bekommt
        chain_store: Persistenter Chain-Speicher
        agent_registry: Verfügbare Agents {name → FlowAgent}
        coder_registry: Verfügbare Coder {name → CoderAgent} (gleicher Name = gleiche Instanz)
        format_registry: Verfügbare Pydantic Models {name → type}

    Returns:
        Liste von Tool-Definitionen für den ToolManager
    """
    agent_registry = agent_registry or {}
    coder_registry = coder_registry or {}
    format_registry = format_registry or {}
    if chain_store is None:
        chain_store = ChainStore(store_path= str(get_app().data_dir) + '/Agents/chains/' + agent.amd.name)

    # ─── Tool 1: create_validate_chain ───

    async def create_validate_chain(
        name: str,
        dsl: str,
        description: str = "",
        tags: str = "",
    ) -> str:
        """Erstellt und validiert eine neue Chain aus DSL."""

        # Parse
        parser = ChainDSLParser()
        structure, parse_errors = parser.parse(dsl)

        # Validate
        available_tools = set()
        if hasattr(agent, "tool_manager"):
            available_tools = set(agent.tool_manager._registry.keys())

        # Format-Klassen: statische Registry + inline model: Definitionen aus DSL
        all_format_classes = set(format_registry.keys()) | set(parser.inline_models.keys())
        # Agents: FlowAgents + Coder (beide via @name erreichbar)
        all_agents = set(agent_registry.keys()) | set(coder_registry.keys())

        validator = ChainValidator(
            available_tools=available_tools,
            available_agents=all_agents,
            available_format_classes=all_format_classes,
        )
        is_valid, val_errors, metadata = validator.validate(structure, dsl)

        all_errors = parse_errors + val_errors

        # Generiere ID
        chain_id = generate_chain_id(name, dsl)

        # Speichere
        stored = StoredChain(
            id=chain_id,
            name=name,
            dsl=dsl.strip(),
            description=description,
            tags=[t.strip() for t in tags.split(",") if t.strip()] if tags else [],
            accepted=False,  # IMMER unsafe
            is_valid=is_valid,
            validation_errors=all_errors,
            custom_functions=metadata.get("custom_functions", []),
            referenced_tools=metadata.get("referenced_tools", []),
            uses_agents=metadata.get("uses_agents", False),
        )
        chain_store.save(stored)

        # Response
        status = "VALID" if is_valid else "INVALID"
        lines = [
            f"Chain '{name}' erstellt [{status}]",
            f"ID: {chain_id}",
            f"Steps: {metadata.get('step_count', '?')}",
            f"Status: UNSAFE (muss manuell akzeptiert werden)",
        ]

        if metadata.get("custom_functions"):
            lines.append(f"Custom Functions: {', '.join(metadata['custom_functions'])}")
        if metadata.get("referenced_tools"):
            lines.append(f"Tools: {', '.join(metadata['referenced_tools'])}")
        if metadata.get("uses_agents"):
            lines.append("Agents: Ja (werden mit Chain-Kontext ausgeführt)")

        if all_errors:
            lines.append(f"\nFehler ({len(all_errors)}):")
            for err in all_errors:
                lines.append(f"  - {err}")

        return "\n".join(lines)

    # ─── Tool 2: run_chain ───

    async def run_chain(
        name_or_id: str,
        input_data: str = "",
        accept: bool = False,
    ) -> str:
        """Führt eine gespeicherte Chain aus."""

        # Chain finden
        stored = chain_store.get(name_or_id) or chain_store.get_by_name(name_or_id)
        if not stored:
            return f"Chain '{name_or_id}' nicht gefunden. Nutze list_auto_get_fitting um verfügbare Chains zu sehen."

        # Validierung prüfen
        if not stored.is_valid:
            return (
                f"Chain '{stored.name}' ist INVALID und kann nicht ausgeführt werden.\n"
                f"Fehler: {'; '.join(stored.validation_errors)}\n"
                f"Erstelle die Chain neu mit create_validate_chain."
            )

        # Akzeptanz prüfen/setzen
        if not stored.accepted:
            if accept:
                chain_store.accept(stored.id)
                stored.accepted = True
            else:
                return (
                    f"Chain '{stored.name}' ist UNSAFE und wurde noch nicht akzeptiert.\n"
                    f"DSL:\n{stored.dsl}\n\n"
                    f"Um sie auszuführen, rufe run_chain erneut mit accept=true auf.\n"
                    f"Dies muss nur einmal gemacht werden."
                )

        # Chain bauen
        try:
            parser = ChainDSLParser()
            structure, errors = parser.parse(stored.dsl)
            if errors:
                return f"Parse-Fehler beim Bauen: {'; '.join(errors)}"

            builder = ChainBuilder(
                agent_registry=agent_registry,
                coder_registry=coder_registry,
                tool_executor=agent.arun_function if hasattr(agent, "arun_function") else None,
                format_registry=format_registry,
                inline_models=parser.inline_models,
                chain_name=stored.name,
            )
            chain = builder.build(structure)

        except Exception as e:
            return f"Fehler beim Bauen der Chain: {e}"

        # Ausführen
        try:
            start = time.time()
            result = await chain.a_run(input_data, session_id=f"chain_{stored.name}")
            elapsed = time.time() - start

            # Stats updaten
            stored.run_count += 1
            stored.last_run = datetime.now().isoformat()
            chain_store.save(stored)

            return (
                f"Chain '{stored.name}' ausgeführt ({elapsed:.1f}s)\n"
                f"Run #{stored.run_count}\n\n"
                f"--- RESULT ---\n{result}"
            )
        except Exception as e:
            return f"Fehler bei Chain-Ausführung: {e}"

    # ─── Tool 3: list_auto_get_fitting ───

    async def list_auto_get_fitting(
        task_description: str = "",
        show_all: bool = False,
    ) -> str:
        """Listet Chains auf und findet passende für einen Task."""

        all_chains = chain_store.list_all()

        if not all_chains:
            return "Keine Chains gespeichert. Nutze create_validate_chain um eine neue Chain zu erstellen."

        lines = [f"Gespeicherte Chains ({len(all_chains)}):\n"]

        for c in all_chains:
            status_parts = []
            if c.accepted:
                status_parts.append("ACCEPTED")
            else:
                status_parts.append("UNSAFE")
            if c.is_valid:
                status_parts.append("VALID")
            else:
                status_parts.append("INVALID")
            status = " | ".join(status_parts)

            lines.append(f"  [{c.id}] {c.name}")
            lines.append(f"    Status: {status}")
            if c.description:
                lines.append(f"    Beschreibung: {c.description}")
            if c.tags:
                lines.append(f"    Tags: {', '.join(c.tags)}")
            lines.append(f"    Runs: {c.run_count}")
            if show_all:
                lines.append(f"    DSL: {c.dsl}")
            lines.append("")

        # Auto-Matching wenn task_description gegeben
        if task_description:
            task_lower = task_description.lower()
            task_words = set(task_lower.split())

            scored: list[tuple[float, StoredChain]] = []
            for c in all_chains:
                score = 0.0
                # Name-Match
                if any(w in c.name.lower() for w in task_words):
                    score += 3.0
                # Description-Match
                if c.description and any(w in c.description.lower() for w in task_words):
                    score += 2.0
                # Tag-Match
                for tag in c.tags:
                    if any(w in tag.lower() for w in task_words):
                        score += 1.5
                # Tool-Match
                for tool in c.referenced_tools:
                    if any(w in tool.lower() for w in task_words):
                        score += 1.0
                # Bonus für accepted + valid
                if c.accepted and c.is_valid:
                    score += 0.5
                # Bonus für Nutzungshäufigkeit
                score += min(c.run_count * 0.1, 1.0)

                if score > 0:
                    scored.append((score, c))

            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                lines.append(f"--- PASSENDE CHAINS für '{task_description}' ---\n")
                for score, c in scored[:5]:
                    ready = "READY" if (c.accepted and c.is_valid) else "NEEDS ACCEPT" if c.is_valid else "INVALID"
                    lines.append(f"  [{ready}] {c.name} (Score: {score:.1f})")
                    lines.append(f"    ID: {c.id}")
                    lines.append(f"    DSL: {c.dsl[:120]}{'...' if len(c.dsl) > 120 else ''}")
                    lines.append("")

                best = scored[0][1]
                if best.accepted and best.is_valid:
                    lines.append(
                        f"Empfehlung: run_chain(name_or_id=\"{best.id}\", input_data=\"...\")"
                    )
                elif best.is_valid:
                    lines.append(
                        f"Empfehlung: run_chain(name_or_id=\"{best.id}\", accept=true, input_data=\"...\")"
                    )
            else:
                lines.append(f"Keine passenden Chains für '{task_description}' gefunden.")
                lines.append("Erstelle eine neue mit create_validate_chain.")

        return "\n".join(lines)

    return [
    {
        "tool_func": create_validate_chain,
        "name": "create_validate_chain",
        "description": (
            "Erstellt und validiert eine automatisierte Chain aus DSL-Notation. "
            "Chains werden als UNSAFE gespeichert, bis sie manuell akzeptiert werden. "
            "Diese Funktion parst die DSL, prüft Syntax und speichert die Definition.\n\n"
            "### DSL DOKUMENTATION & REGELN\n"
            "PRE-BLOCKS (vor der Chain, eine pro Zeile):\n"
            "- model:Name(field: type, ...)  → Pydantic-Klasse inline definieren\n"
            "- def:name(x) -> ausdruck       → Custom Python-Funktion\n\n"
            "DSL FORMAT (FLOW):\n"
            "- Sequential:  step >> step >> step\n"
            "- Parallel:    (step + step)\n"
            "- Fallback:    (step | fallback_step)\n"
            "- Conditional: IS(key==value) >> true_step % false_step\n\n"
            "STEP TYPEN:\n"
            "- tool:name(arg=\"val\")        → Registriertes Tool ausführen\n"
            "- @agent_name(\"fokus\")        → Agent mit Fokus-Instruktion (Chain-Kontext)\n"
            "- @coder_name(\"fokus\")        → CoderAgent (gleicher Name = gleicher Kontext/Worktree)\n"
            "- def:name                     → Referenz auf oben definierte Custom-Function\n"
            "- CF(ModelName) - \"key\"        → Pydantic-Format + Feld-Extraktion\n"
            "- CF(ModelName) - \"key[n]\"     → Format + Auto-Parallelisierung über Liste\n\n"
            "VARIABLEN in Tool-Args:\n"
            "- {prev}  → Output des vorherigen Steps\n"
            "- {input} → Originaler Chain-Input\n\n"
            "BEISPIEL:\n"
            "  model:SearchResult(title: str, url: str)\n"
            "  def:clean(text) -> text.strip()\n"
            "  tool:web_search(query=\"{prev}\") >> CF(SearchResult) - \"title\" >> def:clean\n\n"
            "### PARAMETER INFORMATIONEN\n"
            "- name (str, required): Eindeutiger Bezeichner für die Chain (ID).\n"
            "- dsl (str, required): Die vollständige Chain-Definition in DSL-Notation (mehrzeilig möglich).\n"
            "- description (str, optional): Eine menschenlesbare Beschreibung, was die Chain tut (für Auto-Matching wichtig).\n"
            "- tags (str, optional): Komma-getrennte Schlagwörter für die Suche (z.B. 'research,summary,web')."
        ),
        "category": ["automation", "dsl", "creation"],
        "flags": {
            "creates_resource": True,
            "requires_validation": True,
            "default_status": "UNSAFE"
        }
    },
    {
        "tool_func": run_chain,
        "name": "run_chain",
        "description": (
            "Führt eine gespeicherte Chain aus. "
            "Die Chain muss vorher mit 'create_validate_chain' erstellt worden sein und den Status VALID haben. "
            "Sicherheitsmechanismus: Beim allerersten Aufruf einer neuen Chain muss zwingend 'accept=True' gesetzt werden, "
            "um die Chain als sicher zu markieren (Status: ACCEPTED). Danach kann sie beliebig oft ohne diesen Parameter ausgeführt werden.\n\n"
            "Verhalten:\n"
            "- Agents innerhalb der Chain wissen, dass sie Teil eines Prozesses sind und fokussieren sich strikt auf ihren Step.\n"
            "- Der Output des letzten Steps wird zurückgegeben.\n\n"
            "### PARAMETER INFORMATIONEN\n"
            "- name_or_id (str, required): Der Name oder die ID der auszuführenden Chain.\n"
            "- input_data (str, optional): Die Start-Daten für den ersten Step der Chain (ersetzt die Variable {input}).\n"
            "- accept (boolean, optional): Muss beim ersten Run 'True' sein, um die Chain dauerhaft zu autorisieren."
        ),
        "category": ["execution", "automation"],
        "flags": {
            "executes_code": True,
            "requires_safety_accept": True
        }
    },
    {
        "tool_func": list_auto_get_fitting,
        "name": "list_auto_get_fitting",
        "description": (
            "Listet alle gespeicherten Chains auf und analysiert deren Eignung. "
            "Diese Funktion dient sowohl als einfache Liste als auch als semantische Suche.\n\n"
            "Funktionsweise:\n"
            "Wenn eine 'task_description' angegeben wird, führt das System ein Auto-Matching durch. "
            "Es findet die am besten passenden Chains basierend auf Name, Beschreibung, Tags und den verwendeten Tools innerhalb der Chain. "
            "Das Ergebnis zeigt Status (ACCEPTED/UNSAFE, VALID/INVALID), Beschreibung, Tags und die Anzahl der bisherigen Ausführungen.\n\n"
            "### PARAMETER INFORMATIONEN\n"
            "- task_description (str, optional): Eine natürlichsprachliche Beschreibung der Aufgabe, die gelöst werden soll. "
            "Dient zum Filtern und Ranking der Chains.\n"
            "- show_all (boolean, optional): Wenn True, wird die vollständige DSL-Definition jeder Chain im Output angezeigt (Standard: False)."
        ),
        "category": ["search", "information", "discovery"],
        "flags": {
            "read_only": True,
            "auto_match": True
        }
    }
]
