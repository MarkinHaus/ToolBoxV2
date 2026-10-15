"""FlowAgent-Integration – einziges Modul das toolboxv2 importiert.

Bekannte API-Korrekturen gegenüber Spec-Text (verifiziert gegen echten Code):

1. `a_run_llm_completion` ist ein async-Generator.  Mit stream=False gibt er das
   letzte Element zurück (ein LiteLLM ModelResponse).  Wir lesen
   `.choices[0].message` aus, nicht das Objekt direkt.

2. `search_vfs` gibt SearchResult-Objekte zurück.  Das Snippet-Feld heißt
   `line_content`, nicht `snippet`.

3. `session_manager.get(sid)` kann None zurückgeben (lazy).  Wir nutzen
   `get_or_create` damit VfsStore nie auf None.vfs zugreift.

4. LLM-Streaming: FlowAgentLLM implementiert die optionale stream()-Methode,
   die den async-Generator von a_run_llm_completion direkt durchreicht.
   Die Engine akkumuliert Chunks und parst Tool-Calls aus dem Gesamttext.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import AsyncIterator

from .core import Call, Capsule, Config, PageEngine, Reply
from .kernel import InProcessKernel
from .store import DirStore, Fts, JsonlEvents

logger = logging.getLogger(__name__)


def _chunk_text(item) -> str:
    """Extrahiert sichtbaren Text aus einem Stream-Chunk oder ModelResponse."""
    # str
    if isinstance(item, str):
        return item
    # LiteLLM delta chunk: .choices[0].delta.content
    try:
        delta = item.choices[0].delta
        if getattr(delta, "content", None):
            return delta.content
    except (AttributeError, IndexError):
        pass
    # ModelResponse: .choices[0].message.content
    try:
        msg = item.choices[0].message
        if getattr(msg, "content", None):
            return msg.content
    except (AttributeError, IndexError):
        pass
    # Objekt mit .content
    if getattr(item, "content", None) and isinstance(item.content, str):
        return item.content
    return ""


def _extract_tool_calls(item) -> list[dict]:
    """Extrahiert Tool-Calls aus einem finalen ModelResponse-/Result-Objekt.
    Gibt [{"id", "name", "args"}, ...] zurück (args als dict)."""
    tcs = None
    # .choices[0].message.tool_calls
    try:
        tcs = item.choices[0].message.tool_calls
    except (AttributeError, IndexError):
        pass
    # direkt .tool_calls (result_obj aus _process_streaming_response)
    if tcs is None:
        tcs = getattr(item, "tool_calls", None)
    if not tcs:
        return []

    out = []
    for i, tc in enumerate(tcs):
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None) if fn else None
        raw = getattr(fn, "arguments", "{}") if fn else "{}"
        if not name:
            continue
        try:
            args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        except json.JSONDecodeError:
            try:
                from toolboxv2.mods.isaa.base.Agent.execution_engine import (
                    _parse_tool_arguments,
                )
                args = _parse_tool_arguments(raw)
            except Exception:
                args = {}
        out.append({"id": getattr(tc, "id", None) or f"c{i}", "name": name, "args": args})
    return out


# ================================================================== LLM
class FlowAgentLLM:
    """LLM-Port über FlowAgent.a_run_llm_completion.

    Unterstützt sowohl complete() (non-stream) als auch stream() (live chunks).
    Die Engine ruft stream() automatisch wenn vorhanden; complete() ist der Fallback.
    """

    def __init__(self, agent, model_preference: str = "fast"):
        self.agent = agent
        self.pref = model_preference

    # ---- non-streaming path (Kontext-Schritte, Fallback)
    async def complete(self, messages: list[dict], tools: list[dict]) -> Reply:
        return await self._to_reply(messages, tools)

    async def _to_reply(self, messages: list[dict], tools: list[dict]) -> Reply:
        """Holt eine vollständige ModelResponse mit Tool-Calls (stream=False).

        a_run_llm_completion ist eine async-Methode:
          - stream=False gibt den letzten yielded Wert zurück.
          - Mit get_response_message=True ist das eine LiteLLM ModelResponse.
        """
        last = await self.agent.a_run_llm_completion(
            messages=messages,
            tools=tools,
            stream=False,
            get_response_message=True,
            with_context=False,
            model_preference=self.pref,
        )
        return self._parse_model_response(last)

    def _parse_model_response(self, last) -> Reply:
        """Baut ein Reply aus einer LiteLLM ModelResponse (oder einem String)."""
        if last is None:
            return Reply()

        # LiteLLM ModelResponse: .choices[0].message enthält content + tool_calls
        try:
            msg = last.choices[0].message
        except (AttributeError, IndexError):
            # Some paths return a message-like object directly (e.g. _AssistantMessage
            # with .content/.tool_calls) instead of a ModelResponse with .choices.
            if hasattr(last, "content") or hasattr(last, "tool_calls"):
                msg = last
            else:
                return Reply(content=str(last) if last else "")

        calls = []
        for tc in getattr(msg, "tool_calls", None) or []:
            raw = getattr(tc.function, "arguments", "{}")
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
            except json.JSONDecodeError:
                try:
                    from toolboxv2.mods.isaa.base.Agent.execution_engine import (
                        _parse_tool_arguments,
                    )
                    args = _parse_tool_arguments(raw)
                except Exception:
                    args = {}
            calls.append(Call(
                id=getattr(tc, "id", f"c{len(calls)}"),
                name=tc.function.name,
                args=args,
            ))

        return Reply(
            content=getattr(msg, "content", "") or "",
            calls=calls,
            reasoning=getattr(msg, "reasoning_content", None) or "",
        )

    # ---- streaming path (Action-Schritte, live Token-Ausgabe)
    async def stream(
        self, messages: list[dict], tools: list[dict]
    ) -> AsyncIterator[str]:
        """Liefert Text-Chunks live.  Tool-Calls kommen am Ende als JSON-Block,
        damit _parse_streamed() in core.py sie erkennt.

        a_run_llm_completion (async-Methode, stream=True) gibt die Generator-FUNKTION
        `internal_stream` zurück.  Diese yielded Text-Chunks (str) und am Ende – bei
        get_response_message=True – ein ModelResponse-Objekt mit den Tool-Calls.
        """
        stream_fn = await self.agent.a_run_llm_completion(
            messages=messages,
            tools=tools,
            stream=True,
            get_response_message=True,   # finales Objekt enthält Tool-Calls
            with_context=False,
            model_preference=self.pref,
        )
        agen = stream_fn() if callable(stream_fn) else stream_fn

        final_calls = []
        async for item in agen:
            if isinstance(item, str):
                if item:
                    yield item
                continue
            # Nicht-String: entweder ein Text tragendes Chunk-Objekt oder das finale Result
            text = _chunk_text(item)
            if text:
                yield text
            # Tool-Calls aus dem (finalen) Objekt sammeln
            calls = _extract_tool_calls(item)
            if calls:
                final_calls = calls

        # Tool-Calls als JSON-Block anhängen -> _parse_streamed() findet sie
        if final_calls:
            yield "\n```json\n" + json.dumps(final_calls) + "\n```"


# ================================================================== Tools
class FlowAgentTools:
    """Tools-Port über FlowAgent.tool_manager und arun_function."""

    def __init__(self, agent):
        self.agent = agent
        self.index = Fts()
        self.docs: dict[str, str] = {}
        for t in agent.tool_manager.get_all():
            cats = t.category if isinstance(t.category, list) else [t.category]
            self.docs[t.name] = (t.description or "").split("\n")[0]
            self.index.add(
                t.name,
                f"{t.name} {self.docs[t.name]} {' '.join(c for c in cats if c)}",
            )

    async def call(self, name: str, args: dict):
        return await self.agent.arun_function(name, **args)

    def search(self, query: str, k: int) -> list[tuple[str, str]]:
        if not query:
            # Leerer Query = alle Tools (für dir(tools) / digest im Kernel)
            return [(name, self.docs.get(name, "")) for name in self.docs][:k]
        return [
            (name, self.docs.get(name, ""))
            for name, _ in self.index.search(query, k)
        ]

    def close(self) -> None:
        """P2.12: release the in-memory Fts SQLite handle (no ResourceWarning)."""
        close = getattr(self.index, "close", None)
        if callable(close):
            close()


# ================================================================== Store
class VfsStore:
    """Store auf dem ISAA Session-VFS.

    Text landet in /.page/<sid>/... und wird vom VFSMemoryIndexer indexiert.
    Bytes (Bilder) gehen in einen DirStore daneben.

    Korrekturen gegenüber Spec-Text:
    - session_manager.get_or_create() statt get() (vermeidet None.vfs)
    - search_vfs-Ergebnisse haben .line_content, nicht .snippet
    - mkdir() ohne parents=True aufrufen wenn der Parent schon existiert
    """

    def __init__(self, agent, bytes_root: Path):
        self.agent = agent
        self.bytes = DirStore(bytes_root)

    async def ensure(self, sid: str) -> None:
        """Pre-create + initialize the session's VFS on the event loop.

        ``session_manager.get_or_create`` is an async method, but the Store API
        (``put``/``get``/``search``) is synchronous and runs deep inside the
        engine loop.  We therefore warm the session once (via the page engine's
        prep hook) so ``_vfs_and_path`` can resolve it synchronously afterwards.
        """
        sm = getattr(self.agent, "session_manager", None)
        if sm is None:
            return
        try:
            await sm.get_or_create(sid)
        except Exception as e:  # pragma: no cover - env dependent
            logger.warning("VfsStore.ensure(%s) failed: %s", sid, e)

    def _vfs(self, sid: str):
        """Resolve an already-initialized session's VFS synchronously.

        Returns the VFS or ``None`` if the session is not warm yet (callers then
        fall back to the local bytes DirStore instead of crashing on a coroutine).
        """
        sm = getattr(self.agent, "session_manager", None)
        if sm is None:
            return None
        session = getattr(sm, "sessions", {}).get(sid)
        # Never touch the async get_or_create here: a coroutine has no .vfs.
        if session is None:
            return None
        return getattr(session, "vfs", None)

    def _vfs_and_path(self, path: str):
        """Gibt (vfs|None, vfs_path) zurück.  vfs ist None wenn die Session
        noch nicht aufgewärmt ist (Fallback auf DirStore)."""
        sid, _, rest = path.partition("/")
        return self._vfs(sid), f"/.page/{sid}/{rest}"

    def put(self, path: str, data: str | bytes) -> str:
        if isinstance(data, bytes):
            return self.bytes.put(path, data)
        vfs, vpath = self._vfs_and_path(path)
        if vfs is None:
            # Session not warm yet: persist in the local bytes store so the
            # reference stays valid and recall() still works.
            self.bytes.put(path, data.encode() if isinstance(data, str) else data)
            return path
        # Elternverzeichnis sicherstellen
        parent = vpath.rsplit("/", 1)[0]
        if not vfs._path_exists(parent):
            vfs.mkdir(parent, parents=True)
        result = vfs.write(vpath, data)
        if not result.get("success"):
            logger.warning("VfsStore.put failed for %s: %s", vpath, result.get("error"))
        return path

    def get(self, path: str) -> bytes:
        vfs, vpath = self._vfs_and_path(path)
        if vfs is not None:
            result = vfs.read(vpath)
            if result.get("success"):
                content = result.get("content", "")
                return content.encode() if isinstance(content, str) else content
        # Fallback auf DirStore (Bytes, nicht aufgewärmt, oder nicht im VFS)
        return self.bytes.get(path)

    def search(self, query: str, k: int, prefix: str = "") -> list[tuple[str, str]]:
        try:
            from toolboxv2.mods.isaa.base.patch.power_vfs import search_vfs, SearchMode
        except ImportError:
            return []
        vfs, root = self._vfs_and_path(prefix or "/")
        if vfs is None:
            return []
        hits = search_vfs(
            vfs=vfs,
            query=query,
            path=root,
            mode=SearchMode.CONTENT,
            max_results=k,
        )
        results = []
        for h in hits:
            # SearchResult hat .path und .line_content (nicht .snippet!)
            hit_path = getattr(h, "path", "")
            snippet = getattr(h, "line_content", "") or ""
            # Pfad zurück auf Store-Format mappen: /.page/<sid>/... -> <sid>/...
            store_path = hit_path.removeprefix("/.page/")
            results.append((store_path, snippet))
        return results

    def close(self) -> None:
        """Gibt den SQLite-Handle des Bytes-DirStore frei (Windows-Dateisperre)."""
        close = getattr(self.bytes, "close", None)
        if callable(close):
            close()

    def search_global(self, query: str, k: int = 3) -> list[tuple[str, str]]:
        """Search the global page library ``/global/.memory/pages/...``.

        Used by the warm-start prep hook: it looks for pages written by similar
        past tasks.  Read-only; returns ``[(vfs_path, snippet), ...]`` or ``[]``
        when the library or the search backend is unavailable.
        """
        try:
            from toolboxv2.mods.isaa.base.patch.power_vfs import search_vfs, SearchMode
        except ImportError:
            return []
        sm = getattr(self.agent, "session_manager", None)
        # The library lives on a shared/global VFS; fall back to the first warm
        # session's VFS if there is no dedicated global one.
        vfs = None
        for cand in ("global_vfs", "shared_vfs"):
            vfs = getattr(sm, cand, None)
            if vfs is not None:
                break
        if vfs is None:
            sessions = getattr(sm, "sessions", {}) or {}
            for sess in sessions.values():
                vfs = getattr(sess, "vfs", None)
                if vfs is not None:
                    break
        if vfs is None:
            return []
        try:
            hits = search_vfs(
                vfs=vfs,
                query=query,
                path="/global/.memory/pages",
                mode=SearchMode.CONTENT,
                max_results=k,
            )
        except Exception as e:  # pragma: no cover - env dependent
            logger.debug("search_global failed: %s", e)
            return []
        return [
            (getattr(h, "path", ""), getattr(h, "line_content", "") or "")
            for h in hits
        ]


# ================================================================== prep hooks
GLOBAL_PAGES = "/global/.memory/pages"


def _latest_query(session) -> str:
    """The most recent user query from the session tail (text only).

    Prep hooks run at commit time; the task to match against is the latest user
    turn.  Returns ``""`` when there is nothing to match yet.
    """
    for it in reversed(session.tail):
        for m in it.get("msgs", []):
            if m.get("role") != "user":
                continue
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(
                    p.get("text", "") for p in c if p.get("type") == "text"
                )
    return ""


def _skills_hook(agent):
    """Prep hook: surface skills whose triggers match the latest query.

    Pure suggestion in P3 — never activates a skill or edits the task plan.
    Skill matching itself lives in the agent's ``SkillsManager`` (the page
    engine stays heuristic-free).
    """

    async def hook(_engine, session) -> str:
        query = _latest_query(session)
        sm = getattr(getattr(agent, "session_manager", None), "skills_manager", None)
        if not query or sm is None:
            return ""
        try:
            if hasattr(sm, "match_skills_async"):
                skills = await sm.match_skills_async(query, 3)
            else:
                skills = sm.match_skills(query, 3)
        except Exception as e:  # pragma: no cover - env dependent
            logger.debug("skills prep hook failed: %s", e)
            return ""
        if not skills:
            return ""
        lines = [f"- {sk.name}: {(sk.description or '').splitlines()[0][:120]}" for sk in skills]
        return "Relevant skills (suggestion — activate with the skill tool):\n" + "\n".join(lines)

    return hook


def _recall_hook(store: "VfsStore"):
    """Prep hook: recall related offloaded page content for the latest query.

    Reads only; the hits are a P3 suggestion, never auto-applied.
    """

    async def hook(_engine, session) -> str:
        query = _latest_query(session)
        if not query:
            return ""
        try:
            hits = store.search(query, 3, prefix=f"{session.id}/")
        except Exception as e:  # pragma: no cover - env dependent
            logger.debug("recall prep hook failed: %s", e)
            return ""
        if not hits:
            return ""
        lines = [f"- recall(ref={ref!r}): {snippet[:120]}" for ref, snippet in hits]
        return "Related earlier content (suggestion):\n" + "\n".join(lines)

    return hook


def _warm_start_hook(store: "VfsStore"):
    """Prep hook: warm-start from the global page library.

    Looks under ``/global/.memory/pages/...`` (indexed VFS) for pages from
    similar past tasks and offers them as a P3 suggestion.  Read-only; it never
    seeds the plan or edits P2 directly.
    """

    async def hook(_engine, session) -> str:
        query = _latest_query(session)
        if not query:
            return ""
        try:
            hits = store.search_global(query, 3)
        except Exception as e:  # pragma: no cover - env dependent
            logger.debug("warm-start prep hook failed: %s", e)
            return ""
        if not hits:
            return ""
        lines = [f"- {path}: {snippet[:120]}" for path, snippet in hits]
        return (
            f"Warm start from the page library ({GLOBAL_PAGES}; suggestion):\n"
            + "\n".join(lines)
        )

    return hook


def _skill_loader(agent):
    """Build the ``skill_loader`` callback for the ``activate_skill`` action.

    ``load(name_or_id) -> {instruction, tools, title} | None``.  Looks the skill
    up in the agent's ``SkillsManager`` (by id first, then by ``.name``).  The
    engine writes the instruction into P3 and grants ``tool:<t>`` for each tool
    the skill uses.
    """

    def load(name: str):
        sm = getattr(getattr(agent, "session_manager", None), "skills_manager", None)
        if sm is None:
            return None
        skills = getattr(sm, "skills", {}) or {}
        skill = skills.get(name) or next(
            (sk for sk in skills.values() if getattr(sk, "name", None) == name), None
        )
        if skill is None:
            return None
        return {
            "instruction": getattr(skill, "instruction", "") or "",
            "tools": list(getattr(skill, "tools_used", []) or []),
            "title": getattr(skill, "name", name),
        }

    return load


# ================================================================== audit bridge
# Which page events become native-audit entries, and their action suffix.
# High-frequency stream events (content_chunk, iteration_start, tool_start,
# tool_result, llm_call) are intentionally NOT audited — only decisions and
# system activity, so there is no hot-path latency.
_AUDIT_EVENTS: dict[str, str] = {
    "run_begin": "run.begin",
    "run_end": "run.end",
    "page_commit": "page.commit",
    "decider": "decider",            # covers all 4 decider consumers
    "check": "validate.check",       # soft-check path (decider-backed)
    "error_recall": "error.recall",
    "cap_decision": "capability",
    "skill_activated": "skill.activate",
    "prep_error": "prep.error",
    "fold_cancelled": "fold.cancelled",
}


class AuditEventSink:
    """Compose an existing ``on_event`` sink with the native toolbox audit log.

    Forwards EVERY event to ``inner`` unchanged (JSONL stays complete) and, for
    the selected ``_AUDIT_EVENTS``, also calls
    ``audit_logger.log_action(user_id, action, resource, status, details)`` — the
    same API CloudM/auth/obs use, so page activity lands in the native audit
    pipeline.  No new calls in the engine hot path (it reuses ``emit``); no extra
    buffering; a broken audit logger never breaks the engine.
    """

    def __init__(self, inner, audit_logger, *, agent_name: str = "page"):
        self._inner = inner or (lambda type, **data: None)
        self._audit = audit_logger
        self._agent = agent_name

    def __call__(self, type: str, **data) -> None:
        # 1) inner sink always (JsonlEvents etc.) — unchanged behaviour
        try:
            self._inner(type, **data)
        finally:
            # 2) selected events -> native audit (fail-safe, never raises)
            if self._audit is None:
                return
            action = _AUDIT_EVENTS.get(type)
            if action is None:
                return
            try:
                status = "SUCCESS"
                if type == "cap_decision" and data.get("how") == "deny":
                    status = "FAILURE"
                elif type in ("prep_error", "fold_cancelled"):
                    status = "FAILURE"
                self._audit.log_action(
                    user_id=self._agent,
                    action=f"agent.page.{action}",
                    resource=f"/agent/{self._agent}/page",
                    status=status,
                    details={k: v for k, v in data.items()},
                )
            except Exception as e:  # pragma: no cover - audit must never break the run
                logger.debug("page audit log failed for %s: %s", type, e)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()


def _resolve_audit_logger():
    """Resolve ``app.audit_logger`` once (None if unavailable). No per-event cost."""
    try:
        from toolboxv2 import get_app
        app = get_app()
        audit = getattr(app, "audit_logger", None)
        return audit
    except Exception:
        return None


# ================================================================== factory
def make_page_engine(
    agent,
    root: str | Path | None = None,
    cfg: Config | None = None,
) -> PageEngine:
    """Verdrahtet PageEngine mit einem echten FlowAgent.

    Nutzt InProcessKernel mit AgentToolProxy, VfsStore für Text,
    DirStore für Bytes/Bilder, und JsonlEvents für Observability.
    """
    if root is None:
        from toolboxv2 import get_app
        root = (
            Path(get_app().data_dir) / "Agents" / agent.amd.name / "page"
        )
    root = Path(root)

    store = VfsStore(agent, root / "media")

    async def _warm_vfs(_engine, session) -> str:
        """Prep hook: initialize the session VFS so the sync Store path can
        resolve it without awaiting an async get_or_create (fixes the
        'coroutine object has no attribute vfs' crash)."""
        await store.ensure(session.id)
        return ""

    # Compose the JSONL event sink with the native audit pipeline (opt-out via
    # amd.enable_audit). Zero hot-path cost: the engine already emits events;
    # the sink only forwards decisions + system activity to app.audit_logger.
    _sink = JsonlEvents(root / "events.jsonl")
    if getattr(agent.amd, "enable_audit", True):
        _audit = _resolve_audit_logger()
        if _audit is not None:
            _sink = AuditEventSink(
                _sink, _audit, agent_name=getattr(agent.amd, "name", "page")
            )

    return PageEngine(
        root,
        FlowAgentLLM(agent),
        tools=FlowAgentTools(agent),
        kernel=InProcessKernel,
        store=store,
        capsule=Capsule.load(str(root / "capsule.json")),
        cfg=cfg,
        static_prompt=agent.amd.system_message or "",
        on_event=_sink,
        prep={
            "_warm_vfs": _warm_vfs,          # must run first (warms the VFS)
            "skills": _skills_hook(agent),
            "recall": _recall_hook(store),
            "warm_start": _warm_start_hook(store),
        },
        skill_loader=_skill_loader(agent),
    )



# ================================================================== V3 drop-in factory
def make_page_execution_engine(
    agent,
    root: str | Path | None = None,
    cfg: Config | None = None,
):
    """Wire a :class:`PageExecutionEngine` (V3 drop-in) around a real FlowAgent.

    This is the object ``FlowAgent._get_execution_engine`` should return when
    ``amd.engine == "page"`` — it exposes the same public API as the V3
    ``ExecutionEngine`` (execute / execute_stream / resume / pause / cancel /
    list_executions / get_execution / get_interrupted_runs / get_stats).
    """
    from .compat_v3 import PageExecutionEngine

    engine = make_page_engine(agent, root=root, cfg=cfg)
    return PageExecutionEngine(engine, agent=agent)


def get_execution_engine_for(agent, cache_attr: str = "_execution_engine_cache"):
    """Engine switch helper for ``FlowAgent._get_execution_engine``.

    Returns a cached ``PageExecutionEngine`` when ``agent.amd.engine == "page"``,
    otherwise ``None`` (caller falls back to the V3 engine).  Honors the existing
    ``_execution_engine_cache`` attribute so the engine is reused.

    Intended one-line integration at the top of ``_get_execution_engine``:

        page_engine = get_execution_engine_for(self)
        if page_engine is not None:
            return page_engine
    """
    if getattr(getattr(agent, "amd", None), "engine", "v3") != "page":
        return None
    cached = getattr(agent, cache_attr, None)
    if cached is not None and cached.__class__.__name__ == "PageExecutionEngine":
        return cached
    engine = make_page_execution_engine(agent)
    try:
        setattr(agent, cache_attr, engine)
    except Exception:
        pass
    return engine
