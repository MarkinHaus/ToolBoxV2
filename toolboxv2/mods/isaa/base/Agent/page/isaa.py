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

import dataclasses
import inspect
import json
import logging
import types
from pathlib import Path
from typing import Any, AsyncIterator

from .core import CURRENT_SESSION, Call, Capsule, Config, PageEngine, Reply
from .kernel import InProcessKernel
from .store import DirStore, Fts, JsonlEvents

from toolboxv2 import get_logger
logger = get_logger()


def _delta_text(item) -> str:
    """Visible text from a STREAMING DELTA chunk only.

    Unlike ``_chunk_text`` this deliberately IGNORES a terminal/aggregated
    ``ModelResponse`` (``.choices[0].message.content``): during streaming the
    deltas already carried the full text, so reading the final object's content
    again would duplicate the whole answer. Returns "" for a non-delta object.
    """
    if isinstance(item, str):
        return item
    try:
        delta = item.choices[0].delta
        if getattr(delta, "content", None):
            return delta.content
    except (AttributeError, IndexError):
        pass
    return ""


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
            # Non-string item: either a streaming DELTA chunk (contributes text)
            # or the terminal aggregated ModelResponse (full content + tool_calls).
            # Only DELTA text may be yielded — the terminal object's full content
            # would DUPLICATE the whole answer (deltas already streamed it), so we
            # take ONLY its tool_calls from it. (Bug: "Hallo" -> "HalloHallo".)
            text = _delta_text(item)
            if text:
                yield text
            calls = _extract_tool_calls(item)
            if calls:
                final_calls = calls

        # Tool-Calls als JSON-Block anhängen -> _parse_streamed() findet sie
        if final_calls:
            yield "\n```json\n" + json.dumps(final_calls) + "\n```"


# ================================================================== Tools
class FlowAgentTools:
    """Tools-Port über FlowAgent.tool_manager und arun_function.

    * LIVE-Sicht: Index/Docs werden bei Registry-Änderung neu gebaut (``refresh``).
      Vorher war es ein Snapshot aus ``__init__`` -> alles, was danach registriert
      wurde (vfs_shell/vfs_view/search_vfs/sandbox_*/docker_* aus
      ``init_session_tools``, MCP, ``add_tool``), war für ``tool_search``,
      ``dir(tools)`` und den Kernel-Digest unsichtbar.
    * Session-Warmup: ``ensure_session`` ruft ``agent.init_session_tools`` – das hat auf dem
      Page-Pfad bisher niemand getan (nur FlowAgent.a_run*).
    * Session-gebundene Tools (``SESSION_TOOLS``) sind Closures über EINE Session und global
      im tool_manager registriert (zuletzt initialisierte Session gewinnt -> Sub-/Parallel-
      Sessions würden auf dem VFS der falschen Session arbeiten).  ``call`` bindet sie
      deshalb pro aktueller Session (``CURRENT_SESSION``).
    """

    SESSION_TOOLS = ("vfs_shell", "vfs_view")

    def __init__(self, agent):
        self.agent = agent
        self.index = Fts()
        self.docs: dict[str, str] = {}
        self._sig: int | None = None
        self._warm: set[str] = set()
        self._bound: dict[tuple[str, str], Any] = {}
        self.refresh(force=True)

    # ------------------------------------------------------------ discovery
    def refresh(self, force: bool = False) -> bool:
        """Index neu aufbauen, falls sich die Menge der Tool-Namen geändert hat."""
        tools = list(self.agent.tool_manager.get_all())
        sig = hash(tuple(sorted(t.name for t in tools)))
        if sig == self._sig and not force:
            return False
        index, docs = Fts(), {}
        for t in tools:
            cats = t.category if isinstance(t.category, list) else [t.category]
            docs[t.name] = (t.description or "").split("\n")[0]
            index.add(t.name, f"{t.name} {docs[t.name]} {' '.join(c for c in cats if c)}")
        old = self.index
        self.index, self.docs, self._sig = index, docs, sig
        close = getattr(old, "close", None)
        if callable(close) and old is not index:
            close()
        return True

    def search(self, query: str, k: int) -> list[tuple[str, str]]:
        self.refresh()
        if not query:
            # Leerer Query = alle Tools (für dir(tools) / digest im Kernel)
            return list(self.docs.items())[:k]
        toks = query.lower().split()
        direct = [n for n in self.docs if all(t in n.lower() for t in toks)]  # Name-Treffer zuerst
        try:
            bm25 = [n for n, _ in self.index.search(query, k)]
        except Exception:  # FTS-Syntaxfehler im Query -> nur Name-Treffer
            bm25 = []
        order = list(dict.fromkeys(direct + bm25))[:k]
        return [(n, self.docs.get(n, "")) for n in order]

    # ------------------------------------------------------------ sessions
    async def ensure_session(self, sid: str) -> None:
        """Session-VFS + session-gebundene Tools registrieren (idempotent)."""
        if sid in self._warm:
            return
        sm = getattr(self.agent, "session_manager", None)
        if sm is None:
            self._warm.add(sid)
            return
        try:
            session = await sm.get_or_create(sid)
            self.agent.init_session_tools(session)  # no-op wenn tools_initialized
        except Exception as e:  # pragma: no cover - env dependent
            logger.warning("FlowAgentTools.ensure_session(%s) failed: %s", sid, e)
            return  # beim nächsten Call erneut versuchen
        self._warm.add(sid)
        self.refresh()

    def _session_fn(self, name: str, sid: str):
        key = (name, sid)
        fn = self._bound.get(key)
        if fn is None:
            sm = getattr(self.agent, "session_manager", None)
            session = getattr(sm, "sessions", {}).get(sid) if sm is not None else None
            if session is None:
                return None
            from toolboxv2.mods.isaa.base.patch.vfs_shell_tool import (
                make_vfs_shell,
                make_vfs_view,
            )
            fn = (make_vfs_shell if name == "vfs_shell" else make_vfs_view)(session)
            self._bound[key] = fn  # lazy: nach evtl. RestrictedVFSWrapper der Session
        return fn

    def drop_session(self, sid: str) -> None:
        self._warm.discard(sid)
        for key in [k for k in self._bound if k[1] == sid]:
            self._bound.pop(key, None)

    # ------------------------------------------------------------ call
    async def call(self, name: str, args: dict):
        sid = CURRENT_SESSION.get()
        if sid is not None:
            await self.ensure_session(sid)
            if name in self.SESSION_TOOLS:
                fn = self._session_fn(name, sid)
                if fn is not None:
                    out = fn(**args)
                    return await out if inspect.isawaitable(out) else out
        return await self.agent.arun_function(name, **args)

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

    def __init__(self, agent, bytes_root: Path, root: Path | None = None):
        self.agent = agent
        self.bytes = DirStore(bytes_root)
        # Parent dir that holds the per-session folders (where state.json lives).
        # Each session dir is mounted at /.page/<sid> with auto_sync so offloaded
        # tail/summary entries persist on disk next to state.json and survive a
        # restart (otherwise /.page/... lives only in-memory and recall fails
        # after the process exits with a FileNotFoundError on the bytes store).
        self.root = Path(root) if root is not None else None
        self._mounted: set[str] = set()

    async def ensure(self, sid: str) -> None:
        """Pre-create + initialize the session's VFS on the event loop, and
        mount the session's on-disk dir at /.page/<sid> for write-through
        persistence.

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
            return
        self._mount_session(sid)

    def _mount_session(self, sid: str) -> None:
        """Mount root/<sid> at /.page/<sid> (auto_sync) once, for persistence."""
        if self.root is None or sid in self._mounted:
            return
        vfs = self._vfs(sid)
        if vfs is None or not hasattr(vfs, "mount"):
            return
        local = self.root / sid
        try:
            local.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover - env dependent
            logger.warning("VfsStore mount dir create failed for %s: %s", sid, e)
            return
        vfs_path = f"/.page/{sid}"
        existing = getattr(vfs, "mounts", {})
        if isinstance(existing, dict) and vfs_path in existing:
            self._mounted.add(sid)
            return
        try:
            result = vfs.mount(str(local), vfs_path=vfs_path, auto_sync=True)
            if isinstance(result, dict) and result.get("success") is False:
                # Already mounted or benign failure: still mark to avoid retrying.
                logger.debug("VfsStore.mount(%s) note: %s", sid, result.get("error"))
            self._mounted.add(sid)
        except Exception as e:  # pragma: no cover - env dependent
            logger.warning("VfsStore.mount(%s) failed: %s", sid, e)

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
        # Fallback auf DirStore (Bytes, nicht aufgewärmt, oder nicht im VFS).
        # Ein fehlender Ref darf NICHT als FileNotFoundError durchschlagen
        # (sonst crasht recall mit rohem Traceback) -> leere Bytes zurückgeben,
        # der Aufrufer meldet das als sauberen Fehler.
        try:
            return self.bytes.get(path)
        except FileNotFoundError:
            logger.debug("VfsStore.get miss (not in VFS, not on disk): %s", path)
            return b""
        except Exception as e:  # pragma: no cover - env dependent
            logger.warning("VfsStore.get failed for %s: %s", path, e)
            return b""

    def delete_prefix(self, prefix: str) -> str:
        """Delete every entry under ``prefix`` (store path like ``"<sid>/"``).

        Removes the subtree from the VFS, from the bytes DirStore, and from the
        physical mounted session dir so offloaded tail/summary entries are gone
        for good (used by ``PageEngine.clear_session``). Fail-safe."""
        sid, _, rest = prefix.partition("/")
        vfs = self._vfs(sid)
        vpath = f"/.page/{sid}/{rest}".rstrip("/")
        if vfs is not None:
            try:
                rmdir = getattr(vfs, "rmdir", None)
                if callable(rmdir):
                    rmdir(vpath, force=True)
                else:
                    vfs.delete(vpath)
            except Exception as e:  # pragma: no cover - env dependent
                logger.debug("VfsStore.delete_prefix vfs rm %s: %s", vpath, e)
        # bytes DirStore
        dp = getattr(self.bytes, "delete_prefix", None)
        if callable(dp):
            try:
                dp(prefix)
            except Exception as e:  # pragma: no cover
                logger.debug("VfsStore.delete_prefix bytes %s: %s", prefix, e)
        # physical mounted dir (root/<sid>/<rest>)
        if self.root is not None:
            import shutil
            target = self.root / sid / rest.rstrip("/") if rest.strip("/") else self.root / sid
            try:
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                elif target.exists():
                    target.unlink()
            except Exception as e:  # pragma: no cover
                logger.debug("VfsStore.delete_prefix disk %s: %s", target, e)
        return prefix

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


# ================================================================== obs bridge
# Page events the obs bridge consumes. Everything else is forwarded to inner
# only. High-frequency stream chunks (content/content_chunk) are ignored here —
# obs records one step per completed LLM call (the ``llm_call`` event), which
# already carries ms + usage.
def _usage_tokens(usage: dict) -> tuple[int, int]:
    """Read (tokens_in, tokens_out) from a usage dict, tolerant of key names."""
    if not isinstance(usage, dict):
        return 0, 0
    tin = usage.get("prompt_tokens", usage.get("tokens_in", usage.get("input_tokens", 0)))
    tout = usage.get("completion_tokens", usage.get("tokens_out", usage.get("output_tokens", 0)))
    try:
        return int(tin or 0), int(tout or 0)
    except (TypeError, ValueError):
        return 0, 0


class ObsEventSink:
    """Compose an existing ``on_event`` sink with a native ``ObservabilityLayer``.

    Forwards EVERY event to ``inner`` unchanged, and in addition drives obs so
    page runs land in the native observability pipeline exactly like V3 runs —
    WITHOUT touching ``core.py``.

    KEY INSIGHT (verified against a live run): ``FlowAgent.a_run_llm_completion``
    ALREADY records the real token usage onto the SAME ``agent.obs`` via
    ``record_llm_start`` / ``record_llm_end`` — and it fires those calls just
    BEFORE the engine emits the ``llm_call`` event. obs only attaches an LLM
    record to a step if a step is OPEN at that moment. Therefore the sink must
    NOT write tokens itself (that would overwrite the real record with zero).
    Instead it keeps the page RUN + STEP structure so the FlowAgent's tokens
    land in the right step:

      run_begin   -> begin_run(run_id, query, session_id) + open a step
      llm_call    -> end_step() (the FlowAgent already filled it) + open the
                     next step (to catch the next FlowAgent LLM call)
      page_commit -> record_vfs_delta into the open step (page update as a diff;
                     the initial "init" commit is skipped)
      tool_start  -> record_tool_start(name)
      tool_result -> record_tool_end(name, status)
      run_end     -> discard the trailing (unused, pre-opened) step + end_run

    obs is ContextVar-based; page runs stream in a separate async task, so the
    sink re-asserts the active run id before each obs call (``_reassert``).
    Fail-safe (obs errors never break the engine), no-op when obs is None,
    inner always receives every event, events with no active run are ignored.
    """

    def __init__(self, inner, obs, *, agent_name: str = "page"):
        self._inner = inner or (lambda type, **data: None)
        self._obs = obs
        self._agent = agent_name
        self._iter = 0           # per-run step counter
        self._active = False     # True between run_begin and run_end
        self._run_id = ""        # the run obs is tracking for this stream
        self._step_open = False  # True while a step is open (pre-opened for the FlowAgent)

    def _reassert(self) -> None:
        """Re-point obs at the run we are tracking.

        obs resolves the active run via a ContextVar that ``begin_run`` sets.
        Page runs stream inside a separate asyncio task, so later events can fire
        in a DIFFERENT context than ``run_begin`` — then obs would silently drop
        them. We re-assert the active run id before every obs call: via the test
        hook ``set_active`` if present, and via the real
        ``observability._active_run_id`` ContextVar when available.
        """
        if not self._run_id:
            return
        setter = getattr(self._obs, "set_active", None)
        if callable(setter):
            try:
                setter(self._run_id)
            except Exception:
                pass
        sticky = getattr(self._obs, "set_sticky_run", None)
        if callable(sticky):
            try:
                sticky(self._run_id)
            except Exception:
                pass
        try:
            from toolboxv2.mods.isaa.base.Agent.observability import _active_run_id
            _active_run_id.set(self._run_id)
        except Exception:
            pass

    def _open_step(self) -> None:
        """Open the next step so the FlowAgent's upcoming record_llm_* attaches."""
        self._iter += 1
        self._reassert()
        self._obs.begin_step(self._iter)
        self._step_open = True

    def _close_step(self) -> None:
        if not self._step_open:
            return
        self._reassert()
        self._obs.end_step()
        self._step_open = False

    def _discard_step(self) -> None:
        if not self._step_open:
            return
        self._reassert()
        discard = getattr(self._obs, "discard_step", None)
        if callable(discard):
            discard()
        else:  # pragma: no cover - fallback for older obs without discard_step
            self._obs.end_step()
        self._step_open = False

    def __call__(self, type: str, **data) -> None:
        try:
            self._inner(type, **data)
        finally:
            if self._obs is None:
                return
            try:
                self._dispatch(type, data)
            except Exception as e:  # pragma: no cover - obs must never break the run
                logger.debug("page obs bridge failed for %s: %s", type, e)

    def _dispatch(self, type: str, data: dict) -> None:
        if type == "run_begin":
            self._iter = 0
            self._active = True
            self._step_open = False
            self._run_id = str(data.get("run") or data.get("run_id") or "page")
            self._obs.begin_run(
                run_id=self._run_id,
                query=data.get("query", ""),
                session_id=data.get("session", data.get("session_id", "")),
            )
            # Pre-open a step so the FlowAgent's first record_llm_* lands in it.
            self._open_step()
        elif type == "llm_call":
            if not self._active:
                return
            # The FlowAgent already filled the open step with the real tokens
            # (it fired record_llm_start/end just before this event). Close it,
            # then open the next step to catch a possible next LLM call.
            self._close_step()
            self._open_step()
        elif type == "page_commit":
            # A page update -> record it as a VFS delta in the current (open)
            # step. The initial empty-page commit ("init") is not a diff.
            if not self._active or data.get("reason") == "init":
                return
            if not self._step_open:
                self._open_step()
            self._reassert()
            delta = {
                "path": f"page://v{data.get('v')}",
                "action": str(data.get("reason", "commit")),
                "lines_added": 0,
                "lines_removed": 0,
                **{k: v for k, v in data.items()},
            }
            self._obs.record_vfs_delta(
                path=delta["path"], action=delta["action"], delta_dict=delta,
            )
        elif type == "tool_start":
            if self._active:
                self._reassert()
                self._obs.record_tool_start(
                    name=str(data.get("name", "")),
                    call_id=str(data.get("id", "")),
                )
        elif type == "tool_result":
            if self._active:
                self._reassert()
                self._obs.record_tool_end(
                    name=str(data.get("name", "")),
                    status="error" if data.get("ok") is False else "ok",
                    call_id=str(data.get("id", "")),
                )
        elif type == "run_end":
            if not self._active:
                return
            self._active = False
            # The last pre-opened step is unused (no LLM call followed) -> drop it
            # so it does not count as an extra empty iteration.
            self._discard_step()
            self._reassert()
            self._obs.end_run(
                success=bool(data.get("success", True)),
                final_answer=str(data.get("final", data.get("final_answer", ""))),
            )
            # Clear the cross-task anchor so a later background fold / next run is
            # not mis-attributed to this (now closed) run.
            clear = getattr(self._obs, "set_sticky_run", None)
            if callable(clear):
                try:
                    clear(None)
                except Exception:
                    pass

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

    store = VfsStore(agent, root / "media", root=root)
    tools_port = FlowAgentTools(agent)

    async def _warm_vfs(_engine, session) -> str:
        """Prep hook: initialize the session VFS so the sync Store path can
        resolve it without awaiting an async get_or_create (fixes the
        'coroutine object has no attribute vfs' crash).  Registriert zusätzlich
        die Session-Tools (vfs_shell/vfs_view/...) -> sofort in tool_search sichtbar."""
        await store.ensure(session.id)
        await tools_port.ensure_session(session.id)
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
    # Compose the native ObservabilityLayer bridge on top (opt-in: only when the
    # agent has an obs layer). Page runs then land in obs like V3 runs — steps,
    # tokens, tool calls — without any change to core.py. Fail-safe + no-op if
    # obs is absent; it wraps the (possibly audit-wrapped) inner sink so every
    # event still reaches JSONL + audit unchanged.
    _obs = getattr(agent, "obs", None)
    if _obs is not None:
        _sink = ObsEventSink(_sink, _obs, agent_name=getattr(agent.amd, "name", "page"))

    return PageEngine(
        root,
        FlowAgentLLM(agent),
        tools=tools_port,
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



# ================================================================== sub-agent factory
SUB_AGENT_PROMPT = """\
You are a focused SUB-AGENT for one specific task.
- You can ONLY write to {output_dir}/ (reads: entire VFS).
- You CANNOT spawn sub-agents and CANNOT ask follow-up questions - work with the given information.
- Token budget: {budget}. Step limit: {max_steps}.
- Write your result to {output_dir}/result.md (if it exists already: _result.md), then call final_answer.
- The live status shows your remaining steps ("steps N/limit"). Start wrapping up at
  20% remaining: write result.md, then final_answer. If the step limit hits without
  final_answer, the parent sees NO result. Never keep exploring when you could deliver.
"""


def get_base_page_engine(agent, root: str | Path | None = None, cfg: Config | None = None) -> PageEngine:
    """Gemeinsame Basis-PageEngine pro Agent (llm/tools/store/prep werden von allen Subs geteilt).

    Nutzt die Engine des PageExecutionEngine, falls ``amd.engine == "page"``; sonst eine eigene,
    am Agent gecachte Instanz (``_page_base_engine``)."""
    inner = getattr(getattr(agent, "_execution_engine_cache", None), "engine", None)
    if isinstance(inner, PageEngine):
        return inner
    base = getattr(agent, "_page_base_engine", None)
    if base is None:
        base = make_page_engine(agent, root=root, cfg=cfg)
        try:
            agent._page_base_engine = base
        except Exception:
            pass
    return base


def make_page_sub_engine(
    agent,
    *,
    output_dir: str,
    max_steps: int,
    budget: int | None = None,
    allow: list[str] | None = None,
) -> PageEngine:
    """Eigene PageEngine-Instanz für EINEN Sub-Agent (isolierte Capsule + cfg, geteilte Ports).

    Wie ``PageEngine.spawn_sub``, aber langlebig (Resume) und mit V3-kompatiblem Verhalten:
    keine Rückfragen (ask=None), Step-Limit = max_iterations, Sub-Agent-Prompt."""
    base = get_base_page_engine(agent)
    prompt = SUB_AGENT_PROMPT.format(output_dir=output_dir.rstrip("/"), budget=budget, max_steps=max_steps)
    sysmsg = (getattr(getattr(agent, "amd", None), "system_message", "") or "").strip()
    sub = PageEngine(
        base.root, base.llm, tools=base.tools, kernel=base.kernel_factory,
        store=base.store, decider=base.decider,
        capsule=Capsule(allow=base.capsule.intersect(allow), pool=list(base.capsule.pool)),
        ask=None, cfg=dataclasses.replace(base.cfg, max_steps=int(max_steps)),
        prep=base.prep, static_prompt=f"{sysmsg}\n\n{prompt}".strip(),
        on_event=base.emit, skill_loader=base.skill_loader,
    )
    sub.live = types.SimpleNamespace(narrator_msg="")  # SubAgentManager.get_status liest live.narrator_msg
    sub.sub_output_dir = output_dir  # Fix 09.11.2026: _do_code_run nutzt es fuer den VFS-Pfad-Guard
    return sub


async def close_page_sub_engine(sub: PageEngine) -> None:
    """Nur die EIGENEN Sessions/Kernels/Locks der Sub-Engine schließen (Store/Sink gehören der Basis)."""
    for ss in list(sub.sessions.values()):
        await sub._cancel_bg(ss)
        if ss.prep:
            ss.prep.cancel()
        if ss.kernel:
            await ss.kernel.close()
        if ss.lock_handle:
            ss.lock_handle.close()
        drop = getattr(sub.tools, "drop_session", None)
        if callable(drop):
            drop(ss.id)
    sub.sessions.clear()


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
