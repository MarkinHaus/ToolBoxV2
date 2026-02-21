"""
MemoryKnowledgeActor — Mini-Agent for V2 Memory System (HybridMemoryStore)

Adapted from AgentKnowledgeActor.py to work with the new V2 memory backend
(SQLite + FAISS + FTS5) instead of KnowledgeBase.

The agent can:
- Search across memory spaces (vector + BM25 + concept relation)
- Add/remove data points with embeddings
- Manage entities and relations in the SQLite graph
- Combine data points via LLM summarization
- Analyze concept clusters and relations
- Run an autonomous analysis loop driven by LLM tool selection
"""

import asyncio
import inspect
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Defines the structure for a tool call requested by the LLM."""
    tool_name: str = Field(..., description="The name of the tool to be executed.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="The parameters to pass to the tool.")


def _format_query_results(results: list[dict]) -> str:
    """Format HybridMemoryStore query results into a readable string."""
    if not results:
        return "No results found."
    lines = ["QUERY RESULTS", "=" * 40]
    for i, hit in enumerate(results, 1):
        content = hit.get("content", "").strip()
        score = hit.get("score", 0.0)
        content_type = hit.get("content_type", "text")
        concepts = hit.get("concepts", [])
        source = hit.get("meta", {}).get("source", "")

        lines.append(f"\n--- Result {i} (score: {score:.4f}, type: {content_type}) ---")
        if source:
            lines.append(f"Source: {source}")
        lines.append(content[:500])
        if concepts:
            lines.append(f"Concepts: {', '.join(concepts[:10])}")
    return "\n".join(lines)


def _format_entity_relations(relations: list[dict]) -> str:
    """Format entity relation results into a readable string."""
    if not relations:
        return "No relations found."
    lines = ["ENTITY RELATIONS", "=" * 40]
    for rel in relations:
        lines.append(
            f"  {rel.get('name', rel['id'])} ({rel.get('type', '?')}) "
            f"--[{rel['rel_type']}]--> weight={rel.get('weight', 1.0):.2f} "
            f"depth={rel.get('depth', 1)}"
        )
    return "\n".join(lines)


class MemoryKnowledgeActor:
    """
    Mini-agent that works with V2 HybridMemoryStore to analyze, connect,
    and manipulate data through LLM-driven tool selection loops.

    Works with either:
    - A single HybridMemoryStore (direct)
    - An AISemanticMemory instance (multi-space, with embedding generation)
    """

    def __init__(
        self,
        memory,
        space_name: str = "default",
        embed_fn=None,
    ):
        """
        Initialize the MemoryKnowledgeActor.

        Args:
            memory: Either a HybridMemoryStore or AISemanticMemory instance
            space_name: The memory space to work with (for AISemanticMemory)
            embed_fn: Async callable (text: str) -> np.ndarray for embeddings.
                      If memory is AISemanticMemory, uses its get_embeddings().
                      Must be provided if memory is a raw HybridMemoryStore.
        """
        from .hybrid_memory import HybridMemoryStore
        from .ai_semantic_memory import AISemanticMemory

        self.space_name = space_name
        self.analysis_history: list[dict] = []

        if isinstance(memory, AISemanticMemory):
            self._semantic = memory
            self._store = memory._get_or_create_store(
                memory._sanitize_name(space_name)
            )
            self._embed_fn = embed_fn or memory.get_embeddings
        elif isinstance(memory, HybridMemoryStore):
            self._semantic = None
            self._store = memory
            if embed_fn is None:
                raise ValueError(
                    "embed_fn is required when using a raw HybridMemoryStore"
                )
            self._embed_fn = embed_fn
        else:
            raise TypeError(f"Unsupported memory type: {type(memory)}")

        self._register_tools()

    # ──────────────────────────────────────────────────────────────────
    # Tool Registration
    # ──────────────────────────────────────────────────────────────────

    def _register_tools(self):
        """Register all available tools (work-set + analysis-set)."""
        self.tools: dict[str, Any] = {}

        # Work set — data manipulation
        self.tools.update({
            "add_data_point": self.add_data_point,
            "remove_data_point": self.remove_data_point,
            "add_entity": self.add_entity,
            "add_relation": self.add_relation,
            "remove_relation": self.remove_relation,
            "combine_2_data_points": self.combine_2_data_points,
        })

        # Analysis set — querying and inspection
        self.tools.update({
            "search": self.search,
            "search_by_concept": self.search_by_concept,
            "get_related_entities": self.get_related_entities,
            "get_entry_concepts": self.get_entry_concepts,
            "get_stats": self.get_stats,
            "list_concepts": self.list_concepts,
            "final_analysis": self.final_analysis,
        })

    def _get_tool_signatures(self) -> str:
        """Generate formatted tool signatures for the LLM prompt."""
        signatures = []
        for name, func in self.tools.items():
            try:
                sig = inspect.signature(func)
                doc = inspect.getdoc(func) or "No description available."
                signatures.append(f"- {name}{sig}:\n  {doc.strip()}")
            except TypeError:
                signatures.append(f"- {name}(...): No signature available.")
        return "\n".join(signatures)

    # ──────────────────────────────────────────────────────────────────
    # Work Set: Data Manipulation Tools
    # ──────────────────────────────────────────────────────────────────

    async def add_data_point(
        self,
        text: str,
        content_type: str = "text",
        concepts: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a new data point to the memory store with embedding generation.

        Args:
            text: The text content to add.
            content_type: Type of content (text, code, fact, entity).
            concepts: List of concept keywords to associate.
            metadata: Additional metadata dict.
        """
        embedding = await self._embed_fn(text)
        entry_id = self._store.add(
            content=text,
            embedding=embedding,
            content_type=content_type,
            meta=metadata,
            concepts=concepts,
        )
        return f"Added data point '{entry_id}' ({content_type}, {len(text)} chars, {len(concepts or [])} concepts)."

    async def remove_data_point(self, entry_id: str, hard: bool = False) -> str:
        """Remove a data point by its entry ID.

        Args:
            entry_id: The ID of the entry to remove.
            hard: If True, permanently delete. Otherwise soft-delete.
        """
        self._store.delete(entry_id, hard=hard)
        mode = "hard-deleted" if hard else "soft-deleted"
        return f"Entry '{entry_id}' has been {mode}."

    async def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add an entity to the knowledge graph.

        Args:
            entity_id: Unique entity ID (e.g. 'company:spacex', 'person:elon').
            entity_type: Type (person, company, project, location, module).
            name: Human-readable name.
            metadata: Additional metadata.
        """
        self._store.add_entity(entity_id, entity_type, name, meta=metadata)
        return f"Entity '{entity_id}' ({entity_type}: {name}) added to graph."

    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        weight: float = 1.0,
    ) -> str:
        """Add a relationship between two entities in the graph.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.
            rel_type: Relationship type (WORKS_AT, DEPENDS_ON, LOCATED_IN, PART_OF, etc.).
            weight: Relation strength (0.0 to 1.0).
        """
        self._store.add_relation(source_id, target_id, rel_type, weight=weight)
        return f"Relation added: {source_id} --[{rel_type}, w={weight}]--> {target_id}"

    async def remove_relation(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
    ) -> str:
        """Remove a relationship between two entities.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.
            rel_type: Relationship type to remove.
        """
        self._store._exec(
            "DELETE FROM relations WHERE source_id=? AND target_id=? AND rel_type=?",
            (source_id, target_id, rel_type),
        )
        self._store._get_conn().commit()
        return f"Relation removed: {source_id} --[{rel_type}]--> {target_id}"

    async def combine_2_data_points(self, query1: str, query2: str) -> str:
        """Retrieve two data points by query, summarize them via LLM, and add the result.

        Args:
            query1: Search query to find the first data point.
            query2: Search query to find the second data point.
        """
        emb1 = await self._embed_fn(query1)
        emb2 = await self._embed_fn(query2)

        res1 = self._store.query(query_text=query1, query_embedding=emb1, k=1)
        res2 = self._store.query(query_text=query2, query_embedding=emb2, k=1)

        if not res1 or not res2:
            return "Could not retrieve one or both data points."

        text_to_combine = f"Point 1: {res1[0]['content']}\n\nPoint 2: {res2[0]['content']}"

        from toolboxv2 import get_app
        summary = await get_app().get_mod("isaa").mini_task_completion(
            mini_task="Combine the following two data points into a single, coherent text.",
            user_task=text_to_combine,
            agent_name="summary",
        )

        if hasattr(summary, "as_result"):
            summary = summary.as_result().get()
        if hasattr(summary, "get"):
            summary = summary.get()

        return await self.add_data_point(
            text=str(summary),
            content_type="fact",
            concepts=None,
            metadata={"source": "combination", "original_queries": [query1, query2]},
        )

    # ──────────────────────────────────────────────────────────────────
    # Analysis Set: Query and Inspection Tools
    # ──────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.2,
        content_types: Optional[list[str]] = None,
    ) -> str:
        """Search the memory using vector + BM25 + concept relation (RRF fusion).

        Args:
            query: The search query text.
            k: Number of results to return.
            min_similarity: Minimum similarity threshold.
            content_types: Filter by content types (text, code, fact, entity).
        """
        embedding = await self._embed_fn(query)
        results = self._store.query(
            query_text=query,
            query_embedding=embedding,
            k=k,
            search_modes=("vector", "bm25", "relation"),
            min_similarity=min_similarity,
            content_types=content_types,
        )
        return _format_query_results(results)

    async def search_by_concept(self, concept: str, k: int = 10) -> str:
        """Find all entries associated with a specific concept.

        Args:
            concept: The concept keyword to search for.
            k: Maximum number of results.
        """
        rows = self._store._exec(
            """
            SELECT ci.entry_id, e.content, e.content_type
            FROM concept_index ci
            JOIN entries e ON e.id = ci.entry_id
            WHERE ci.concept = ? AND e.is_active = 1 AND e.space = ?
            LIMIT ?
            """,
            (concept.lower(), self._store.space, k),
        ).fetchall()

        if not rows:
            return f"No entries found for concept '{concept}'."

        lines = [f"Entries for concept '{concept}' ({len(rows)} found):"]
        for row in rows:
            content_preview = row["content"][:200].replace("\n", " ")
            lines.append(f"  [{row['entry_id']}] ({row['content_type']}) {content_preview}")
        return "\n".join(lines)

    async def get_related_entities(
        self,
        entity_id: str,
        depth: int = 2,
        direction: str = "outgoing",
    ) -> str:
        """Get entities related to a given entity via graph traversal.

        Args:
            entity_id: The starting entity ID.
            depth: How many hops to traverse (1=direct, 2=2-hop).
            direction: 'outgoing' (default), 'incoming', or 'both'.
        """
        results = self._store.get_related(entity_id, depth=depth, direction=direction)
        return _format_entity_relations(results)

    async def get_entry_concepts(self, entry_id: str) -> str:
        """Get all concepts associated with a specific entry.

        Args:
            entry_id: The entry ID to look up.
        """
        concepts = self._store._load_concepts(entry_id)
        if not concepts:
            return f"No concepts found for entry '{entry_id}'."
        return f"Concepts for '{entry_id}': {', '.join(concepts)}"

    async def get_stats(self) -> str:
        """Get statistics about the current memory store (entries, entities, relations, concepts)."""
        stats = self._store.stats()
        lines = [
            "MEMORY STATS",
            "=" * 30,
            f"Active entries: {stats['active']}",
            f"Total entries:  {stats['total']}",
            f"Entities:       {stats['entities']}",
            f"Relations:      {stats['relations']}",
            f"Concepts:       {stats['concepts']}",
            f"FAISS vectors:  {stats['faiss_size']}",
            f"Space:          {stats['space']}",
            f"Embedding dim:  {stats['dim']}",
            f"With TTL:       {stats['with_ttl']}",
        ]
        return "\n".join(lines)

    async def list_concepts(self, limit: int = 50) -> str:
        """List the most frequent concepts in the memory store.

        Args:
            limit: Maximum number of concepts to return.
        """
        rows = self._store._exec(
            """
            SELECT ci.concept, COUNT(*) as cnt
            FROM concept_index ci
            JOIN entries e ON e.id = ci.entry_id
            WHERE e.is_active = 1 AND e.space = ?
            GROUP BY ci.concept
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (self._store.space, limit),
        ).fetchall()

        if not rows:
            return "No concepts found in memory."

        lines = [f"Top {len(rows)} concepts:"]
        for row in rows:
            lines.append(f"  {row['concept']} ({row['cnt']} entries)")
        return "\n".join(lines)

    def final_analysis(self, summary: str) -> str:
        """Signal the end of the analysis loop and provide the final summary.

        Args:
            summary: The final analysis summary text.
        """
        return f"FINAL ANALYSIS COMPLETE: {summary}"

    # ──────────────────────────────────────────────────────────────────
    # Orchestration Loop
    # ──────────────────────────────────────────────────────────────────

    async def start_analysis_loop(
        self,
        user_task: str,
        max_iterations: int = 10,
        agent_name: str = "summary",
    ) -> list[dict]:
        """
        Start the LLM-driven analysis loop.

        The agent will iteratively choose tools to call based on the task
        and accumulated history, until it calls final_analysis or hits
        max_iterations.

        Args:
            user_task: The user query or topic to analyze.
            max_iterations: Max number of tool calls before forcing stop.
            agent_name: The ISAA agent name to use for LLM calls.

        Returns:
            The complete analysis history (list of role/content dicts).
        """
        self.analysis_history = [{"role": "user", "content": user_task}]

        system_prompt = f"""You are an expert analysis agent working with a memory system.
Your goal is to analyze the user's topic using the available tools.
In each step, choose ONE tool to call to progress your analysis.
Base your decision on the user's request and the history of previous tool calls.
When you have gathered enough information, call `final_analysis` with your summary.

Available Tools:
{self._get_tool_signatures()}

Respond ONLY with a JSON object:
{{
  "tool_name": "name_of_the_tool_to_call",
  "parameters": {{ "param1": "value1" }}
}}

Valid tool names: {list(self.tools.keys())}
You MUST call final_analysis by iteration {max_iterations} at the latest.
"""

        for i in range(max_iterations):
            from toolboxv2 import get_app

            llm_response = await get_app().get_mod("isaa").mini_task_completion_format(
                mini_task=system_prompt,
                user_task=f"Analysis History:\n{json.dumps(self.analysis_history, indent=2, default=str)}",
                format_schema=ToolCall,
                agent_name=agent_name,
            )

            tool_name = llm_response.get("tool_name")
            parameters = llm_response.get("parameters", {})

            self.analysis_history.append({"role": "assistant", "content": llm_response})

            if tool_name in self.tools:
                tool_function = self.tools[tool_name]
                try:
                    if asyncio.iscoroutinefunction(tool_function):
                        result = await tool_function(**parameters)
                    else:
                        result = tool_function(**parameters)

                    self.analysis_history.append(
                        {"role": "tool", "content": {"tool": tool_name, "result": str(result)}}
                    )

                    if tool_name == "final_analysis":
                        break
                except Exception as e:
                    self.analysis_history.append(
                        {"role": "tool", "content": {"tool": tool_name, "error": str(e)}}
                    )
            else:
                self.analysis_history.append(
                    {"role": "tool", "content": {"error": f"Tool '{tool_name}' not found."}}
                )

        return self.analysis_history
