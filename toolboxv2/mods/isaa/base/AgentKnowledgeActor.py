# Fügen Sie diese Importe am Anfang Ihrer Datei hinzu, falls sie fehlen
import asyncio
import json
from typing import Any

from pydantic import Field, BaseModel
import inspect

from toolboxv2 import get_app
from toolboxv2.mods.isaa.base.KnowledgeBase import KnowledgeBase, RetrievalResult


# ... (Ihr gesamter bestehender Code von asyncio bis zum Ende der KnowledgeBase-Klasse) ...

# ====================================================================================
# NEUE AGENT-KLASSE ZUR DYNAMISCHEN TOOL-NUTZUNG
# ====================================================================================

class ToolCall(BaseModel):
    """Defines the structure for a tool call requested by the LLM."""
    tool_name: str = Field(..., description="The name of the tool to be executed.")
    parameters: dict[str, Any] = Field({}, description="The parameters to pass to the tool.")


class AgentKnowledge:
    """
    An agent that orchestrates the use of a KnowledgeBase by dynamically
    selecting tools in a loop using an LLM to analyze a given topic.
    """

    def __init__(self, kb: KnowledgeBase):
        """
        Initializes the agent with a KnowledgeBase instance.

        Args:
            kb (KnowledgeBase): An initialized KnowledgeBase object.
        """
        self.kb = kb
        self.analysis_history = []
        self._register_tools()

    def _register_tools(self):
        """Identifies and registers available tools from class methods."""
        self.tools = {}
        # Arbeits-Set (Manipulation der Wissensdatenbank)
        self.tools.update({
            "add_data_point": self.add_data_point,
            "remove_data_point": self.remove_data_point,
            "add_relation": self.add_relation,
            "remove_relation": self.remove_relation,
            "combine_2_data_points": self.combine_2_data_points,
        })
        # Analyse-Set (Analyse der Wissensdatenbank)
        self.tools.update({
            "retrieve_with_overview": self.kb.retrieve_with_overview,
            "get_largest_cluster_points": self.get_largest_cluster_points,
            "get_smallest_cluster_points": self.get_smallest_cluster_points,
            "get_single_points": self.get_single_points,
            "get_common_relations": self.get_common_relations,
            "get_uncommon_relations": self.get_uncommon_relations,
            "final_analysis": self.final_analysis,
        })

    def _get_tool_signatures(self) -> str:
        """Generates a formatted string of tool signatures for the LLM prompt."""
        signatures = []
        for name, func in self.tools.items():
            try:
                sig = inspect.signature(func)
                doc = inspect.getdoc(func) or "No description available."
                signatures.append(f"- {name}{sig}:\n  {doc.strip()}")
            except TypeError:
                # For methods that are not standard functions
                signatures.append(f"- {name}(...): No signature available.")
        return "\n".join(signatures)

    # ----------------------------------------------------------------------------------
    # Arbeits-Set: Tools zur Manipulation der Wissensdatenbank
    # ----------------------------------------------------------------------------------

    async def add_data_point(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Adds a new data point (chunk) to the Knowledge Base."""
        if metadata is None:
            metadata = {}
        added, duplicates = await self.kb.add_data([text], [metadata], direct=True)
        return f"Successfully added {added} new data point(s). Filtered {duplicates} duplicate(s)."

    async def remove_data_point(self, concept_to_remove: str) -> str:
        """Removes data points related to a specific concept."""
        removed_count = await self.kb.forget_irrelevant([concept_to_remove])
        return f"Removed {removed_count} data point(s) related to '{concept_to_remove}'."

    async def add_relation(self, source_concept: str, target_concept: str, relation_type: str) -> str:
        """Adds a new relationship between two concepts in the graph."""
        graph = self.kb.concept_extractor.concept_graph
        source = graph.concepts.get(source_concept.lower())
        if not source:
            return f"Error: Source concept '{source_concept}' not found."
        if relation_type not in source.relationships:
            source.relationships[relation_type] = set()
        source.relationships[relation_type].add(target_concept)
        return f"Successfully added relation: {source_concept} --[{relation_type}]--> {target_concept}"

    async def remove_relation(self, source_concept: str, target_concept: str, relation_type: str) -> str:
        """Removes a relationship between two concepts."""
        graph = self.kb.concept_extractor.concept_graph
        source = graph.concepts.get(source_concept.lower())
        if not source or relation_type not in source.relationships:
            return f"Error: No relation of type '{relation_type}' found for concept '{source_concept}'."
        if target_concept in source.relationships[relation_type]:
            source.relationships[relation_type].remove(target_concept)
            return f"Successfully removed relation: {source_concept} --[{relation_type}]--> {target_concept}"
        return f"Error: Target concept '{target_concept}' not found in relation."

    async def combine_2_data_points(self, query1: str, query2: str) -> str:
        """Retrieves two data points, summarizes them into a new one, and adds it to the KB."""
        res1 = await self.kb.retrieve(query1, k=1)
        res2 = await self.kb.retrieve(query2, k=1)
        if not res1 or not res2:
            return "Could not retrieve one or both data points."

        text_to_combine = f"Point 1: {res1[0].text}\n\nPoint 2: {res2[0].text}"

        from toolboxv2 import get_app
        summary_response = await get_app().get_mod("isaa").mini_task_completion(
            mini_task="Combine the following two data points into a single, coherent text.",
            user_task=text_to_combine,
            agent_name="summary"
        )

        await self.add_data_point(summary_response, {"source": "combination", "original_queries": [query1, query2]})
        return f"Successfully combined and added new data point: {summary_response[:100]}..."

    # ----------------------------------------------------------------------------------
    # Analyse-Set: Tools zur Analyse der Wissensdatenbank
    # ----------------------------------------------------------------------------------

    async def get_largest_cluster_points(self, query: str) -> dict:
        """Finds the largest topic cluster related to a query and returns its summary and main chunks."""
        results: RetrievalResult = await self.kb.retrieve_with_overview(query, k=10)
        if not results.overview:
            return {"error": "No topics found for this query."}
        largest_topic = max(results.overview, key=lambda x: x['chunk_count'])
        return largest_topic

    async def get_smallest_cluster_points(self, query: str) -> dict:
        """Finds the smallest (but not single-point) topic cluster related to a query."""
        results = await self.kb.retrieve_with_overview(query, k=10)
        non_single_topics = [t for t in results.overview if t['chunk_count'] > 1]
        if not non_single_topics:
            return {"error": "No multi-point clusters found."}
        smallest_topic = min(non_single_topics, key=lambda x: x['chunk_count'])
        return smallest_topic

    async def get_single_points(self, query: str) -> list[dict]:
        """Retrieves highly relevant individual data points (chunks) for a query."""
        results = await self.kb.retrieve(query, k=3, include_connected=False)
        return [{"text": chunk.text, "metadata": chunk.metadata} for chunk in results]

    async def get_common_relations(self, concept: str) -> dict:
        """Finds all relationships associated with a given concept."""
        concept_lower = concept.lower()
        if concept_lower not in self.kb.concept_extractor.concept_graph.concepts:
            return {"error": f"Concept '{concept}' not found."}
        relations = self.kb.concept_extractor.concept_graph.concepts[concept_lower].relationships
        return {k: list(v) for k, v in relations.items()}

    async def get_uncommon_relations(self, concept1: str, concept2: str) -> dict:
        """Finds relationships that one concept has but the other does not."""
        rels1 = await self.get_common_relations(concept1)
        rels2 = await self.get_common_relations(concept2)
        if "error" in rels1 or "error" in rels2:
            return {"error": "One or both concepts not found."}

        uncommon = {
            f"{concept1}_only": {k: v for k, v in rels1.items() if k not in rels2},
            f"{concept2}_only": {k: v for k, v in rels2.items() if k not in rels1}
        }
        return uncommon

    def final_analysis(self, summary: str) -> str:
        """
        Signals the end of the analysis loop and provides the final summary.
        This is a special tool that stops the loop.
        """
        return f"FINAL ANALYSIS COMPLETE: {summary}"

    # ----------------------------------------------------------------------------------
    # Orchestrierungs-Logik
    # ----------------------------------------------------------------------------------

    async def start_analysis_loop(self, user_task: str, max_iterations: int = 10) -> list:
        """
        Starts the dynamic analysis loop.

        Args:
            user_task (str): The initial user query or topic to analyze.
            max_iterations (int): The maximum number of tool calls to prevent infinite loops.

        Returns:
            list: The complete history of the analysis.
        """
        self.analysis_history = [{"role": "user", "content": user_task}]

        system_prompt = f"""
You are an expert analysis agent. Your goal is to analyze the user's topic using a knowledge base.
You have access to a set of tools. In each step, you must choose ONE tool to call to progress your analysis.
Base your decision on the user's request and the history of previous tool calls.
When you have gathered enough information and are ready to provide a final answer, call the `final_analysis` tool.

Available Tools:
{self._get_tool_signatures()}

Respond ONLY with a JSON object in the format:
{{
  "tool_name": "name_of_the_tool_to_call",
  "parameters": {{ "param1": "value1", "param2": "value2" }}
}}
"""

        for i in range(max_iterations):
            print(f"\n--- Iteration {i + 1}/{max_iterations} ---")

            # 1. Ask LLM for the next tool to use
            from toolboxv2 import get_app
            print(self.analysis_history)
            llm_response = await get_app().get_mod("isaa").mini_task_completion_format(
                mini_task=system_prompt,
                user_task=f"Analysis History:\n{json.dumps(self.analysis_history, indent=2)}",
                format_schema=ToolCall,
                agent_name="summary"
            )

            # 2. Execute the chosen tool
            tool_name = llm_response.get("tool_name")
            parameters = llm_response.get("parameters", {})
            print(f"Agent chose tool: {tool_name} with parameters: {parameters}")

            self.analysis_history.append({"role": "assistant", "content": llm_response})

            if tool_name in self.tools:
                tool_function = self.tools[tool_name]
                try:
                    # Check if the tool is async
                    if asyncio.iscoroutinefunction(tool_function):
                        result = await tool_function(**parameters)
                    else:
                        result = tool_function(**parameters)

                    self.analysis_history.append({"role": "tool", "content": {"result": result}})
                    print(f"Tool Result: {result}")

                    # Check for termination condition
                    if tool_name == "final_analysis":
                        print("\nAnalysis loop finished.")
                        break
                except Exception as e:
                    error_message = f"Error executing tool {tool_name}: {e}"
                    print(error_message)
                    self.analysis_history.append({"role": "tool", "content": {"error": error_message}})
            else:
                error_message = f"Tool '{tool_name}' not found."
                print(error_message)
                self.analysis_history.append({"role": "tool", "content": {"error": error_message}})

        return self.analysis_history


async def agent_main():
    """Example usage of the AgentKnowledge class."""
    # 1. Initialize the Knowledge Base and add some data
    print("Initializing Knowledge Base...")
    kb = KnowledgeBase(n_clusters=3, model_name="openrouter/mistralai/mistral-7b-instruct")

    initial_texts = [
        "Graph theory is the study of graphs, which are mathematical structures used to model pairwise relations between objects.",
        "A graph in this context is made up of vertices (also called nodes or points) which are connected by edges (also called links or lines).",
        "The Königsberg Bridge Problem is a famous historical problem in graph theory.",
        "Large Language Models (LLMs) are often based on the transformer architecture and are trained on massive amounts of text data.",
        "LLMs can be used for various tasks, including text generation, summarization, and analysis.",
        "Knowledge Graphs can be used to store information in a structured way, which can be beneficial for LLM performance and fact-checking."
    ]
    await kb.add_data(initial_texts, direct=True)
    print("Knowledge Base populated.")

    # 2. Initialize the Agent
    agent = AgentKnowledge(kb)

    # 3. Start the analysis loop with a user task
    user_query = "Analyze the relationship between Large Language Models and Graph Theory, and provide a summary of how they can be used together."
    print(f"\nStarting analysis for: '{user_query}'")

    final_history = await agent.start_analysis_loop(user_query)

    print("\n--- Final Analysis History ---")
    print(json.dumps(final_history, indent=2))


# Um dieses Beispiel auszuführen, ersetzen Sie den `if __name__ == "__main__":` Block
# in Ihrer Datei mit dem folgenden:
if __name__ == "__main__":
    get_app(name="agent_knowledge_test")
    asyncio.run(agent_main())
