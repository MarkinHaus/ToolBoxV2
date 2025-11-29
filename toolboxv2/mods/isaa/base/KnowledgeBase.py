import asyncio
import contextlib
import hashlib
import json
import math
import os
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import networkx as nx
import numpy as np
from pydantic import BaseModel


from toolboxv2 import Spinner, get_app, get_logger
from toolboxv2.mods.isaa.base.VectorStores import AbstractVectorStore
from toolboxv2.mods.isaa.base.VectorStores.FaissVectorStore import FaissVectorStore

@dataclass(slots=True)
class Chunk:
    """Represents a chunk of text with its embedding and metadata"""
    text: str
    embedding: np.ndarray
    metadata: dict[str, Any]
    content_hash: str
    cluster_id: int | None = None


@dataclass(slots=True)
class Chunk:
    """Represents a chunk of text with its embedding and metadata"""
    text: str
    embedding: np.ndarray
    metadata: dict[str, Any]
    content_hash: str
    cluster_id: int | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        # Zwei Chunks gelten als gleich, wenn sie denselben content_hash haben
        return self.content_hash == other.content_hash

    def __hash__(self) -> int:
        # Verwende nur content_hash, da embedding & metadata nicht hashbar sind
        return hash(self.content_hash)



@dataclass(slots=True)
class RetrievalResult:
    """Structure for organizing retrieval results"""
    overview: list[dict[str, Any]]          # List of topic summaries
    details: list["Chunk"]                  # Detailed chunks
    cross_references: dict[str, list["Chunk"]]  # Related chunks by topic

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary"""
        def chunk_to_dict(chunk):
            return {
                "text": chunk.text,
                "embedding": chunk.embedding.tolist() if isinstance(chunk.embedding, np.ndarray) else chunk.embedding,
                "metadata": chunk.metadata,
                "content_hash": chunk.content_hash,
                "cluster_id": chunk.cluster_id,
            }

        return {
            "overview": self.overview,
            "details": [chunk_to_dict(c) for c in self.details],
            "cross_references": {
                key: [chunk_to_dict(c) for c in val]
                for key, val in self.cross_references.items()
            }
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert the result to a JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

class TopicSummary(NamedTuple):
    topic_id: int
    summary: str
    key_chunks: list[Chunk]
    related_chunks: list[Chunk]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, where=norms != 0)


class rConcept(BaseModel):
    """
    Represents a key concept with its relationships and associated metadata.

    Attributes:
        name (str): The name of the concept.
        category (str): The category of the concept (e.g., 'technical', 'domain', 'method', etc.).
        relationships (Dict[str, List[str]]): A mapping where each key is a type of relationship and the
            value is a list of related concept names.
        importance_score (float): A numerical score representing the importance or relevance of the concept.
        context_snippets (List[str]): A list of text snippets providing context where the concept appears.
    """
    name: str
    category: str
    relationships: dict[str, list[str]]
    importance_score: float
    context_snippets: list[str]

@dataclass
class Concept:
    name: str
    category: str
    relationships: dict[str, set[str]]
    importance_score: float
    context_snippets: list[str]
    metadata: dict[str, Any]


class TConcept(BaseModel):
    """
    Represents the criteria or target parameters for concept selection and filtering.

    Attributes:
        min_importance (float): The minimum importance score a concept must have to be considered.
        target_concepts (List[str]): A list of names of target concepts to focus on.
        relationship_types (List[str]): A list of relationship types to be considered in the analysis.
        categories (List[str]): A list of concept categories to filter or group the concepts.
    """
    min_importance: float
    target_concepts: list[str]
    relationship_types: list[str]
    categories: list[str]


class Concepts(BaseModel):
    """
    Represents a collection of key concepts.

    Attributes:
        concepts (List[rConcept]): A list of Concept instances, each representing an individual key concept.
    """
    concepts: list[rConcept]

class ConceptAnalysis(BaseModel):
    """
    Represents the analysis of key concepts.

    Attributes:
        key_concepts (list[str]): A list of primary key concepts identified.
        relationships (list[str]): A list of relationships between the identified key concepts.
        importance_hierarchy (list[str]): A list that represents the hierarchical importance of the key concepts.
    """
    key_concepts: list[str]
    relationships: list[str]
    importance_hierarchy: list[str]


class TopicInsights(BaseModel):
    """
    Represents insights related to various topics.

    Attributes:
        primary_topics (list[str]): A list of main topics addressed.
        cross_references (list[str]): A list of cross-references that connect different topics.
        knowledge_gaps (list[str]): A list of identified gaps in the current knowledge.
    """
    primary_topics: list[str]
    cross_references: list[str]
    knowledge_gaps: list[str]


class RelevanceAssessment(BaseModel):
    """
    Represents an assessment of the relevance of the data in relation to a specific query.

    Attributes:
        query_alignment (float): A float representing the alignment between the query and the data.
        confidence_score (float): A float indicating the confidence level in the alignment.
        coverage_analysis (str): A textual description analyzing the data coverage.
    """
    query_alignment: float
    confidence_score: float
    coverage_analysis: str


class DataModel(BaseModel):
    """
    The main data model that encapsulates the overall analysis.

    Attributes:
        main_summary (str): A Detailed overview summarizing the key findings and relations format MD string.
        concept_analysis (ConceptAnalysis): An instance containing the analysis of key concepts.
        topic_insights (TopicInsights): An instance containing insights regarding the topics.
        relevance_assessment (RelevanceAssessment): An instance assessing the relevance and alignment of the query.
    """
    main_summary: str
    concept_analysis: ConceptAnalysis
    topic_insights: TopicInsights
    relevance_assessment: RelevanceAssessment

class ConceptGraph:
    """Manages concept relationships and hierarchies"""

    def __init__(self):
        self.concepts: dict[str, Concept] = {}

    def add_concept(self, concept: Concept):
        """Add or update a concept in the graph"""
        if concept.name.lower() in self.concepts:
            # Merge relationships and context
            existing = self.concepts[concept.name.lower()]
            for rel_type, related in concept.relationships.items():
                if rel_type not in existing.relationships:
                    existing.relationships[rel_type] = set()
                existing.relationships[rel_type].update(related)
            existing.context_snippets.extend(concept.context_snippets)
            # Update importance score with rolling average
            existing.importance_score = (existing.importance_score + concept.importance_score) / 2
        else:
            self.concepts[concept.name.lower()] = concept

    def get_related_concepts(self, concept_name: str, relationship_type: str | None = None) -> set[str]:
        """Get related concepts, optionally filtered by relationship type"""
        if concept_name not in self.concepts:
            return set()

        concept = self.concepts[concept_name.lower()]
        if relationship_type:
            return concept.relationships.get(relationship_type, set())

        related = set()
        for relations in concept.relationships.values():
            related.update(relations)
        return related


    def convert_to_networkx(self) -> nx.DiGraph:
        """Convert ConceptGraph to NetworkX graph with layout"""
        print(f"Converting to NetworkX graph with {len(self.concepts.values())} concepts")

        G = nx.DiGraph()

        if len(self.concepts.values()) == 0:
            return G

        for concept in self.concepts.values():
            cks = '\n - '.join(concept.context_snippets[:4])
            G.add_node(
                concept.name,
                size=concept.importance_score * 10,
                group=concept.category,
                title=f"""
                    {concept.name}
                    Category: {concept.category}
                    Importance: {concept.importance_score:.2f}
                    Context: \n - {cks}
                    """
            )

            for rel_type, targets in concept.relationships.items():
                for target in targets:
                    G.add_edge(concept.name, target, label=rel_type, title=rel_type)

        return G

class GraphVisualizer:
    @staticmethod
    def visualize(nx_graph: nx.DiGraph, output_file: str = "concept_graph.html", get_output=False):
        """Create interactive visualization using PyVis"""
        from pyvis.network import Network
        net = Network(
            height="800px",
            width="100%",
            notebook=False,
            directed=True,
            bgcolor="#1a1a1a",
            font_color="white"
        )

        net.from_nx(nx_graph)

        net.save_graph(output_file)
        print(f"Graph saved to {output_file} Open in browser to view.", len(nx_graph))
        if get_output:
            c = open(output_file, encoding="utf-8").read()
            os.remove(output_file)
            return c



class ConceptExtractor:
    """Handles extraction of concepts and relationships from text"""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.concept_graph = ConceptGraph()
        self._results_lock = asyncio.Lock()

    async def extract_concepts(self, texts: list[str], metadatas: list[dict[str, Any]]) -> list[list[Concept]]:
        """
        Extract concepts from texts using concurrent processing with rate limiting.
        Requests are made at the specified rate while responses are processed asynchronously.
        """
        # Ensure metadatas list matches texts length
        metadatas = metadatas + [{}] * (len(texts) - len(metadatas))

        # Initialize rate limiter

        system_prompt = (
            "Analyze the given text and extract key concepts and their relationships. For each concept:\n"
            "1. Identify the concept name and category (technical, domain, method, property, ...)\n"
            "2. Determine relationships with other concepts (uses, part_of, similar_to, depends_on, ...)\n"
            "3. Assess importance (0-1 score) based on centrality to the text\n"
            "4. Extract relevant context snippets\n"
            "5. Max 5 Concepts!\n"
            "only return in json format!\n"
            """{"concepts": [{
                "name": "concept_name",
                "category": "category_name",
                "relationships": {
                    "relationship_type": ["related_concept1", "related_concept2"]
                },
                "importance_score": 0.0,
                "context_snippets": ["relevant text snippet"]
            }]}\n"""
        )

        # Prepare all requests
        requests = [
            (idx, f"Text to Convert in to JSON structure:\n{text}", system_prompt, metadata)
            for idx, (text, metadata) in enumerate(zip(texts, metadatas, strict=False))
        ]

        async def process_single_request(idx: int, prompt: str, system_prompt: str, metadata: dict[str, Any]):
            """Process a single request with rate limiting"""
            try:
                # Wait for rate limit
                self.kb.stats['concept_calls'] += 1
                # Make API call without awaiting the response


                from toolboxv2 import get_app
                response_future = await get_app().get_mod("isaa").mini_task_completion_format(
                    mini_task=system_prompt,
                    user_task=prompt,
                    format_schema=Concepts,
                    agent_name="summary")

                return idx, response_future

            except Exception as e:
                print(f"Error initiating request {idx}: {str(e)}")
                return idx, None

        async def process_response(idx: int, response) -> list[Concept]:
            """Process the response once it's ready"""
            if response is None:
                return []

            return await self._process_response(response, metadatas[idx])

        request_tasks = []
        batch_size = self.kb.batch_size

        for batch_start in range(0, len(requests), batch_size):
            batch = requests[batch_start:batch_start + batch_size]

            # Create tasks for the batch
            batch_tasks = [
                process_single_request(idx, prompt, sys_prompt, meta)
                for idx, prompt, sys_prompt, meta in batch
            ]
            request_tasks.extend(batch_tasks)

        # Execute all requests with rate limiting
        request_results = await asyncio.gather(*request_tasks)

        # Process responses as they complete
        response_tasks = [
            process_response(idx, response_future)
            for idx, response_future in request_results
        ]

        # Gather all results
        all_results = await asyncio.gather(*response_tasks)

        # Sort results by original index
        sorted_results = [[] for _ in texts]
        for idx, concepts in enumerate(all_results):

            async with self._results_lock:
                sorted_results[idx] = concepts

        return sorted_results

    async def _process_response(self, concept_data: dict[str, Any], metadata: dict[str, Any]) -> list[Concept]:
        """Helper method to process a single response and convert it to Concepts"""
        try:
            concepts = []
            for concept_info in concept_data.get("concepts", []):
                concept = Concept(
                    name=concept_info["name"],
                    category=concept_info.get("category", "N/A"),
                    relationships={k: set(v) for k, v in concept_info.get("relationships", {}).items()},
                    importance_score=concept_info.get("importance_score", 0.1),
                    context_snippets=concept_info.get("context_snippets", "N/A"),
                    metadata=metadata
                )
                concepts.append(concept)
                self.concept_graph.add_concept(concept)

            return concepts

        except Exception:
            self.kb.stats['concept_errors'] += 1
            return []

    async def process_chunks(self, chunks: list[Chunk]) -> None:
        """
        Process all chunks in batch to extract and store concepts.
        Each chunk's metadata will be updated with the concept names and relationships.
        """
        # Gather all texts from the chunks.
        texts = [chunk.text for chunk in chunks]
        # Call extract_concepts once with all texts.
        all_concepts = await self.extract_concepts(texts, [chunk.metadata for chunk in chunks])

        # Update each chunk's metadata with its corresponding concepts.
        for chunk, concepts in zip(chunks, all_concepts, strict=False):
            chunk.metadata["concepts"] = [c.name for c in concepts]
            chunk.metadata["concept_relationships"] = {
                c.name: {k: list(v) for k, v in c.relationships.items()}
                for c in concepts
            }

    async def query_concepts(self, query: str) -> dict[str, any]:
        """Query the concept graph based on natural language query"""

        system_prompt = """
        Convert the natural language query about concepts into a structured format that specifies:
        1. Main concepts of interest
        2. Desired relationship types
        3. Any category filters
        4. Importance threshold

        Format as JSON.
        """

        prompt = f"""
        Query: {query}

        Convert to this JSON structure:
        {{
            "target_concepts": ["concept1", "concept2"],
            "relationship_types": ["type1", "type2"],
            "categories": ["category1", "category2"],
            "min_importance": 0.0
        }}
        """

        try:

            from toolboxv2 import get_app
            response = await get_app().get_mod("isaa").mini_task_completion_format(
                mini_task=system_prompt,
                user_task=prompt,
                format_schema=TConcept,
                agent_name="summary")

            query_params = response

            results = {
                "concepts": {},
                "relationships": [],
                "groups": []
            }

            # Find matching concepts
            for concept_name in query_params["target_concepts"]:
                if concept_name in self.concept_graph.concepts:
                    concept = self.concept_graph.concepts[concept_name]
                    if concept.importance_score >= query_params["min_importance"]:
                        results["concepts"][concept_name] = {
                            "category": concept.category,
                            "importance": concept.importance_score,
                            "context": concept.context_snippets
                        }

                        # Get relationships
                        for rel_type in query_params["relationship_types"]:
                            related = self.concept_graph.get_related_concepts(
                                concept_name, rel_type
                            )
                            for related_concept in related:
                                results["relationships"].append({
                                    "from": concept_name,
                                    "to": related_concept,
                                    "type": rel_type
                                })

            # Group concepts by category
            category_groups = defaultdict(list)
            for concept_name, concept_info in results["concepts"].items():
                category_groups[concept_info["category"]].append(concept_name)
            results["groups"] = [
                {"category": cat, "concepts": concepts}
                for cat, concepts in category_groups.items()
            ]

            return results

        except Exception as e:
            print(f"Error querying concepts: {str(e)}")
            return {"concepts": {}, "relationships": [], "groups": []}


class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 3600,
        chunk_overlap: int = 130,
        separator: str = "\n"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def approximate(self, text_len: int) -> float:
        """
        Approximate the number of chunks and average chunk size for a given text length

        Args:
            text_len (int): Length of the text to be split

        Returns:
            Tuple[int, int]: (number_of_chunks, approximate_chunk_size)
        """
        if text_len <= self.chunk_size:
            return 1, text_len

        # Handle extreme overlap cases
        if self.chunk_overlap >= self.chunk_size:
            estimated_chunks = text_len
            return estimated_chunks, 1

        # Calculate based on overlap ratio
        overlap_ratio = self.chunk_overlap / self.chunk_size
        base_chunks = text_len / self.chunk_size
        estimated_chunks = base_chunks * 2 / (overlap_ratio if overlap_ratio > 0 else 1)

        # print('#',estimated_chunks, base_chunks, overlap_ratio)
        # Calculate average chunk size
        avg_chunk_size = max(1, text_len / estimated_chunks)

        return estimated_chunks * avg_chunk_size

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks with overlap"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()

        # If text is shorter than chunk_size, return as is
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Find end of chunk
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to find a natural break point
            last_separator = text.rfind(self.separator, start, end)
            if last_separator != -1:
                end = last_separator

            # Add chunk
            chunks.append(text[start:end])

            # Calculate allowed overlap for this chunk
            chunk_length = end - start
            allowed_overlap = min(self.chunk_overlap, chunk_length - 1)

            # Move start position considering adjusted overlap
            start = end - allowed_overlap

        return chunks

class KnowledgeBase:
    def __init__(self, embedding_dim: int = 256, similarity_threshold: float = 0.61, batch_size: int = 12,
                 n_clusters: int = 4, deduplication_threshold: float = 0.85, model_name=os.getenv("SUMMARYMODEL"),
                 embedding_model=os.getenv("DEFAULTMODELEMBEDDING"),
                 vis_class:str | None = "FaissVectorStore",
                 vis_kwargs:dict[str, Any] | None=None,
                 chunk_size: int = 3600,
                 chunk_overlap: int = 130,
                 separator: str = "\n", **kwargs
                 ):
        """Initialize the knowledge base with given parameters"""

        self.existing_hashes: set[str] = set()
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.deduplication_threshold = deduplication_threshold
        if model_name == "openrouter/mistralai/mistral-nemo":
            batch_size = 9
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.model_name = model_name
        self.sto: list = []

        # Statistics tracking (replaces global i__ variable)
        self.stats = {
            'embeddings_generated': 0,
            'concept_calls': 0,
            'concept_errors': 0
        }

        self.text_splitter = TextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap, separator=separator)
        self.similarity_graph = {}
        self.concept_extractor = ConceptExtractor(self)

        self.vis_class = None
        self.vis_kwargs = None
        self.vdb = None
        self.init_vis(vis_class, vis_kwargs)

    def init_vis(self, vis_class, vis_kwargs):
        if vis_class is None:
            vis_class = "FaissVectorStore"
        if vis_class == "FaissVectorStore":
            if vis_kwargs is None:
                vis_kwargs = {
                    "dimension": self.embedding_dim
                }
            self.vdb = FaissVectorStore(**vis_kwargs)
        else:
            from toolboxv2.mods.isaa.base.VectorStores.taichiNumpyNumbaVectorStores import (
                EnhancedVectorStore,
                FastVectorStore1,
                FastVectorStoreO,
                NumpyVectorStore,
                VectorStoreConfig,
            )
        if vis_class == "FastVectorStoreO":
            if vis_kwargs is None:
                vis_kwargs = {
                    "embedding_size": self.embedding_dim
                }
            self.vdb = FastVectorStoreO(**vis_kwargs)
        if vis_class == "EnhancedVectorStore":
            if vis_kwargs is None:
                vis_kwargs = {
                    "dimension": self.embedding_dim
                }
            vis_kwargs = VectorStoreConfig(**vis_kwargs)
            self.vdb = EnhancedVectorStore(vis_kwargs)
        if vis_class == "FastVectorStore1":
            self.vdb = FastVectorStore1()
        if vis_class == "NumpyVectorStore":
            self.vdb = NumpyVectorStore()

        self.vis_class = vis_class
        self.vis_kwargs = vis_kwargs


    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute SHA-256 hash of text"""
        return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()

    async def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Get normalized embeddings in batches"""
        try:
            async def process_batch(batch: list[str]) -> np.ndarray:
                from toolboxv2.mods.isaa.extras.adapter import litellm_embed
                # print("Processing", batch)
                embeddings = await litellm_embed(texts=batch, model=self.embedding_model, dimensions=self.embedding_dim)
                return normalize_vectors(embeddings)

            tasks = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                tasks.append(process_batch(batch))

            embeddings = await asyncio.gather(*tasks)
            self.stats['embeddings_generated'] += len(texts)
            return np.vstack(embeddings)
        except Exception as e:
            get_logger().error(f"Error generating embeddings: {str(e)}")
            raise

    async def graph_enhanced_retrieve(
        self,
        query: str,
        k: int = 5,
        graph_hops: int = 2,
        relation_weight: float = 0.3,
        min_similarity: float = 0.2
    ) -> dict[str, Any]:
        """
        Kombiniert Vector-Search mit Graph-Traversierung

        Args:
            query: Suchanfrage
            k: Anzahl initial zu findender Chunks
            graph_hops: Tiefe der Graph-Traversierung
            relation_weight: Gewichtung Graph vs Vector (0-1)
            min_similarity: Minimale Ähnlichkeit für Vector-Suche

        Returns:
            Dict mit erweiterten Ergebnissen und Scores
        """
        # 1. Standard Vector-Suche
        query_embedding = (await self._get_embeddings([query]))[0]
        initial_chunks = await self.retrieve(
            query_embedding=query_embedding,
            k=k,
            min_similarity=min_similarity
        )

        if not initial_chunks:
            return {
                "chunks": [],
                "graph_expansion": {},
                "scores": {}
            }

        # 2. Graph-Expansion über Konzepte
        expanded_chunks = await self._expand_via_concepts(
            initial_chunks,
            hops=graph_hops
        )

        # 3. Hybrid-Scoring
        scored_results = self._hybrid_score(
            chunks=expanded_chunks,
            query_embedding=query_embedding,
            initial_chunks=initial_chunks,
            relation_weight=relation_weight
        )

        return scored_results

    async def _expand_via_concepts(
        self,
        chunks: list[Chunk],
        hops: int
    ) -> list[Chunk]:
        """
        Erweitert Chunks über Konzept-Relationen im Graph

        Args:
            chunks: Initial gefundene Chunks
            hops: Anzahl der Traversierungs-Schritte

        Returns:
            Liste erweiterter Chunks
        """
        expanded = set(chunks)
        current_concepts = set()

        # Sammle alle Konzepte aus initial chunks
        for chunk in chunks:
            current_concepts.update(chunk.metadata.get("concepts", []))

        # Traversiere Graph
        visited_concepts = set()
        for hop in range(hops):
            next_concepts = set()

            for concept_name in current_concepts:
                if concept_name in visited_concepts:
                    continue
                visited_concepts.add(concept_name)

                if concept_name.lower() in self.concept_extractor.concept_graph.concepts:
                    concept = self.concept_extractor.concept_graph.concepts[concept_name.lower()]

                    # Hole verwandte Konzepte aus allen Relationstypen
                    for rel_type, related in concept.relationships.items():
                        next_concepts.update(related)

            if not next_concepts:
                break

            # Finde Chunks mit diesen Konzepten
            for chunk in self.vdb.chunks:
                chunk_concepts = set(chunk.metadata.get("concepts", []))
                if chunk_concepts & next_concepts:
                    expanded.add(chunk)

            current_concepts = next_concepts

        return list(expanded)

    def _hybrid_score(
        self,
        chunks: list[Chunk],
        query_embedding: np.ndarray,
        initial_chunks: list[Chunk],
        relation_weight: float = 0.3
    ) -> dict[str, Any]:
        """
        Kombiniert Vector-Similarity mit Graph-basierten Scores

        Args:
            chunks: Alle zu scorenden Chunks
            query_embedding: Query-Embedding für Vector-Similarity
            initial_chunks: Initial gefundene Chunks (für Boost)
            relation_weight: Gewichtung Graph-Score (0-1)

        Returns:
            Dict mit gescorten Chunks und Metadaten
        """
        scored = []
        initial_chunk_ids = {id(chunk) for chunk in initial_chunks}

        for chunk in chunks:
            # 1. Vector Similarity
            vec_sim = float(np.dot(chunk.embedding, query_embedding))

            # 2. Graph Score: Anzahl und Qualität von Konzept-Verbindungen
            chunk_concepts = set(chunk.metadata.get("concepts", []))
            graph_score = 0.0
            relation_details = {}

            for concept_name in chunk_concepts:
                concept_name_lower = concept_name.lower()
                if concept_name_lower in self.concept_extractor.concept_graph.concepts:
                    concept = self.concept_extractor.concept_graph.concepts[concept_name_lower]

                    # Gewichte verschiedene Relationstypen unterschiedlich
                    weights = {
                        "depends_on": 2.0,
                        "uses": 1.5,
                        "part_of": 1.3,
                        "similar_to": 1.0,
                        "related_to": 0.8
                    }

                    for rel_type, related in concept.relationships.items():
                        weight = weights.get(rel_type, 1.0)
                        graph_score += len(related) * weight
                        relation_details[concept_name] = {
                            rel_type: list(related) for rel_type, related in concept.relationships.items()
                        }

            # Normalisiere Graph-Score
            graph_score = min(graph_score / 10.0, 1.0)

            # 3. Initial Chunk Boost
            initial_boost = 1.2 if id(chunk) in initial_chunk_ids else 1.0

            # 4. Hybrid Score berechnen
            final_score = (
                              (1 - relation_weight) * vec_sim +
                              relation_weight * graph_score
                          ) * initial_boost

            scored.append({
                "chunk": chunk,
                "score": final_score,
                "vec_similarity": vec_sim,
                "graph_score": graph_score,
                "is_initial": id(chunk) in initial_chunk_ids,
                "concepts": list(chunk_concepts),
                "relations": relation_details
            })

        # Sortiere nach Score
        scored.sort(key=lambda x: x["score"], reverse=True)

        return {
            "chunks": [item["chunk"] for item in scored],
            "detailed_scores": scored,
            "expansion_stats": {
                "initial_count": len(initial_chunks),
                "expanded_count": len(chunks),
                "expansion_ratio": len(chunks) / len(initial_chunks) if initial_chunks else 0
            }
        }

    def _remove_similar_chunks(self, threshold: float = None, batch_size: int = 1000) -> int:
        """
        Remove chunks that are too similar to each other using batch processing.

        This optimized version processes chunks in batches to avoid O(n²) memory usage.
        For large datasets (>10k chunks), this prevents memory exhaustion.

        Args:
            threshold: Similarity threshold for deduplication (default: self.deduplication_threshold)
            batch_size: Number of chunks to process at once (default: 1000)

        Returns:
            Number of chunks removed
        """
        if len(self.vdb.chunks) < 2:
            return 0

        if threshold is None:
            threshold = self.deduplication_threshold

        try:
            n = len(self.vdb.chunks)

            # For small datasets, use the original fast method
            if n <= batch_size:
                embeddings = np.vstack([c.embedding for c in self.vdb.chunks])
                similarities = np.dot(embeddings, embeddings.T)
                keep_mask = np.ones(n, dtype=bool)

                for i in range(n):
                    if not keep_mask[i]:
                        continue
                    similar_indices = similarities[i] >= threshold
                    similar_indices[i] = False
                    keep_mask[similar_indices] = False
            else:
                # For large datasets, use batch processing to save memory
                embeddings = np.vstack([c.embedding for c in self.vdb.chunks])
                keep_mask = np.ones(n, dtype=bool)

                # Process in batches to avoid full similarity matrix
                for i in range(0, n, batch_size):
                    if not any(keep_mask[i:i+batch_size]):
                        continue  # Skip if all in batch are already marked for removal

                    batch_end = min(i + batch_size, n)
                    batch_embeddings = embeddings[i:batch_end]

                    # Only compute similarities for this batch vs all chunks
                    batch_similarities = np.dot(batch_embeddings, embeddings.T)

                    # Process each chunk in the batch
                    for j in range(batch_end - i):
                        global_idx = i + j
                        if not keep_mask[global_idx]:
                            continue

                        # Find similar chunks
                        similar_indices = batch_similarities[j] >= threshold
                        similar_indices[global_idx] = False  # Don't count self-similarity

                        # Mark similar chunks for removal
                        keep_mask[similar_indices] = False

                    # Free memory
                    del batch_similarities

            # Keep only unique chunks
            unique_chunks = [chunk for chunk, keep in zip(self.vdb.chunks, keep_mask, strict=False) if keep]
            removed_count = len(self.vdb.chunks) - len(unique_chunks)

            # Update chunks and hashes
            self.vdb.chunks = unique_chunks
            self.existing_hashes = {chunk.content_hash for chunk in self.vdb.chunks}

            # Rebuild index if chunks were removed
            if removed_count > 0:
                self.vdb.rebuild_index()

            return removed_count

        except Exception as e:
            get_logger().error(f"Error removing similar chunks: {str(e)}")
            raise

    async def _add_data(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None= None,
    ) -> tuple[int, int]:
        """
        Process and add new data to the knowledge base.

        Optimized to avoid memory leaks:
        - Embeddings are computed only once for unique texts
        - Proper cleanup of intermediate data structures
        - Batch processing for large datasets

        Returns: Tuple of (added_count, duplicate_count)
        """
        if len(texts) == 0:
            return -1, -1
        try:
            # Compute hashes and filter exact duplicates
            hashes = [self.compute_hash(text) for text in texts]
            unique_data = []
            duplicate_count = 0

            for t, m, h in zip(texts, metadata, hashes, strict=False):
                if h in self.existing_hashes:
                    duplicate_count += 1
                    continue
                # Update existing hashes
                self.existing_hashes.add(h)
                unique_data.append((t, m, h))

            if not unique_data:
                return 0, len(texts)

            # Get embeddings ONLY for unique texts (FIX: avoid double computation)
            unique_texts = [t for t, m, h in unique_data]
            unique_embeddings = await self._get_embeddings(unique_texts)

            # Filter by similarity to existing chunks
            final_data = []
            final_embeddings = []
            similarity_filtered = 0

            if len(self.vdb.chunks):
                # Check each unique chunk against existing chunks
                for i, (t, m, h) in enumerate(unique_data):
                    similar_chunks = self.vdb.search(unique_embeddings[i], 5, self.deduplication_threshold)
                    if len(similar_chunks) > 2:
                        similarity_filtered += 1
                        continue
                    final_data.append((t, m, h))
                    final_embeddings.append(unique_embeddings[i])
            else:
                # No existing chunks, use all unique data
                final_data = unique_data
                final_embeddings = unique_embeddings

            # Clean up to free memory
            del unique_embeddings

            if not final_data:  # All were similar to existing chunks
                return 0, duplicate_count + similarity_filtered

            # Create new chunks
            new_chunks = [
                Chunk(text=t, embedding=e, metadata=m, content_hash=h)
                for (t, m, h), e in zip(final_data, final_embeddings, strict=False)
            ]

            # Add new chunks to vector store
            if new_chunks:
                all_embeddings = np.vstack(final_embeddings)
                self.vdb.add_embeddings(all_embeddings, new_chunks)

            # Remove similar chunks from the entire collection
            removed = self._remove_similar_chunks()
            get_logger().info(f"Removed {removed} similar chunks during deduplication")

            # Process new chunks for concepts (only if we have chunks after deduplication)
            chunks_to_process = len(new_chunks) - removed
            if chunks_to_process > 0:
                await self.concept_extractor.process_chunks(new_chunks)

            # Log statistics
            get_logger().debug(
                f"Stats - Embeddings: {self.stats['embeddings_generated']}, "
                f"Concept calls: {self.stats['concept_calls']}, "
                f"Concept errors: {self.stats['concept_errors']}"
            )

            return chunks_to_process, duplicate_count + similarity_filtered + removed

        except Exception as e:
            get_logger().error(f"Error adding data: {str(e)}")
            raise


    async def add_data(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None, direct:bool = False
    ) -> tuple[int, int]:
        """Enhanced version with smart splitting and clustering"""
        if isinstance(texts, str):
            texts = [texts]
        if metadata is None:
            metadata = [{}] * len(texts)
        if isinstance(metadata, dict):
            metadata = [metadata]
        if len(texts) != len(metadata):
            raise ValueError("Length of texts and metadata must match")

        # Filter ungültige Texte
        valid_texts = []
        valid_metadata = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue  # Skip leere Texte
            if len(text) > 1_000_000:
                raise ValueError(f"Text {i} too long: {len(text)} chars")
            valid_texts.append(text)
            valid_metadata.append(metadata[i] if metadata else {})

        if not valid_texts:
            return 0, 0


        texts = valid_texts
        metadata = valid_metadata

        if not direct and len(texts) == 1 and len(texts[0]) < 10_000:
            if len(self.sto) < self.batch_size and len(texts) == 1:
                self.sto.append((texts[0], metadata[0]))
                return -1, -1
            if len(self.sto) >= self.batch_size:
                _ = [texts.append(t) or metadata.append([m]) for (t, m) in self.sto]
                self.sto = []

        # Split large texts
        split_texts = []
        split_metadata = []

        while Spinner("Saving Data to Memory", symbols='t'):

            for idx, text in enumerate(texts):
                chunks = self.text_splitter.split_text(text)
                split_texts.extend(chunks)

                # Adjust metadata for splits
                meta = metadata[idx] if metadata else {}
                if isinstance(meta, list):
                    meta = meta[0]
                for i, _chunk in enumerate(chunks):
                    chunk_meta = meta.copy()
                    chunk_meta.update({
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'original_text_id': idx
                    })
                    split_metadata.append(chunk_meta)

            return await self._add_data(split_texts, split_metadata)

    def _update_similarity_graph(self, embeddings: np.ndarray, chunk_ids: list[int]):
        """Update similarity graph for connected information detection"""
        similarities = np.dot(embeddings, embeddings.T)

        for i in range(len(chunk_ids)):
            for j in range(i + 1, len(chunk_ids)):
                if similarities[i, j] >= self.similarity_threshold:
                    id1, id2 = chunk_ids[i], chunk_ids[j]
                    if id1 not in self.similarity_graph:
                        self.similarity_graph[id1] = set()
                    if id2 not in self.similarity_graph:
                        self.similarity_graph[id2] = set()
                    self.similarity_graph[id1].add(id2)
                    self.similarity_graph[id2].add(id1)

    async def retrieve(
        self,
        query: str="",
        query_embedding: np.ndarray | None = None,
        k: int = 5,
        min_similarity: float = 0.2,
        include_connected: bool = True
    ) -> list[Chunk]:
        """Enhanced retrieval with connected information"""
        if query_embedding is None:
            query_embedding = (await self._get_embeddings([query]))[0]
        k = min(k, len(self.vdb.chunks))
        if k <= 0:
            return []
        initial_results = self.vdb.search(query_embedding, k, min_similarity)

        if not include_connected or not initial_results:
            return initial_results

        # Find connected chunks
        connected_chunks = set()
        for chunk in initial_results:
            chunk_id = self.vdb.chunks.index(chunk)
            if chunk_id in self.similarity_graph:
                connected_chunks.update(self.similarity_graph[chunk_id])

        # Add connected chunks to results
        all_chunks = self.vdb.chunks
        additional_results = [all_chunks[i] for i in connected_chunks
                              if all_chunks[i] not in initial_results]

        # Sort by similarity to query
        all_results = initial_results + additional_results

        return sorted(
            all_results,
            key=lambda x: np.dot(x.embedding, query_embedding),
            reverse=True
        )[:k * 2]  # Return more results when including connected information

    async def forget_irrelevant(self, irrelevant_concepts: list[str], similarity_threshold: float | None=None) -> int:
        """
        Remove chunks similar to irrelevant concepts
        Returns: Number of chunks removed
        """
        if not irrelevant_concepts:
            return 0

        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        try:
            irrelevant_embeddings = await self._get_embeddings(irrelevant_concepts)
            initial_count = len(self.vdb.chunks)

            def is_relevant(chunk: Chunk) -> bool:
                similarities = np.dot(chunk.embedding, irrelevant_embeddings.T)
                do_keep = np.max(similarities) < similarity_threshold
                if do_keep:
                    return True
                for c in chunk.metadata.get("concepts", []):
                    if c in self.concept_extractor.concept_graph.concepts:
                        del self.concept_extractor.concept_graph.concepts[c]
                return False

            relevant_chunks = [chunk for chunk in self.vdb.chunks if is_relevant(chunk)]
            self.vdb.chunks = relevant_chunks
            self.existing_hashes = {chunk.content_hash for chunk in self.vdb.chunks}
            self.vdb.rebuild_index()

            return initial_count - len(self.vdb.chunks)

        except Exception as e:
            get_logger().error(f"Error forgetting irrelevant concepts: {str(e)}")
            raise

    ## ----------------------------------------------------------------

    def _cluster_chunks(
        self,
        chunks: list[Chunk],
        query_embedding: np.ndarray | None = None,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        max_clusters: int = 10
    ) -> dict[int, list[Chunk]]:
        """
        Enhanced clustering of chunks into topics with query awareness
        and dynamic parameter adjustment
        """
        if len(chunks) < 2:
            return {0: chunks}

        embeddings = np.vstack([chunk.embedding for chunk in chunks])

        # Normalize embeddings for cosine similarity
        embeddings = normalize_vectors(embeddings)

        # If query is provided, weight embeddings by query relevance
        if query_embedding is not None:
            query_similarities = np.dot(embeddings, query_embedding)
            # Apply soft weighting to maintain structure while considering query relevance
            embeddings = embeddings * query_similarities[:, np.newaxis]
            embeddings = normalize_vectors(embeddings)

        # Dynamic parameter adjustment based on dataset size
        adjusted_min_cluster_size = max(
            min_cluster_size,
            min(len(chunks) // 10, 5)  # Scale with data size, max 5
        )

        adjusted_min_samples = max(
            min_samples,
            adjusted_min_cluster_size // 2
        )

        # Try different parameter combinations for optimal clustering
        best_clusters = None
        best_score = float('-inf')

        epsilon_range = [0.2, 0.3, 0.4]
        try:
            HDBSCAN = __import__('sklearn.cluster').HDBSCAN
        except:
            print("install scikit-learn pip install scikit-learn for better results")
            return self._fallback_clustering(chunks, query_embedding)

        for epsilon in epsilon_range:
            clusterer = HDBSCAN(
                min_cluster_size=adjusted_min_cluster_size,
                min_samples=adjusted_min_samples,
                metric='cosine',
                cluster_selection_epsilon=epsilon
            )

            cluster_labels = clusterer.fit_predict(embeddings)

            # Skip if all points are noise
            if len(set(cluster_labels)) <= 1:
                continue

            # Calculate clustering quality metrics
            score = self._evaluate_clustering(
                embeddings,
                cluster_labels,
                query_embedding
            )

            if score > best_score:
                best_score = score
                best_clusters = cluster_labels

        # If no good clustering found, fall back to simpler approach
        if best_clusters is None:
            return self._fallback_clustering(chunks, query_embedding)

        # Organize chunks by cluster
        clusters: dict[int, list[Chunk]] = {}

        # Sort clusters by size and relevance
        cluster_scores = []

        for label in set(best_clusters):
            if label == -1:  # Handle noise points separately
                continue

            # Fixed: Use boolean mask to select chunks for current cluster
            cluster_mask = best_clusters == label
            cluster_chunks = [chunk for chunk, is_in_cluster in zip(chunks, cluster_mask, strict=False) if is_in_cluster]

            # Skip empty clusters
            if not cluster_chunks:
                continue

            # Calculate cluster score based on size and query relevance
            score = len(cluster_chunks)
            if query_embedding is not None:
                cluster_embeddings = np.vstack([c.embedding for c in cluster_chunks])
                query_relevance = np.mean(np.dot(cluster_embeddings, query_embedding))
                score = score * (1 + query_relevance)  # Boost by relevance

            cluster_scores.append((label, score, cluster_chunks))

        # Sort clusters by score and limit to max_clusters
        cluster_scores.sort(key=lambda x: x[1], reverse=True)

        # Assign cleaned clusters
        for i, (_, _, cluster_chunks) in enumerate(cluster_scores[:max_clusters]):
            clusters[i] = cluster_chunks

        # Handle noise points by assigning to nearest cluster
        noise_chunks = [chunk for chunk, label in zip(chunks, best_clusters, strict=False) if label == -1]
        if noise_chunks:
            self._assign_noise_points(noise_chunks, clusters, query_embedding)

        return clusters

    @staticmethod
    def _evaluate_clustering(
        embeddings: np.ndarray,
        labels: np.ndarray,
        query_embedding: np.ndarray | None = None
    ) -> float:
        """
        Evaluate clustering quality using multiple metrics
        """
        if len(set(labels)) <= 1:
            return float('-inf')

        # Calculate silhouette score for cluster cohesion
        try:
            sil_score = __import__('sklearn.metrics').silhouette_score(embeddings, labels, metric='cosine')
        except:
            print("install scikit-learn pip install scikit-learn for better results")
            sil_score = 0

        # Calculate Davies-Bouldin score for cluster separation
        try:
            db_score = -__import__('sklearn.metrics').davies_bouldin_score(embeddings, labels)  # Negated as lower is better
        except:
            print("install scikit-learn pip install scikit-learn for better results")
            db_score = 0

        # Calculate query relevance if provided
        query_score = 0
        if query_embedding is not None:
            unique_labels = set(labels) - {-1}
            if unique_labels:
                query_sims = []
                for label in unique_labels:
                    cluster_mask = labels == label
                    cluster_embeddings = embeddings[cluster_mask]
                    cluster_centroid = np.mean(cluster_embeddings, axis=0)
                    query_sims.append(np.dot(cluster_centroid, query_embedding))
                query_score = np.mean(query_sims)

        # Combine scores with weights
        combined_score = (
            0.4 * sil_score +
            0.3 * db_score +
            0.3 * query_score
        )

        return combined_score

    @staticmethod
    def _fallback_clustering(
        chunks: list[Chunk],
        query_embedding: np.ndarray | None = None
    ) -> dict[int, list[Chunk]]:
        """
        Simple fallback clustering when HDBSCAN fails
        """
        if query_embedding is not None:
            # Sort by query relevance
            chunks_with_scores = [
                (chunk, np.dot(chunk.embedding, query_embedding))
                for chunk in chunks
            ]
            chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
            chunks = [c for c, _ in chunks_with_scores]

        # Create fixed-size clusters
        clusters = {}
        cluster_size = max(2, len(chunks) // 5)

        for i in range(0, len(chunks), cluster_size):
            clusters[len(clusters)] = chunks[i:i + cluster_size]

        return clusters

    @staticmethod
    def _assign_noise_points(
        noise_chunks: list[Chunk],
        clusters: dict[int, list[Chunk]],
        query_embedding: np.ndarray | None = None
    ) -> None:
        """
        Assign noise points to nearest clusters
        """
        if not clusters:
            clusters[0] = noise_chunks
            return

        for chunk in noise_chunks:
            best_cluster = None
            best_similarity = float('-inf')

            for cluster_id, cluster_chunks in clusters.items():
                cluster_embeddings = np.vstack([c.embedding for c in cluster_chunks])
                cluster_centroid = np.mean(cluster_embeddings, axis=0)

                similarity = np.dot(chunk.embedding, cluster_centroid)

                # Consider query relevance in assignment if available
                if query_embedding is not None:
                    query_sim = np.dot(chunk.embedding, query_embedding)
                    similarity = 0.7 * similarity + 0.3 * query_sim

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_id

            if best_cluster is not None:
                clusters[best_cluster].append(chunk)

    @staticmethod
    def _generate_topic_summary(
        chunks: list[Chunk],
        query_embedding: np.ndarray,
        max_sentences=3
    ) -> str:
        """Generate a summary for a topic using most representative chunks"""
        if not chunks:
            return ""

        # Find chunks most similar to cluster centroid
        embeddings = np.vstack([chunk.embedding for chunk in chunks])
        centroid = embeddings.mean(axis=0)

        # Calculate similarities to both centroid and query
        centroid_sims = np.dot(embeddings, centroid)
        query_sims = np.dot(embeddings, query_embedding)

        # Combine both similarities
        combined_sims = 0.7 * centroid_sims + 0.3 * query_sims

        # Select top sentences from most representative chunks
        top_indices = np.argsort(combined_sims)[-max_sentences:]
        summary_chunks = [chunks[i] for i in top_indices]

        # Extract key sentences
        sentences = []
        for chunk in summary_chunks:
            sentences.extend(sent.strip() for sent in chunk.text.split('.') if sent.strip())

        return '. '.join(sentences[:max_sentences]) + '.'

    async def retrieve_with_overview(
        self,
        query: str,
        query_embedding=None,
        k: int = 5,
        min_similarity: float = 0.2,
        max_sentences: int = 5,
        cross_ref_depth: int = 2,
        max_cross_refs: int = 10,
        use_graph_expansion: bool = True,  # NEU
        graph_hops: int = 2,  # NEU
        relation_weight: float = 0.3  # NEU
    ) -> RetrievalResult:
        """
        Enhanced retrieval mit Graph-Awareness und better cross-reference handling

        Args:
            use_graph_expansion: Nutze Graph-basierte Expansion (empfohlen)
            graph_hops: Tiefe der Graph-Traversierung
            relation_weight: Gewichtung Graph vs Vector (0-1)
        """
        # Get initial results with query embedding
        if query_embedding is None:
            query_embedding = (await self._get_embeddings([query]))[0]

        # ========== NEU: Wähle Retrieval-Methode ==========
        if use_graph_expansion:
            # Nutze Graph-Enhanced Retrieval
            graph_results = await self.graph_enhanced_retrieve(
                query=query,
                k=k,
                graph_hops=graph_hops,
                relation_weight=relation_weight,
                min_similarity=min_similarity
            )
            initial_results = graph_results["chunks"][:k * 2]
            all_relevant_chunks = graph_results["chunks"]
        else:
            # Standard Vector-Retrieval
            initial_results = await self.retrieve(
                query_embedding=query_embedding,
                k=k,
                min_similarity=min_similarity
            )

            if not initial_results:
                return RetrievalResult([], [], {})

            # Find cross-references (alte Methode)
            initial_ids = {self.vdb.chunks.index(chunk) for chunk in initial_results}
            related_ids = self._find_cross_references(
                initial_ids,
                depth=cross_ref_depth,
                query_embedding=query_embedding
            )

            all_chunks = self.vdb.chunks
            all_relevant_chunks = initial_results + [
                chunk for i, chunk in enumerate(all_chunks)
                if i in related_ids and self._is_relevant_cross_ref(
                    chunk,
                    query_embedding,
                    initial_results
                )
            ]
        # ========== ENDE NEU ==========

        # Enhanced clustering with dynamic cluster size
        clusters = self._cluster_chunks(
            all_relevant_chunks,
            query_embedding=query_embedding
        )

        # Fallback: If no clusters are found, treat all relevant chunks as a single cluster.
        if not clusters:
            print("No clusters found. Falling back to using all relevant chunks as a single cluster.")
            clusters = {0: all_relevant_chunks}

        # Generate summaries and organize results
        overview = []
        cross_references = {}

        for cluster_id, cluster_chunks in clusters.items():
            summary = self._generate_topic_summary(
                cluster_chunks,
                query_embedding,
                max_sentences=max_sentences
            )

            # Enhanced chunk sorting with combined scoring
            sorted_chunks = self._sort_chunks_by_relevance(
                cluster_chunks,
                query_embedding,
                initial_results
            )

            # Separate direct matches and cross-references
            direct_matches_ = [{'text': c.text, 'metadata': c.metadata} for c in sorted_chunks if c in initial_results]
            direct_matches = []
            for match in direct_matches_:
                if match in direct_matches:
                    continue
                direct_matches.append(match)
            cross_refs_ = [c for c in sorted_chunks if c not in initial_results]
            cross_refs = []
            for match in cross_refs_:
                if match in cross_refs:
                    continue
                cross_refs.append(match)

            # Limit cross-references while maintaining diversity
            selected_cross_refs = self._select_diverse_cross_refs(
                cross_refs,
                max_cross_refs,
                query_embedding
            )

            topic_info = {
                'topic_id': cluster_id,
                'summary': summary,
                'main_chunks': [x for x in direct_matches[:3]],
                'chunk_count': len(cluster_chunks),
                'relevance_score': self._calculate_topic_relevance(
                    cluster_chunks,
                    query_embedding
                )
            }
            overview.append(topic_info)

            if selected_cross_refs:
                cross_references[f"topic_{cluster_id}"] = selected_cross_refs

        # Sort overview by relevance score
        overview.sort(key=lambda x: x['relevance_score'], reverse=True)

        return RetrievalResult(
            overview=overview,
            details=initial_results,
            cross_references=cross_references
        )

    def _find_cross_references(
        self,
        chunk_ids: set[int],
        depth: int,
        query_embedding: np.ndarray
    ) -> set[int]:
        """Enhanced cross-reference finding with relevance scoring"""
        related_ids = set(chunk_ids)
        current_depth = 0
        frontier = set(chunk_ids)

        while current_depth < depth and frontier:
            new_frontier = set()
            for chunk_id in frontier:
                if chunk_id in self.similarity_graph:
                    # Score potential cross-references by relevance
                    candidates = self.similarity_graph[chunk_id] - related_ids
                    scored_candidates = [
                        (cid, self._calculate_topic_relevance(
                            [self.vdb.chunks[cid]],
                            query_embedding
                        ))
                        for cid in candidates
                    ]

                    # Filter by relevance threshold
                    relevant_candidates = {
                        cid for cid, score in scored_candidates
                        if score > 0.5  # Adjustable threshold
                    }
                    new_frontier.update(relevant_candidates)

            related_ids.update(new_frontier)
            frontier = new_frontier
            current_depth += 1

        return related_ids

    @staticmethod
    def _is_relevant_cross_ref(
        chunk: Chunk,
        query_embedding: np.ndarray,
        initial_results: list[Chunk]
    ) -> bool:
        """Determine if a cross-reference is relevant enough to include"""
        # Calculate similarity to query
        query_similarity = np.dot(chunk.embedding, query_embedding)

        # Calculate similarity to initial results
        initial_similarities = [
            np.dot(chunk.embedding, r.embedding) for r in initial_results
        ]
        max_initial_similarity = max(initial_similarities)

        # Combined relevance score
        relevance_score = 0.7 * query_similarity + 0.3 * max_initial_similarity

        return relevance_score > 0.6  # Adjustable threshold

    @staticmethod
    def _select_diverse_cross_refs(
        cross_refs: list[Chunk],
        max_count: int,
        query_embedding: np.ndarray
    ) -> list[Chunk]:
        """Select diverse and relevant cross-references"""
        if not cross_refs or len(cross_refs) <= max_count:
            return cross_refs

        # Calculate diversity scores
        embeddings = np.vstack([c.embedding for c in cross_refs])
        similarities = np.dot(embeddings, embeddings.T)

        selected = []
        remaining = list(enumerate(cross_refs))

        while len(selected) < max_count and remaining:
            # Score remaining chunks by relevance and diversity
            scores = []
            for idx, chunk in remaining:
                relevance = np.dot(chunk.embedding, query_embedding)
                diversity = 1.0
                if selected:
                    # Calculate diversity penalty based on similarity to selected chunks
                    selected_similarities = [
                        similarities[idx][list(cross_refs).index(s)]
                        for s in selected
                    ]
                    diversity = 1.0 - max(selected_similarities)

                combined_score = 0.7 * relevance + 0.3 * diversity
                scores.append((combined_score, idx, chunk))

            # Select the highest scoring chunk
            scores.sort(reverse=True)
            _, idx, chunk = scores[0]
            selected.append(chunk)
            remaining = [(i, c) for i, c in remaining if i != idx]

        return selected

    @staticmethod
    def _calculate_topic_relevance(
        chunks: list[Chunk],
        query_embedding: np.ndarray,
    ) -> float:
        """Calculate overall topic relevance score"""
        if not chunks:
            return 0.0

        similarities = [
            np.dot(chunk.embedding, query_embedding) for chunk in chunks
        ]
        return np.mean(similarities)

    @staticmethod
    def _sort_chunks_by_relevance(
        chunks: list[Chunk],
        query_embedding: np.ndarray,
        initial_results: list[Chunk]
    ) -> list[Chunk]:
        """Sort chunks by combined relevance score"""
        scored_chunks = []
        for chunk in chunks:
            query_similarity = np.dot(chunk.embedding, query_embedding)
            initial_similarities = [
                np.dot(chunk.embedding, r.embedding)
                for r in initial_results
            ]
            max_initial_similarity = max(initial_similarities) if initial_similarities else 0

            # Combined score favoring query relevance
            combined_score = 0.7 * query_similarity + 0.3 * max_initial_similarity
            scored_chunks.append((combined_score, chunk))

        scored_chunks.sort(reverse=True)
        return [chunk for _, chunk in scored_chunks]

    async def query_concepts(self, query: str) -> dict[str, any]:
        """Query concepts extracted from the knowledge base"""
        return await self.concept_extractor.query_concepts(query)

    async def unified_retrieve(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.2,
        cross_ref_depth: int = 2,
        max_cross_refs: int = 10,
        max_sentences: int = 10,
        use_graph_expansion: bool = True,
        graph_hops: int = 2,
        relation_weight: float = 0.3
    ) -> dict[str, Any]:
        """
        Unified retrieval mit optionaler Graph-Expansion

        Args:
            query: Suchanfrage
            k: Anzahl Primär-Ergebnisse
            min_similarity: Min. Ähnlichkeit für Vector-Suche
            cross_ref_depth: Tiefe für Cross-References
            max_cross_refs: Max. Cross-References pro Topic
            max_sentences: Max. Sentences im Summary
            use_graph_expansion: Nutze Graph-Expansion (NEU)
            graph_hops: Graph-Traversierungs-Tiefe (NEU)
            relation_weight: Graph vs Vector Gewichtung (NEU)

        Returns:
            Dict mit umfassenden Ergebnissen
        """
        # Get concept information
        concept_results = await self.concept_extractor.query_concepts(query)

        query_embedding = (await self._get_embeddings([query]))[0]

        # Wähle Retrieval-Methode
        if use_graph_expansion:
            graph_results = await self.graph_enhanced_retrieve(
                query=query,
                k=k,
                graph_hops=graph_hops,
                relation_weight=relation_weight,
                min_similarity=min_similarity
            )
            basic_results = graph_results["chunks"][:k * 2]
            expansion_stats = graph_results.get("expansion_stats", {})
        else:
            basic_results = await self.retrieve(
                query_embedding=query_embedding,
                k=k,
                min_similarity=min_similarity
            )
            expansion_stats = {}

        if len(basic_results) == 0:
            return {}
        if len(basic_results) == 1 and isinstance(basic_results[0], str) and basic_results[0].endswith(
            '[]\n - []\n - []'):
            return {}

        # Get retrieval overview
        overview_results = await self.retrieve_with_overview(
            query=query,
            query_embedding=query_embedding,
            k=k,
            min_similarity=min_similarity,
            cross_ref_depth=cross_ref_depth,
            max_cross_refs=max_cross_refs,
            max_sentences=max_sentences
        )

        # Prepare context for LLM summary
        context = {
            "concepts": {
                "main_concepts": concept_results.get("concepts", {}),
                "relationships": concept_results.get("relationships", []),
                "concept_groups": concept_results.get("groups", [])
            },
            "topics": [
                {
                    "id": topic["topic_id"],
                    "summary": topic["summary"],
                    "relevance": topic["relevance_score"],
                    "chunk_count": topic["chunk_count"]
                }
                for topic in overview_results.overview
            ],
            "key_chunks": [
                {
                    "text": chunk.text,
                    "metadata": chunk.metadata
                }
                for chunk in basic_results[:k]
            ],
            "graph_expansion": expansion_stats
        }

        # Generate comprehensive summary using LLM
        system_prompt = """
        Analyze the provided search results and generate a comprehensive summary
        that includes:
        1. Main concepts and their relationships
        2. Key topics and their relevance
        3. Most important findings and insights
        4. Cross-references and connections between topics
        5. Potential gaps or areas for further investigation

        Format the response as a JSON object with these sections.
        """

        prompt = f"""
        Query: {query}

        Context:
        {json.dumps(context, indent=2)}

        Generate a comprehensive analysis and summary following the structure:
        """

        try:
            from toolboxv2 import get_app
            llm_response = await get_app().get_mod("isaa").mini_task_completion_format(
                mini_task=system_prompt,
                user_task=prompt,
                format_schema=DataModel,
                agent_name="summary")
            summary_analysis = llm_response
        except Exception as e:
            get_logger().error(f"Error generating summary: {str(e)}")
            summary_analysis = {
                "main_summary": "Error generating summary",
                "error": str(e)
            }
            raise e

        # Compile final results
        return {
            "summary": summary_analysis,
            "raw_results": {
                "concepts": concept_results,
                "overview": {
                    "topics": overview_results.overview,
                    "cross_references": overview_results.cross_references
                },
                "relevant_chunks": [
                    {
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "cluster_id": chunk.cluster_id
                    }
                    for chunk in basic_results[:k * 2]
                ]
            },
            "metadata": {
                "query": query,
                "timestamp": time.time(),
                "retrieval_params": {
                    "k": k,
                    "min_similarity": min_similarity,
                    "cross_ref_depth": cross_ref_depth,
                    "max_cross_refs": max_cross_refs,
                    "use_graph_expansion": use_graph_expansion,
                    "graph_hops": graph_hops,
                    "relation_weight": relation_weight
                },
                "expansion_stats": expansion_stats
            }
        }

    def save(self, path: str) -> bytes | None:
        """
        Save the complete knowledge base to disk, including all sub-components

        Args:
            path (str): Path where the knowledge base will be saved
        """
        try:
            data = {
                # Core components
                'vdb': self.vdb.save(),
                'vis_kwargs': self.vis_kwargs,
                'vis_class': self.vis_class,
                'existing_hashes': self.existing_hashes,

                # Configuration parameters
                'embedding_dim': self.embedding_dim,
                'similarity_threshold': self.similarity_threshold,
                'batch_size': self.batch_size,
                'n_clusters': self.n_clusters,
                'deduplication_threshold': self.deduplication_threshold,
                'model_name': self.model_name,
                'embedding_model': self.embedding_model,

                # Cache and graph data
                'similarity_graph': self.similarity_graph,
                'sto': self.sto,

                # Text splitter configuration
                'text_splitter_config': {
                    'chunk_size': self.text_splitter.chunk_size,
                    'chunk_overlap': self.text_splitter.chunk_overlap,
                    'separator': self.text_splitter.separator
                },

                # Concept extractor data
                'concept_graph': {
                    'concepts': {
                        name: {
                            'name': concept.name,
                            'category': concept.category,
                            'relationships': {k: list(v) for k, v in concept.relationships.items()},
                            'importance_score': concept.importance_score,
                            'context_snippets': concept.context_snippets,
                            'metadata': concept.metadata
                        }
                        for name, concept in self.concept_extractor.concept_graph.concepts.items()
                    }
                }
            }
            b = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

            if path is None:
                return b

            path = Path(path)
            tmp = path.with_suffix(path.suffix + ".tmp") if path.suffix else path.with_name(path.name + ".tmp")

            try:
                # Schreibe zuerst in eine temporäre Datei
                with open(tmp, "wb") as f:
                    f.write(b)
                    f.flush()
                    os.fsync(f.fileno())  # sicherstellen, dass die Daten auf Platte sind
                # Atomischer Austausch
                os.replace(tmp, path)
            finally:
                # Aufräumen falls tmp noch existiert (bei Fehlern)
                if tmp.exists():
                    with contextlib.suppress(Exception):
                        tmp.unlink()
            return None
            # print(f"Knowledge base successfully saved to {path} with {len(self.concept_extractor.concept_graph.concepts.items())} concepts")

        except Exception as e:
            print(f"Error saving knowledge base: {str(e)}")
            raise
    def init_vdb(self, db:AbstractVectorStore=AbstractVectorStore):
        pass
    @classmethod
    def load(cls, path: str | bytes) -> 'KnowledgeBase':
        """
        Load a complete knowledge base from disk, including all sub-components

        Args:
            path (str): Path from where to load the knowledge base

        Returns:
            KnowledgeBase: A fully restored knowledge base instance
        """
        try:
            if isinstance(path, bytes | bytearray | memoryview):
                data_bytes = bytes(path)
                try:
                    data = pickle.loads(data_bytes)
                except Exception as e:
                    raise EOFError(f"Fehler beim pickle.loads von bytes: {e}") from e
            else:
                p = Path(path)
                if not p.exists():
                    raise FileNotFoundError(f"{p} existiert nicht")
                size = p.stat().st_size
                if size == 0:
                    raise EOFError(f"{p} ist leer (0 bytes)")
                try:
                    with open(p, "rb") as f:
                        try:
                            data = pickle.load(f)
                        except EOFError as e:
                            # Debug info: erste bytes ausgeben
                            f.seek(0)
                            snippet = f.read(128)
                            raise EOFError(
                                f"EOFError beim Laden {p} (Größe {size} bytes). Erste 128 bytes: {snippet!r}") from e

                except Exception as e:
                    raise ValueError(f"Invalid path type {e}") from e
            # Create new knowledge base instance with saved configuration
            kb = cls(
                embedding_dim=data['embedding_dim'],
                similarity_threshold=data['similarity_threshold'],
                batch_size=data['batch_size'],
                n_clusters=data['n_clusters'],
                deduplication_threshold=data['deduplication_threshold'],
                model_name=data['model_name'],
                embedding_model=data['embedding_model']
            )

            # Restore core components
            kb.init_vis(data.get('vis_class'), data.get('vis_kwargs'))
            kb.vdb.load(data['vdb'])
            kb.existing_hashes = data['existing_hashes']

            # Restore cache and graph data
            kb.similarity_graph = data.get('similarity_graph', {})
            kb.sto = data.get('sto', [])

            # Restore text splitter configuration
            splitter_config = data.get('text_splitter_config', {})
            kb.text_splitter = TextSplitter(
                chunk_size=splitter_config.get('chunk_size', 12_000),
                chunk_overlap=splitter_config.get('chunk_overlap', 200),
                separator=splitter_config.get('separator', '\n')
            )

            # Restore concept graph
            concept_data = data.get('concept_graph', {}).get('concepts', {})
            for concept_info in concept_data.values():
                concept = Concept(
                    name=concept_info['name'],
                    category=concept_info['category'],
                    relationships={k: set(v) for k, v in concept_info['relationships'].items()},
                    importance_score=concept_info['importance_score'],
                    context_snippets=concept_info['context_snippets'],
                    metadata=concept_info['metadata']
                )
                kb.concept_extractor.concept_graph.add_concept(concept)

            # print(f"Knowledge base successfully loaded from {path} with {len(concept_data)} concepts")
            return kb

        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")
            import traceback
            traceback.print_exception(e)
            raise

    async def vis(self,output_file: str = "concept_graph.html", get_output_html=False, get_output_net=False):

        if not self.concept_extractor.concept_graph.concepts:

            if len(self.sto) > 2:
                await self.add_data([t for (t, m) in self.sto], [m for (t, m) in self.sto], direct=True)
                # self.sto = []
            if not self.concept_extractor.concept_graph.concepts:
                print("NO Concepts defined and no data in sto")
                return None


        net = self.concept_extractor.concept_graph.convert_to_networkx()
        if get_output_net:
            return net
        return GraphVisualizer.visualize(net, output_file=output_file, get_output=get_output_html)

async def main():
    kb = KnowledgeBase(n_clusters=3, model_name="openrouter/mistralai/mistral-7b-instruct")

    # Generate test data
    texts = [
                "The quick brown fox jumps over the lazy dog",
                "Machine learning is a subset of artificial intelligence",
                "Python is a popular programming language",
                "Python is The popular programming language",
                "Deep learning models require significant computational resources",
                "Natural language processing helps computers understand human language",
                """
        Machine learning models require significant computational resources.
        GPUs are often used to accelerate training of deep neural networks.
        The training process involves optimizing model parameters.
        """,
        """
        Neural networks consist of layers of interconnected nodes.
        Each node processes input data using activation functions.
        Deep learning networks have multiple hidden layers.
        """,
        """
        GPUs are specialized processors designed for parallel computation.
        They excel at matrix operations common in machine learning.
        Modern GPUs have thousands of cores for parallel processing.
        """,
        """
        Training data quality is crucial for machine learning success.
        Data preprocessing includes cleaning and normalization steps.
        Feature engineering helps improve model performance.
        """
            ] * 20  # Create more data for testing

    metadata = [{"source": f"example{i}", "timestamp": time.time()}
                for i in range(len(texts))]

    # Benchmark operations
    async def benchmark(name, coro):
        start = time.perf_counter()
        result = await coro
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed:.2f} seconds")
        return result, elapsed

    # Run operations
    t0 = 0
    #for i, (t,m )in enumerate(zip(texts, metadata)):
    _, t = await benchmark(f"Adding data ({0})", kb.add_data(texts, metadata))
    t0 += t
    print("Total time from 7.57 to: 5.5 (one by one) ", t0, _)
    #await benchmark("Forgetting irrelevant",
    #                kb.forget_irrelevant(["lazy", "unimportant"]))
    results_, _ = await benchmark("Retrieving",
                              kb.retrieve("machine learning", k=2))
    #results_v = await benchmark("generate_visualization_data",
    #                          kb.generate_visualization_data())

    #print(results_v)

    print("\nRetrieval results:")
    for chunk in results_:
        print(f"\nText: {chunk.text}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Cluster: {chunk.cluster_id}")

    results, _ = await benchmark("Retrieving with_overview", kb.retrieve_with_overview(
        "GPU computing in machine learning",
        k=6
    ))
    print("\nOverview of Topics:")
    for topic in results.overview:
        print(f"\nTopic {topic['topic_id']}:")
        print(f"Summary: {topic['summary']}")
        print(f"Number of related chunks: {topic['chunk_count']}")

    print("\nDetailed Results:")
    for chunk in results.details:
        print(f"\nText: {chunk.text}")

    print("\nCross References:")
    for topic, refs in results.cross_references.items():
        print(f"\n{topic}:")
        for ref in refs:
            print(f"- {ref.text[:100]}...")

    results, _ = await benchmark("Retrieving unified_retrieve", kb.unified_retrieve(
        query="GPU computing in machine learning",
        k=5,
        min_similarity=0.7
    ))

    # Access raw retrieval results
    for chunk in results["raw_results"]["relevant_chunks"]:
        print(f"Text: {chunk['text']}")

    print(json.dumps(results, indent=2))

    print ("I / len(T)", kb.stats, len(texts))

    nx_graph = kb.concept_extractor.concept_graph.convert_to_networkx()
    GraphVisualizer.visualize(nx_graph, "test_output_file.html")

    kb.save("bas.pkl")

    return kb

text = "test 123".encode("utf-8", errors="replace").decode("utf-8")


async def test_graph_enhanced_retrieval():
    """
    Umfassender Test für Graph-Enhanced Retrieval
    """
    print("=" * 80)
    print("TEST: Graph-Enhanced Retrieval System")
    print("=" * 80)

    # Initialize Knowledge Base
    kb = KnowledgeBase(
        n_clusters=3,
        model_name=os.getenv("SUMMARYMODEL", "openrouter/mistralai/mistral-7b-instruct"),
        batch_size=12,
        requests_per_second=85.
    )

    # Test Data mit klaren Konzept-Beziehungen
    test_data = [
        """
        Machine Learning is a subset of Artificial Intelligence.
        It uses algorithms to learn patterns from data.
        Deep Learning is a specialized form of Machine Learning.
        """,
        """
        Neural Networks are the foundation of Deep Learning.
        They consist of layers of interconnected nodes.
        Each layer transforms the input data progressively.
        """,
        """
        Training Neural Networks requires large datasets.
        GPUs accelerate the training process significantly.
        Backpropagation is used to update network weights.
        """,
        """
        Natural Language Processing uses Machine Learning techniques.
        Transformers are a type of Neural Network architecture.
        BERT and GPT are popular Transformer models.
        """,
        """
        Computer Vision applies Deep Learning to image analysis.
        Convolutional Neural Networks excel at image tasks.
        Object detection and segmentation are common applications.
        """,
        """
        Reinforcement Learning trains agents through rewards.
        It differs from supervised learning approaches.
        Q-Learning and Policy Gradients are key algorithms.
        """
    ]

    metadata = [{"source": f"doc_{i}", "topic": "AI"} for i in range(len(test_data))]

    print("\n" + "─" * 80)
    print("PHASE 1: Adding Data")
    print("─" * 80)

    added, duplicates = await kb.add_data(test_data, metadata, direct=True)
    print(f"✓ Added: {added} chunks")
    print(f"✓ Duplicates filtered: {duplicates}")
    print(f"✓ Total chunks in KB: {len(kb.vdb.chunks)}")
    print(f"✓ Total concepts: {len(kb.concept_extractor.concept_graph.concepts)}")

    # Test Queries
    test_queries = [
        "How does Deep Learning work?",
        "GPU acceleration in AI",
        "Transformer architecture"
    ]

    print("\n" + "─" * 80)
    print("PHASE 2: Comparing Standard vs Graph-Enhanced Retrieval")
    print("─" * 80)

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: '{query}'")
        print(f"{'=' * 80}")

        # Standard Retrieval
        print("\n[STANDARD RETRIEVAL]")
        standard_results = await kb.retrieve(query, k=3, min_similarity=0.1)
        print(f"  Found: {len(standard_results)} chunks")
        for i, chunk in enumerate(standard_results[:2], 1):
            print(f"  {i}. Concepts: {chunk.metadata.get('concepts', [])[:3]}")
            print(f"     Text: {chunk.text[:80]}...")

        # Graph-Enhanced Retrieval
        print("\n[GRAPH-ENHANCED RETRIEVAL]")
        graph_results = await kb.graph_enhanced_retrieve(
            query=query,
            k=3,
            graph_hops=2,
            relation_weight=0.3,
            min_similarity=0.1
        )

        print(f"  Initial: {graph_results['expansion_stats']['initial_count']} chunks")
        print(f"  Expanded: {graph_results['expansion_stats']['expanded_count']} chunks")
        print(f"  Expansion ratio: {graph_results['expansion_stats']['expansion_ratio']:.2f}x")

        print(f"\n  Top 3 Results (by hybrid score):")
        for i, item in enumerate(graph_results['detailed_scores'][:3], 1):
            chunk = item['chunk']
            print(f"\n  {i}. Score: {item['score']:.3f} "
                  f"(Vec: {item['vec_similarity']:.3f}, Graph: {item['graph_score']:.3f})")
            print(f"     Initial Match: {'✓' if item['is_initial'] else '✗'}")
            print(f"     Concepts: {item['concepts'][:3]}")
            print(f"     Text: {chunk.text[:80]}...")

    print("\n" + "─" * 80)
    print("PHASE 3: Unified Retrieval Comparison")
    print("─" * 80)

    query = "Explain Neural Networks and their training"

    # Without Graph Expansion
    print("\n[WITHOUT Graph Expansion]")
    results_without = await kb.unified_retrieve(
        query=query,
        k=3,
        use_graph_expansion=False
    )

    if results_without:
        chunk_count_without = len(results_without.get('raw_results', {}).get('relevant_chunks', []))
        print(f"  Chunks returned: {chunk_count_without}")
        print(f"  results_without: {results_without}")

    # With Graph Expansion
    print("\n[WITH Graph Expansion]")
    results_with = await kb.unified_retrieve(
        query=query,
        k=3,
        use_graph_expansion=True,
        graph_hops=2,
        relation_weight=0.3
    )

    if results_with:
        chunk_count_with = len(results_with.get('raw_results', {}).get('relevant_chunks', []))
        expansion_stats = results_with.get('metadata', {}).get('expansion_stats', {})
        print(f"  Chunks returned: {chunk_count_with}")
        print(f"  Expansion ratio: {expansion_stats.get('expansion_ratio', 0):.2f}x")

        summary = results_with.get('summary', {})
        print(f"\n  Summary Preview:")
        print(f"  {summary.get('main_summary', 'N/A')[:200]}...")

    print("\n" + "─" * 80)
    print("PHASE 4: Concept Graph Visualization")
    print("─" * 80)

    nx_graph = kb.concept_extractor.concept_graph.convert_to_networkx()
    print(f"  Nodes: {nx_graph.number_of_nodes()}")
    print(f"  Edges: {nx_graph.number_of_edges()}")

    # Save visualization
    html_output = await kb.vis(output_file="test_graph_enhanced.html", get_output_html=True)
    if html_output:
        print(f"  ✓ Graph visualization saved")

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return kb


async def test_edge_cases():
    """Test edge cases und error handling"""
    print("\n" + "=" * 80)
    print("EDGE CASE TESTS")
    print("=" * 80)

    kb = KnowledgeBase(n_clusters=3, model_name=os.getenv("SUMMARYMODEL"))

    # Test 1: Empty query
    print("\n[TEST 1: Empty Knowledge Base]")
    try:
        results = await kb.graph_enhanced_retrieve("test query", k=3)
        print(f"  ✓ Handled empty KB: {len(results['chunks'])} chunks returned")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Add minimal data
    await kb.add_data(["Test document about AI"], direct=True)

    # Test 2: No concepts extracted
    print("\n[TEST 2: Query with no matching concepts]")
    try:
        results = await kb.graph_enhanced_retrieve(
            "completely unrelated topic xyz123",
            k=5,
            min_similarity=0.0
        )
        print(f"  ✓ Handled: {len(results['chunks'])} chunks, "
              f"expansion: {results['expansion_stats']['expansion_ratio']:.2f}x")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test 3: High graph_hops
    print("\n[TEST 3: Very high graph_hops value]")
    try:
        results = await kb.graph_enhanced_retrieve(
            "AI",
            k=3,
            graph_hops=10
        )
        print(f"  ✓ Handled: {results['expansion_stats']['expanded_count']} chunks expanded")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "=" * 80)


# Main test runner - ERSETZE die bestehende main() falls gewünscht
async def run_all_tests():
    """Run alle Tests"""
    try:
        # Haupt-Test
        kb = await test_graph_enhanced_retrieval()

        # Edge Cases
        await test_edge_cases()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)

        return kb

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


# Wenn du die Tests ausführen willst:
if __name__ == "__main__":
    get_app(name="test_graph_enhanced")
    asyncio.run(run_all_tests())

#if __name__ == "__main__":
#    get_app(name="main2")
#
#    asyncio.run(main())

