import asyncio
import hashlib
import math
import os
import pickle
import time
from typing import Any

import networkx as nx
import numpy as np
from litellm import batch_completion, ModelResponse, fallbacks
from pydantic import BaseModel

from toolboxv2 import get_logger, get_app, Spinner
from sklearn.cluster import HDBSCAN

from typing import Dict, Set, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import json
import asyncio
from collections import defaultdict
import re

from toolboxv2.mods.isaa.extras.filter import after_format
from toolboxv2.mods.isaa.base.VectorStores.defaults import AbstractVectorStore, FastVectorStoreO, FaissVectorStore, \
    EnhancedVectorStore, FastVectorStore1
from toolboxv2.mods.isaa.extras.adapter import litellm_complete
import numpy as np
from typing import List
from collections import deque


i__ = [0, 0, 0]

@dataclass(slots=True)
class Chunk:
    """Represents a chunk of text with its embedding and metadata"""
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    content_hash: str
    cluster_id: Optional[int] = None


@dataclass
class RetrievalResult:
    """Structure for organizing retrieval results"""
    overview: List[Dict[str, any]]  # List of topic summaries
    details: List[Chunk]  # Detailed chunks
    cross_references: Dict[str, List[Chunk]]  # Related chunks by topic


class TopicSummary(NamedTuple):
    topic_id: int
    summary: str
    key_chunks: List[Chunk]
    related_chunks: List[Chunk]


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
    relationships: Dict[str, List[str]]
    importance_score: float
    context_snippets: List[str]

@dataclass
class Concept:
    name: str
    category: str
    relationships: Dict[str, Set[str]]
    importance_score: float
    context_snippets: List[str]
    metadata: Dict[str, Any]


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
    target_concepts: List[str]
    relationship_types: List[str]
    categories: List[str]


class Concepts(BaseModel):
    """
    Represents a collection of key concepts.

    Attributes:
        concepts (List[rConcept]): A list of Concept instances, each representing an individual key concept.
    """
    concepts: List[rConcept]

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
        self.concepts: Dict[str, Concept] = {}

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

    def get_related_concepts(self, concept_name: str, relationship_type: Optional[str] = None) -> Set[str]:
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
            c = open(output_file, "r", encoding="utf-8").read()
            os.remove(output_file)
            return c


class DynamicRateLimiter:
    def __init__(self):
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    def update_rate(self, requests_per_second: float):
        """Update rate limit dynamically"""
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else float('inf')

    async def acquire(self):
        """Acquire permission to make a request"""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)
            self.last_request_time = time.time()


class ConceptExtractor:
    """Handles extraction of concepts and relationships from text"""

    def __init__(self, knowledge_base, requests_per_second = 85.):
        self.kb = knowledge_base
        self.concept_graph = ConceptGraph()
        self.requests_per_second = requests_per_second

    async def extract_concepts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[List[Concept]]:
        """
        Extract concepts from texts using concurrent processing with rate limiting.
        Requests are made at the specified rate while responses are processed asynchronously.
        """
        # Ensure metadatas list matches texts length
        metadatas = metadatas + [{}] * (len(texts) - len(metadatas))

        # Initialize rate limiter
        rate_limiter = DynamicRateLimiter()

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
            for idx, (text, metadata) in enumerate(zip(texts, metadatas))
        ]

        async def process_single_request(idx: int, prompt: str, system_prompt: str, metadata: Dict[str, Any]):
            """Process a single request with rate limiting"""
            try:
                # Wait for rate limit
                await rate_limiter.acquire()
                i__[1] += 1
                # Make API call without awaiting the response
                response_future = litellm_complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_format=Concepts,
                    model_name=self.kb.model_name,
                    fallbacks=["groq/gemma2-9b-it"] +
                              [m for m in os.getenv("FALLBACKS_MODELS_PREM", '').split(',') if m]
                )

                return idx, response_future

            except Exception as e:
                print(f"Error initiating request {idx}: {str(e)}")
                return idx, None

        async def process_response(idx: int, response_future) -> List[Concept]:
            """Process the response once it's ready"""
            try:
                if response_future is None:
                    return []

                response = await response_future
                return await self._process_response(response, metadatas[idx])

            except Exception as e:
                print(f"Error processing response {idx}: {str(e)}")
                return []

        # Create tasks for all requests
        request_tasks = []
        batch_size = self.kb.batch_size

        rate_limiter.update_rate(self.requests_per_second)

        for batch_start in range(0, len(requests), batch_size):
            batch = requests[batch_start:batch_start + batch_size]
            print("Total Concepts:",len(self.concept_graph.concepts.values()))

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
            sorted_results[idx] = concepts

        return sorted_results

    async def _process_response(self, response: Any, metadata: Dict[str, Any]) -> List[Concept]:
        """Helper method to process a single response and convert it to Concepts"""
        try:
            # Extract content from response
            if hasattr(response, 'choices'):
                content = response.choices[0].message.content
                if content is None:
                    content = response.choices[0].message.tool_calls[0].function.arguments
                if content is None:
                    return []
            elif isinstance(response, str):
                content = response
            else:
                print(f"Unexpected response type: {type(response)}")
                return []

            # Parse JSON and create concepts
            concept_data = after_format(content)
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

        except Exception as e:
            i__[2] +=1
            return []

    async def process_chunks(self, chunks: List[Chunk]) -> None:
        """
        Process all chunks in batch to extract and store concepts.
        Each chunk's metadata will be updated with the concept names and relationships.
        """
        # Gather all texts from the chunks.
        texts = [chunk.text for chunk in chunks]
        # Call extract_concepts once with all texts.
        all_concepts = await self.extract_concepts(texts, [chunk.metadata for chunk in chunks])

        # Update each chunk's metadata with its corresponding concepts.
        for chunk, concepts in zip(chunks, all_concepts):
            chunk.metadata["concepts"] = [c.name for c in concepts]
            chunk.metadata["concept_relationships"] = {
                c.name: {k: list(v) for k, v in c.relationships.items()}
                for c in concepts
            }

    async def query_concepts(self, query: str) -> Dict[str, any]:
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
            response = await litellm_complete(
                model_name=self.kb.model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                response_format=TConcept
            )

            query_params = json.loads(response)

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
        base_chunks = math.ceil(text_len / self.chunk_size)
        estimated_chunks = base_chunks * 2 / (overlap_ratio if overlap_ratio > 0 else 1)

        # print('#',estimated_chunks, base_chunks, overlap_ratio)
        # Calculate average chunk size
        avg_chunk_size = max(1, math.ceil(text_len / estimated_chunks))

        return estimated_chunks * avg_chunk_size

    def split_text(self, text: str) -> List[str]:
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
    def __init__(self, embedding_dim: int = 768, similarity_threshold: float = 0.61, batch_size: int = 64,
                 n_clusters: int = 4, deduplication_threshold: float = 0.85, model_name=os.getenv("DEFAULTMODELSUMMERY"),
                 embedding_model=os.getenv("DEFAULTMODELEMBEDDING"),
                 vis_class:Optional[str] = "FastVectorStoreO",
                 vis_kwargs:Optional[Dict[str, Any]]=None,
                 requests_per_second=85.,
                 chunk_size: int = 3600,
                 chunk_overlap: int = 130,
                 separator: str = "\n"
                 ):
        """Initialize the knowledge base with given parameters"""

        self.existing_hashes: Set[str] = set()
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.deduplication_threshold = deduplication_threshold
        if model_name == "openrouter/mistralai/mistral-nemo":
            batch_size = 9
            requests_per_second = 1.5
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.model_name = model_name
        print("Initialized", model_name)
        self.sto: list = []

        self.text_splitter = TextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap, separator=separator)
        self.similarity_graph = {}
        self.concept_extractor = ConceptExtractor(self, requests_per_second)

        self.vis_class = None
        self.vis_kwargs = None
        self.vdb = None
        self.init_vis(vis_class, vis_kwargs)

    def init_vis(self, vis_class, vis_kwargs):
        if vis_class is None:
            vis_class = "FastVectorStoreO"
        if vis_class == "FastVectorStoreO":
            if vis_kwargs is None:
                vis_kwargs = {
                    "embedding_size": self.embedding_dim
                }
            self.vdb = FastVectorStoreO(**vis_kwargs)
        if vis_class == "FaissVectorStore":
            if vis_kwargs is None:
                vis_kwargs = {
                    "dimension": self.embedding_dim
                }
            self.vdb = FaissVectorStore(**vis_kwargs)
        if vis_class == "EnhancedVectorStore":
            self.vdb = EnhancedVectorStore(**vis_kwargs)
        if vis_class == "FastVectorStore1":
            self.vdb = FastVectorStore1()

        self.vis_class = vis_class
        self.vis_kwargs = vis_kwargs


    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute SHA-256 hash of text"""
        return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()

    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get normalized embeddings in batches"""
        try:
            async def process_batch(batch: List[str]) -> np.ndarray:
                from toolboxv2.mods.isaa.extras.adapter import litellm_embed
                # print("Processing", batch)
                embeddings = await litellm_embed(texts=batch, model=self.embedding_model)
                return normalize_vectors(embeddings)

            tasks = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                tasks.append(process_batch(batch))

            embeddings = await asyncio.gather(*tasks)
            i__[0] += len(texts)
            return np.vstack(embeddings)
        except Exception as e:
            get_logger().error(f"Error generating embeddings: {str(e)}")
            raise



    def _remove_similar_chunks(self, threshold: float = None) -> int:
        """Remove chunks that are too similar to each other"""
        if len(self.vdb.chunks) < 2:
            return 0

        if threshold is None:
            threshold = self.deduplication_threshold

        try:
            # Get all embeddings
            embeddings = np.vstack([c.embedding for c in self.vdb.chunks])
            n = len(embeddings)

            # Compute similarity matrix
            similarities = np.dot(embeddings, embeddings.T)

            # Create mask for chunks to keep
            keep_mask = np.ones(n, dtype=bool)

            # Iterate through chunks
            for i in range(n):
                if not keep_mask[i]:
                    continue

                # Find chunks that are too similar to current chunk
                similar_indices = similarities[i] >= threshold
                similar_indices[i] = False  # Don't count self-similarity

                # Mark similar chunks for removal
                keep_mask[similar_indices] = False

            # Keep only unique chunks
            unique_chunks = [chunk for chunk, keep in zip(self.vdb.chunks, keep_mask) if keep]
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
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]]= None,
    ) -> Tuple[int, int]:
        """
        Process and add new data to the knowledge base
        Returns: Tuple of (added_count, duplicate_count)
        """
        if len(texts) == 0:
            return -1, -1
        try:
            # Compute hashes and filter exact duplicates
            hashes = [self.compute_hash(text) for text in texts]
            unique_data = []
            for t, m, h in zip(texts, metadata, hashes):
                if h in self.existing_hashes:
                    continue
                # Update existing hashes
                self.existing_hashes.add(h)
                unique_data.append((t, m, h))

            if not unique_data:
                return 0, len(texts)

            # Get embeddings
            embeddings = await self._get_embeddings(texts)

            texts = []
            metadata = []
            hashes = []
            embeddings_final = []
            if len(self.vdb.chunks):
                for i, d in enumerate(unique_data):
                    c = self.vdb.search(embeddings[i], 5, self.deduplication_threshold)
                    if len(c) > 2:
                        continue
                    t, m, h = d
                    texts.append(t)
                    metadata.append(m)
                    hashes.append(h)
                    embeddings_final.append(embeddings[i])

            else:
                texts , metadata, hashes = zip(*unique_data)
                embeddings_final = embeddings

            if not texts:  # All were similar to existing chunks
                return 0, len(unique_data)

            # Create and add new chunks
            new_chunks = [
                Chunk(text=t, embedding=e, metadata=m, content_hash=h)
                for t, e, m, h in zip(texts, embeddings_final, metadata, hashes)
            ]
            print("new_chunks ing", len(new_chunks))
            # Add new chunks
            # Update index
            if new_chunks:
                all_embeddings = np.vstack([c.embedding for c in new_chunks])
                self.vdb.add_embeddings(all_embeddings, new_chunks)

            # Remove similar chunks from the entire collection
            removed = self._remove_similar_chunks()
            get_logger().info(f"Removed {removed} similar chunks during deduplication")
            # Invalidate visualization cache

            if len(new_chunks) - removed > 0:
                # Process new chunks for concepts
                await self.concept_extractor.process_chunks(new_chunks)
            print("i________________________________________________________________", i__)

            return len(new_chunks) - removed, len(texts) - len(new_chunks) + removed

        except Exception as e:
            get_logger().error(f"Error adding data: {str(e)}")
            raise


    async def add_data(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[int, int]:
        """Enhanced version with smart splitting and clustering"""
        if isinstance(texts, str):
            texts = [texts]
        if metadata is None:
            metadata = [{}] * len(texts)
        if isinstance(metadata, dict):
            metadata = [metadata]
        if len(texts) != len(metadata):
            raise ValueError("Length of texts and metadata must match")
        if len(texts) == 1 and len(texts[0]) < 10_000:
            if len(self.sto) < self.batch_size and len(texts) == 1:
                self.sto.append((texts[0], metadata[0]))
                return -1, -1
            if len(self.sto) >= self.batch_size:
                _ = [texts.append(t) or metadata.append([m]) for (t, m) in self.sto]
                self.sto = []
                print(len(texts), len(metadata), "NEW data adding")

        # Split large texts
        split_texts = []
        split_metadata = []

        for idx, text in enumerate(texts):
            chunks = self.text_splitter.split_text(text)
            split_texts.extend(chunks)

            # Adjust metadata for splits
            meta = metadata[idx] if metadata else {}
            if isinstance(meta, list):
                meta = meta[0]
            for i, chunk in enumerate(chunks):
                chunk_meta = meta.copy()
                chunk_meta.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'original_text_id': idx
                })
                split_metadata.append(chunk_meta)

        return await self._add_data(split_texts, split_metadata)

    def _update_similarity_graph(self, embeddings: np.ndarray, chunk_ids: List[int]):
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
        query_embedding: Optional[np.ndarray] = None,
        k: int = 5,
        min_similarity: float = 0.2,
        include_connected: bool = True
    ) -> List[Chunk]:
        """Enhanced retrieval with connected information"""
        if query_embedding is None:
            query_embedding = (await self._get_embeddings([query]))[0]
        k = min(k, len(self.vdb.chunks)-1)
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

    async def forget_irrelevant(self, irrelevant_concepts: List[str], similarity_threshold: Optional[float]=None) -> int:
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
        chunks: List[Chunk],
        query_embedding: Optional[np.ndarray] = None,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        max_clusters: int = 10
    ) -> Dict[int, List[Chunk]]:
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
        clusters: Dict[int, List[Chunk]] = {}

        # Sort clusters by size and relevance
        cluster_scores = []

        for label in set(best_clusters):
            if label == -1:  # Handle noise points separately
                continue

            # Fixed: Use boolean mask to select chunks for current cluster
            cluster_mask = best_clusters == label
            cluster_chunks = [chunk for chunk, is_in_cluster in zip(chunks, cluster_mask) if is_in_cluster]

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
        noise_chunks = [chunk for chunk, label in zip(chunks, best_clusters) if label == -1]
        if noise_chunks:
            self._assign_noise_points(noise_chunks, clusters, query_embedding)

        return clusters

    @staticmethod
    def _evaluate_clustering(
        embeddings: np.ndarray,
        labels: np.ndarray,
        query_embedding: Optional[np.ndarray] = None
    ) -> float:
        """
        Evaluate clustering quality using multiple metrics
        """
        if len(set(labels)) <= 1:
            return float('-inf')

        # Calculate silhouette score for cluster cohesion
        from sklearn.metrics import silhouette_score
        try:
            sil_score = silhouette_score(embeddings, labels, metric='cosine')
        except:
            sil_score = -1

        # Calculate Davies-Bouldin score for cluster separation
        from sklearn.metrics import davies_bouldin_score
        try:
            db_score = -davies_bouldin_score(embeddings, labels)  # Negated as lower is better
        except:
            db_score = -1

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
        chunks: List[Chunk],
        query_embedding: Optional[np.ndarray] = None
    ) -> Dict[int, List[Chunk]]:
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
        noise_chunks: List[Chunk],
        clusters: Dict[int, List[Chunk]],
        query_embedding: Optional[np.ndarray] = None
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
        chunks: List[Chunk],
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
        max_cross_refs: int = 10  # New parameter to control cross-reference count
    ) -> RetrievalResult:
        """Enhanced retrieval with better cross-reference handling"""
        # Get initial results with query embedding
        if query_embedding is None:
            query_embedding = (await self._get_embeddings([query]))[0]
        initial_results = await self.retrieve(query_embedding=query_embedding, k=k, min_similarity=min_similarity)

        if not initial_results:
            return RetrievalResult([], [], {})

        # Find cross-references with similarity scoring
        initial_ids = {self.vdb.chunks.index(chunk) for chunk in initial_results}
        related_ids = self._find_cross_references(
            initial_ids,
            depth=cross_ref_depth,
            query_embedding=query_embedding  # Pass query embedding for relevance scoring
        )

        # Get all relevant chunks with smarter filtering
        all_chunks = self.vdb.chunks
        all_relevant_chunks = initial_results + [
            chunk for i, chunk in enumerate(all_chunks)
            if i in related_ids and self._is_relevant_cross_ref(
                chunk,
                query_embedding,
                initial_results
            )
        ]

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
                max_sentences=max_sentences  # Increased for more context
            )

            # Enhanced chunk sorting with combined scoring
            sorted_chunks = self._sort_chunks_by_relevance(
                cluster_chunks,
                query_embedding,
                initial_results
            )

            # Separate direct matches and cross-references
            direct_matches_ = [{'text':c.text, 'metadata':c.metadata} for c in sorted_chunks if c in initial_results]
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
        chunk_ids: Set[int],
        depth: int,
        query_embedding: np.ndarray
    ) -> Set[int]:
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
        initial_results: List[Chunk]
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
        cross_refs: List[Chunk],
        max_count: int,
        query_embedding: np.ndarray
    ) -> List[Chunk]:
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
        chunks: List[Chunk],
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
        chunks: List[Chunk],
        query_embedding: np.ndarray,
        initial_results: List[Chunk]
    ) -> List[Chunk]:
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

    async def query_concepts(self, query: str) -> Dict[str, any]:
        """Query concepts extracted from the knowledge base"""
        return await self.concept_extractor.query_concepts(query)

    async def unified_retrieve(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.2,
        cross_ref_depth: int = 2,
        max_cross_refs: int = 10,
        max_sentences: int = 10
    ) -> Dict[str, Any]:
        """
        Unified retrieval function that combines concept querying, retrieval with overview,
        and basic retrieval, then generates a comprehensive summary using LLM.

        Args:
            query: Search query string
            k: Number of primary results to retrieve
            min_similarity: Minimum similarity threshold for retrieval
            cross_ref_depth: Depth for cross-reference search
            max_cross_refs: Maximum number of cross-references per topic
            max_sentences: Maximum number Sentences in the main summary text

        Returns:
            Dictionary containing comprehensive results including summary and details
        """
        # Get concept information
        concept_results = await self.concept_extractor.query_concepts(query)

        # Get retrieval overview

        query_embedding = (await self._get_embeddings([query]))[0]
        overview_results = await self.retrieve_with_overview(
            query=query,
            query_embedding=query_embedding,
            k=k,
            min_similarity=min_similarity,
            cross_ref_depth=cross_ref_depth,
            max_cross_refs=max_cross_refs,
            max_sentences=max_sentences
        )

        # Get basic retrieval results
        basic_results = await self.retrieve(
            query_embedding=query_embedding,
            k=k,
            min_similarity=min_similarity
        )
        if len(basic_results) == 0:
            return {}
        if len(basic_results) == 1 and isinstance(basic_results[0], str) and basic_results[0].endswith('[]\n - []\n - []'):
            return {}

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
                for chunk in basic_results
            ]
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
            await asyncio.sleep(0.25)
            llm_response = await litellm_complete(
                model_name=self.model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                response_format=DataModel,
            )
            summary_analysis = json.loads(llm_response)
        except Exception as e:
            get_logger().error(f"Error generating summary: {str(e)}")
            summary_analysis = {
                "main_summary": "Error generating summary",
                "error": str(e)
            }

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
                    for chunk in basic_results
                ]
            },
            "metadata": {
                "query": query,
                "timestamp": time.time(),
                "retrieval_params": {
                    "k": k,
                    "min_similarity": min_similarity,
                    "cross_ref_depth": cross_ref_depth,
                    "max_cross_refs": max_cross_refs
                }
            }
        }

    def save(self, path: str) -> Optional[bytes]:
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
            if path is None:
                return pickle.dumps(data)
            # Save to disk using pickle
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Knowledge base successfully saved to {path} with {len(self.concept_extractor.concept_graph.concepts.items())} concepts")

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
            if isinstance(path, str):
                # Load data from disk
                with open(path, 'rb') as f:
                    data = pickle.load(f)
            elif isinstance(path, bytes):
                data = pickle.loads(path)
            else:
                raise ValueError("Invalid path type")

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

            print(f"Knowledge base successfully loaded from {path} with {len(concept_data)} concepts")
            return kb

        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")
            raise

    def vis(self,output_file: str = "concept_graph.html", get_output_html=False, get_output_net=False):
        if not self.concept_extractor.concept_graph.concepts:
            print("NO Concepts defined")
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

    print ("I / len(T)", i__, len(texts))

    nx_graph = kb.concept_extractor.concept_graph.convert_to_networkx()
    GraphVisualizer.visualize(nx_graph, "test_output_file.html")

    kb.save("bas.pkl")

async def rgen():
    kb = KnowledgeBase.load("mem.plk")
    #res =await kb.concept_extractor.extract_concepts(["hallo das ist ein test", "wie geht es dir", "nicht", "Phiskik ist sehr wichtig"], [{}]*4)
    #print(res)
    print(await kb.forget_irrelevant(["lazy dog", "unimportant"], 0.51))
    print(await kb.query_concepts("AI"))
    print(await kb.retrieve("Evaluation metrics for assessing AI Agent performance"))
    print(kb.concept_extractor.concept_graph.concepts.keys())
    #GraphVisualizer.visualize(kb.concept_extractor.concept_graph.convert_to_networkx(), output_file="concept_graph2.html")


text = """Analysis I und Lineare Algebra fur Ingenieurwissenschaften ̈

#### Olivier S`ete J ̈org Liesen

#### 21. Oktober 2019



## Inhaltsverzeichnis

## Inhaltsverzeichnis






- Inhaltsverzeichnis
- Vorwort
- 1 Grundlagen: Mengen und Logik
   - 1.1 Mengen
   - 1.2 Grundlagen der Logik
- 2 Zahlen
   - 2.1 Zahlen und Zahldarstellung
   - 2.2 Ungleichungen
   - 2.3 Reelle Wurzeln
   - 2.4 Absolutbetrag
   - 2.5 Beispiele zum L ̈osen von Ungleichungen
   - 2.6 Summenzeichen
   - 2.7 Produktzeichen
- 3 Komplexe Zahlen
   - 3.1 Definition und Grundrechenarten
   - 3.2 Die Gaußsche Zahlenebene
   - 3.3 Absolutbetrag und Konjugierte
   - 3.4 L ̈osungen quadratischer Gleichungen
- 4 Vollst ̈andige Induktion
   - 4.1 Induktion
   - 4.2 Binomischer Lehrsatz
- 5 Abbildungen
   - 5.1 Definition
   - 5.2 Komposition von Abbildungen
   - 5.3 Umkehrbarkeit
   - 5.4 Eigenschaften reeller Funktionen
      - 5.4.1 Symmetrie
      - 5.4.2 Monotonie
      - 5.4.3 Beschr ̈anktheit
- 6 Elementare Funktionen
   - 6.1 Exponentialfunktion und Logarithmus
   - 6.2 Die trigonometrischen Funktionen Sinus und Cosinus
   - 6.3 Tangens
- 7 Komplexe Zahlen
   - 7.1 Polardarstellung
   - 7.2 Vorteile der Euler- und Polardarstellungen
   - 7.3 Komplexe Wurzeln
   - 7.4 L ̈osungen quadratischer Gleichungen
- 8 Polynome
   - 8.1 Rechenoperationen
   - 8.2 Nullstellen von Polynomen
   - 8.3 Reelle Polynome
   - 8.4 Nullstellen berechnen
- 9 Rationale Funktionen
   - 9.1 Komplexe Partialbruchzerlegung
   - 9.2 Reelle Partialbruchzerlegung
   - 9.3 Zusammenfassung
- 10 Vektorr ̈aume
   - 10.1 Vektorr ̈aume
   - 10.2 Teilr ̈aume
   - 10.3 Linearkombinationen
   - 10.4 Erzeugendensysteme
- 11 Basis und Dimension
   - 11.1 Lineare Unabh ̈angigkeit
   - 11.2 Basis und Dimension
   - 11.3 Koordinaten
- 12 Matrizen
   - 12.1 Definition von Matrizen
   - 12.2 Addition und Skalarmultiplikation
   - 12.3 Matrizenmultiplikation
   - 12.4 Inverse
   - 12.5 Transposition
- 13 Lineare Gleichungssysteme
   - 13.1 Matrixschreibweise eines linearen Gleichungssystems
   - 13.2 Der Gauß-Algorithmus
   - 13.3 Anwendung auf lineare Gleichungssysteme
   - 13.4 Struktur der L ̈osungsmenge
- 14 Weitere Anwendungen des Gauß-Algorithmus
   - 14.1 Der Rang einer Matrix
   - 14.2 L ̈osbarkeitskriterium f ̈ur lineare Gleichungssysteme
   - 14.3 Invertierbarkeit von Matrizen
   - 14.4 Unterschiede zwischen Matrizen und Zahlen
- 15 Lineare Abbildungen
   - 15.1 Definition und erste Eigenschaften
   - 15.2 Kern und Bild
   - 15.3 Dimensionsformel und Konsequenzen
- 16 Koordinaten und Matrixdarstellung
   - 16.1 Koordinaten
   - 16.2 Matrixdarstellung
   - 16.3 Basiswechsel
- 17 Konvergenz von Zahlenfolgen
   - 17.1 Zahlenfolgen
   - 17.2 Konvergenz
   - 17.3 Bestimmte Divergenz
- 18 Berechnung von Grenzwerten
   - 18.1 Grenzwerts ̈atze
   - 18.2 Grenzwerte und Ungleichungen
   - 18.3 Monotonie und Konvergenz
   - 18.4 Wichtige Grenzwerte
- 19 Stetigkeit
   - 19.1 Grenzwerte von Funktionen
   - 19.2 Einseitige Grenzwerte von Funktionen
   - 19.3 Stetigkeit
- 20 S ̈atzeuber stetige Funktionen ̈
   - 20.1 Bestimmung von Nullstellen
   - 20.2 Existenz von Extremwerten
- 21 Differenzierbarkeit
   - 21.1 Definition
   - 21.2 Interpretation der Ableitung
   - 21.3 Rechenregeln
- 22 Erste Anwendungen der Differenzierbarkeit
   - 22.1 Ableitung der Umkehrfunktion
   - 22.2 Nullstellen
   - 22.3 H ̈ohere Ableitungen
   - 22.4 Regel von Bernoulli/de l’Hospital
- 23 Mittelwertsatz und Anwendungen
   - 23.1 Extremwerte
   - 23.2 Mittelwertsatz
   - 23.3 Anwendungen des Mittelwertsatzes
- 24 Taylor-Approximation
   - 24.1 Die Taylor-Approximation
   - 24.2 Extremwerte
- 25 Anwendungen der Taylor-Approximation
   - 25.1 N ̈aherungsweise Berechnung von Funktionswerten
   - 25.2 Fehlerabsch ̈atzung
   - 25.3 Diskretisierung von Ableitungen
   - 25.4 Taylorreihen
- 26 Elementare Funktionen
   - 26.1 Exponential- und Logarithmusfunktion
   - 26.2 Allgemeine Potenzfunktion
   - 26.3 Komplexe Exponentialfunktion
- 27 Elementare Funktionen
   - 27.1 Trigonometrische Funktionen
   - 27.2 Arcus-Funktionen – Umkehrfunktionen der Winkelfunktionen
   - 27.3 Hyperbolische Funktionen
- 28 Das Integral
   - 28.1 Integraldefinition und Fl ̈achenberechnung
   - 28.2 Rechenregeln
   - 28.3 Das Integral als Mittelwert der Funktion
- 29 Integrationsregeln
   - 29.1 Stammfunktionen
   - 29.2 Der Hauptsatz der Differential- und Integralrechnung
   - 29.3 Grundintegrale
   - 29.4 Partielle Integration
- 30 Integrationsregeln 2 und Integration komplexer Funktionen
   - 30.1 Substitutionsregel
   - 30.2 Integration komplexer Funktionen
- 31 Uneigentliche Integrale und Integration rationaler Funktionen
   - 31.1 Unbeschr ̈ankter Definitionsbereich
   - 31.2 Unbeschr ̈ankte Funktion
   - 31.3 Integration rationaler Funktionen
- 32 Die Determinante
   - 32.1 Determinante und Volumenberechnung
   - 32.2 Berechnung von Determinanten
   - 32.3 Der Determinantenmultiplikationssatz
   - 32.4 Charakterisierung invertierbarer Matrizen
- 33 Eigenwerte und Eigenvektoren
   - 33.1 Definition von Eigenwerten und Eigenvektoren
   - 33.2 Berechnung von Eigenwerten und Eigenvektoren
   - 33.3 Eigenvektoren und lineare Unabh ̈angigkeit
- 34 Diagonalisierbarkeit
   - 34.1 Definition und Charakterisierung
   - 34.2 Anwendungen
- 35 Vektorr ̈aume mit Skalarprodukt
   - 35.1 Norm
   - 35.2 Skalarprodukte
   - 35.3 Orthogonale Vektoren
   - 35.4 Orthonormalbasen
   - 35.5 Orthogonale Matrizen
   - 35.6 Unit ̈are Matrizen
- 36 Vektorr ̈aume mit Skalarprodukt
   - 36.1 Kurzeste Abst ̈ ̈ande und orthogonale Projektion
   - 36.2 Das Gram-Schmidt-Verfahren
   - 36.3 QR-Zerlegung
   - 36.4 Lineare Regression
- 37 Reelle Fourieranalysis
   - 37.1 Trigonometrische Polynome
   - 37.2 Reelle Fourierapproximation
- 38 Approximation im quadratischen Mittel
   - 38.1 Fourierkoeffizienten f ̈ur gerade und ungerade Funktionen
   - 38.2 Approximation im quadratischen Mittel
   - 38.3 Fourierreihen
- 39 Komplexe Fourieranalysis
- 40 Reihen
   - 40.1 Konvergenz von Reihen
   - 40.2 Konvergenzkriterien
- 41 Absolut konvergente Reihen
   - 41.1 Absolute Konvergenz
   - 41.3 Komplexe Reihen 41.2 Konvergenzkriterien fur absolute Konvergenz 309 ̈
- 42 Potenzreihen
   - 42.1 Konvergenz von Potenzreihen
   - 42.2 Ableitung von Potenzreihen
   - 42.3 Taylor- und Potenzreihen
- 43 Lineare Differentialgleichungen
   - 43.1 Lineare Differentialgleichungen 1. Ordnung
   - 43.2 Systeme linearer Differentialgleichungen 1. Ordnung
   - 43.3 Lineare skalare Differentialgleichungen 2. Ordnung
- Index


## Vorwort

Dieses Skript wurde f ̈ur die im Wintersemester 2017/2018 erstmals an der TU Berlin
gehaltene Vorlesung”Analysis I und Lineare Algebra f ̈ur Ingenieurwissenschaften“ (6+
SWS) entwickelt.
Es basiert unter anderem auf den folgenden Quellen:

- Dirk Ferus, Vorlesungsskript”Analysis I f ̈ur Ingenieurwissenschaften“, TU Berlin,
    Institut f ̈ur Mathematik, Version vom 23.01.2007.
- Volker Mehrmann, J ̈org Rambau, Ruedi Seiler, Vorlesungsskript”Lineare Algebra
    f ̈ur Ingenieurwissenschaften“, TU Berlin, Institut f ̈ur Mathematik, Version vom
    10.10.2008.
- Kurt Meyberg, Peter Vachenhauer, H ̈ohere Mathematik 1, 4. Auflage, Springer
    Verlag, 1997.
- Christian Karpfinger, H ̈ohere Mathematik in Rezepten, 2. Auflage, Springer Spek-
    trum, 2017.

Jedes Kapitel des Skripts enth ̈alt den Stoff einer 90-min ̈utigen Vorlesung. Kleingedruckte
Textabschnitte sind Ausblicke, die typischerweise nicht in den Vorlesungen behandelt
werden.
Unser besonderer Dank gilt Konstantin Fackeldey, Frank Lutz, Christian Mehl, und
Gabriele Penn-Karras, die bei der Entwicklung des Skripts durch viele konstruktive Dis-
kussionen und wichtige Beitr ̈age geholfen haben. F ̈ur hilfreiche Kommentare bedanken
wir uns bei Christian Kreusler, Patrick Winkert und Jan Zur.
Verbesserungsvorschl ̈age und Fehlerhinweise sind jederzeit willkommen. Bitte senden
Sie diese an

```
sete@math.tu-berlin.de
```
J ̈org Liesen und Olivier S`ete
TU Berlin, Insitut f ̈ur Mathematik
(Stand: 21. Oktober 2019)


An die Lehrenden. Jedes Kapitel des Skripts enth ̈alt den Stoff einer 90-min ̈utigen
Vorlesung. Kleingedruckte Textabschnitte sind Ausblicke, die typischerweise nicht in den
Vorlesungen behandelt werden.
Vorlesung 1 ist bewusst k ̈urzer gehalten, damit vorweg Organisatorisches zur Veran-
staltung besprochen werden kann.
Vorlesung 2 ist etwas zu lang. Hier kann etwas gekurzt werden, oder das Ende (Pro- ̈
dukte, und ggf. Summen) kann in Vorlesung 3 beendet werden.


Vorlesung 1

## 1 Grundlagen: Mengen und Logik

Mengen begegnen uns vornehmlich als L ̈osungsmengen von Gleichungen oder Unglei-
chungen und als Definitionsbereiche von Funktionen (Vorlesung 5).

### 1.1 Mengen

Wir geben eine kurze Einf ̈uhrung in die Sprache der Mengenlehre und die elementaren
Operationen mit Mengen.

Definition 1.1(Menge). EineMengeist die Zusammenfassung von wohlunterschiede-
nen Objekten unserer Anschauung oder Denkens zu einem Ganzen.

Mengen werden typischerweise auf zwei Arten angegeben: Durch Aufz ̈ahlung aller
Elemente, zum Beispiel

```
A={ 1 , 2 , 3 , 4 }, N={ 0 , 1 , 2 , 3 , 4 , 5 ,...}
```
(Nist die Menge dernat ̈urlichen Zahlen), oder durch Angabe einer Eigenschaft, zum
Beispiel
N={n|nist eine nat ̈urliche Zahl}, B={x∈N|x^2 = 4}.
IstAeine Menge, undaeinElementvonA, so schreiben wir
a∈A

(lies:aist Element vonA). Istanicht Element vonA, so schreiben wira /∈A(lies:aist
nicht Element vonA). Zum Beispiel ist 2∈Bund 1∈/B.

Beispiel 1.2. 1){ 1 , 2 , 3 , 4 }={ 2 , 3 , 1 , 4 }={ 1 , 3 , 1 , 2 , 3 , 4 }.
Bei Mengen kommt es nicht auf die Reihenfolge der Elemente an, und jedes Element
wird nur einmal gez ̈ahlt.
2) Die Menge der ungeraden nat ̈urlichen Zahlen ist
U={ 1 , 3 , 5 , 7 , 9 ,...}={n∈N|nist ungerade}={ 2 k+ 1|k∈N}.
Zum Beispiel sind 11∈Uund 2∈/U.


```
3) Die leere Menge∅enth ̈alt kein Element. Alternative Schreibweise:{ }.
```
Definition 1.3(Operationen mit Mengen). SeienA,BMengen.
1)AheißtTeilmengevonB, falls gilt:x∈A⇒x∈B(lies: ausx∈Afolgtx∈B).
Schreibweise:A⊆B.
Man findet desOfteren auch die Schreibweise ̈ A⊂B, die je nach Autor Teilmenge
(A⊆B) oder echte Teilmenge (A⊆BundA 6 =B) bedeuten kann.

##### A B

```
Zum Beispiel ist{ 1 , 2 }⊆{ 1 , 2 , 3 , 4 }, und f ̈ur jede Menge giltA⊆A.
2) Die MengenAundBsindgleich, geschriebenA=B, wenn sie die gleichen Ele-
mente haben. Daher giltA=Bgenau dann, wennA⊆BundB⊆Agelten.
Zum Beispiel ist{ 1 , 2 }={ 2 , 1 }={ 1 , 1 , 2 }, aber{ 1 , 2 }6={ 1 , 2 , 3 }(hier gilt nur⊆)
und{ 1 , 2 }6={ 1 , 3 }.
3) DieVereinigungsmengeoder kurzVereinigungvonAundBist
```
```
A∪B:={x|x∈Aoderx∈B}.
```
```
Ein Gleichheitszeichen mit Doppelpunkt steht f ̈ur eine Definition. Der Doppel-
punkt steht auf der Seite, die definiert wird.
Die Vereinigung enth ̈alt alle Elemente vonAund alle Elemente vonB:
```
##### A B

##### A∪B

```
Zum Beispiel ist{ 1 , 2 }∪{ 1 , 3 }={ 1 , 2 , 3 }.
4) DieSchnittmenge(auchDurchschnittoderSchnitt) vonAundBist
```
```
A∩B:={x|x∈Aundx∈B}.
```
```
Der Schnitt enth ̈alt alle Elemente, die in beiden Mengen gleichzeitig sind:
```
##### A A∩B B

```
Zum Beispiel ist{ 1 , 2 }∩{ 1 , 3 }={ 1 }.
```

```
5) DieDifferenzmenge(auchDifferenz) vonAundBist
```
```
A\B:={x|x∈Aundx /∈B}.
Diese Differenzmenge entsteht, indem man aus der MengeAalle Elemente vonB
entfernt:
```
##### A B

##### A\B

```
Zum Beispiel ist{ 1 , 2 }\{ 1 , 3 }={ 2 }.
6) Daskartesische ProduktvonAundBist die Menge aller (geordneten) Paare (a,b)
mita∈Aundb∈B:
```
```
A×B:={(a,b)|a∈A,b∈B}.
Zum Beispiel istR^2 :=R×Rdie Menge aller Punkte der Ebene.
Allgemeiner definiert man das kartesische Produkt dernMengenA 1 ,...,Anals
```
```
A 1 ×A 2 ×...×An:={(x 1 ,x 2 ,...,xn)|xi∈Ai,i= 1, 2 ,...,n},
also als die Menge der geordnetenn-Tupel.
```
Beispiel 1.4. SeienA={ 1 , 2 }undB={ 1 , 3 }, dann ist

```
A×B={(1,1),(1,3),(2,1),(2,3)}.
```
Beachten Sie, dass die Reihenfolge in den Paaren wichtig ist: Der erste Eintrag ist aus
A, der zweite ausB. Daher ist zum Beispiel (3,2) nicht inA×B.

Weitere wichtige Beispiele von Mengen sindIntervalle. Intervalle sind Teilmengen
der reellen Zahlen von einer der folgenden Formen, wobeia,b∈Rmita≤bist:

```
]a,b[:={x∈R|a < x < b} (offenes Intervall),
[a,b[:={x∈R|a≤x < b} (halboffenes Intervall),
]a,b]:={x∈R|a < x≤b} (halboffenes Intervall),
[a,b]:={x∈R|a≤x≤b} (abgeschlossenes Intervall).
```
Ein abgeschlossenes und beschr ̈anktes Intervall [a,b] heißt auchkompaktesIntervall. F ̈ur
unbeschr ̈ankte Intervalle schreibt man

```
]a,+∞[:={x∈R|a < x} (offenes Intervall),
[a,+∞[:={x∈R|a≤x} (halboffenes Intervall),
]−∞,b[:={x∈R|x < b} (offenes Intervall)
]−∞,b]:={x∈R|x≤b} (halboffenes Intervall),
```

sowie
]−∞,∞[:=R.

Beachten Sie, dass die Symbole∞= +∞und−∞keine reellen Zahlen sind und daher
in keinem Intervall enthalten sind.


### 1.2 Grundlagen der Logik

Definition 1.5.EineAussageist ein Satz, dem genau einer der beiden Wahrheitswerte”wahr“ (w) oder

”falsch“ (f) zugeordnet werden kann.
Beispiel 1.6. 1) ”Berlin ist eine Stadt.“ (wahre Aussage)
2) ”3 + 7 = 11.“ (falsche Aussage)
3) ”Berlin!“ (keine Aussage)
4) ”x^2 −1 = 0“ ist keine Aussage, daxnicht erkl ̈art ist. Wir machen daraus Aussagen durch
Quantifizierung:
(a) ”Es gibtx∈Rmitx^2 −1 = 0.“ (wahre Aussage)
(b) ”F ̈ur allex∈Rgiltx^2 −1 = 0.“ (falsche Aussage)
DieNegationder AussageAist”nichtA“ und wird mit¬Abezeichnet. IstAwahr, so ist¬Afalsch,
und istAfalsch, so ist¬Awahr. Dies kann ̈ubersichtlich in einerWahrheitstafelfestgehalten werden:

```
A ¬A
w f
f w
```
Beispiel 1.7. 1) ”3 + 7 = 11“ ist falsch, also ist”¬(3 + 7 = 11)“ bzw.”3 + 7 6 = 11“ wahr.
2) A:”Alle Schafe sind weiß.“
¬A:”Nicht alle Schafe sind weiß.“
Anders formuliert bedeutet¬A:”Es gibt ein Schaf, das nicht weiß ist.“
Wir merken uns fur die Negation: Aus ̈ ”f ̈ur alle“ wird”es gibt“ und umgekehrt.
3) B:”Es gibtx∈Rmitx^2 =−1.“ (falsche Aussage)
¬B:”F ̈ur allex∈Rgiltx^26 =−1.“ (wahre Aussage)
Die Ausssage”AundB“ ist wahr, wenn beide Aussagen einzeln wahr sind, und die Aussage”A
oder B“ ist wahr, sobald beide Aussagen oder auch nur eine von beiden wahr ist. Die zugeh ̈origen
Wahrheitstafeln sind:
A B AundB
w w w
w f f
f w f
f f f

```
A B AoderB
w w w
w f w
f w w
f f f
```
DieImplikation(Folgerung)A⇒B(lies”ausAfolgtB“ oder”wennA, dannB“) ist wie folgt definiert:

```
A B A⇒B
w w w
w f f
f w w
f f w
```
Beispiel 1.8. 1) ”Wenn heute Mittwoch ist, dann ist morgen Donnerstag.“ Diese Implikation ist
wahr (egal ob heute nun Mittwoch ist oder nicht.)
2) ”Wenn 1 = 0, dann ist 1 = 1.“ Diese Implikation ist wahr.
Zwei Aussagen sind ̈aquivalent,A⇔B, falls Sie den gleichen Wahrheitswert haben:
A B A⇔B
w w w
w f f
f w f
f f w

Die AussageA⇔Bist genau dann wahr, wenn”A⇒BundB⇒A“ wahr ist.


Beispiel 1.9.Fur alle ̈ x∈Rgilt:x= 1⇒x^2 = 1. (wahre Aussage)
F ̈ur allex∈Rgilt:x= 1⇔x^2 = 1. (falsche Aussage)
F ̈ur allex∈Rgilt: (x= 1 oderx=−1)⇔x^2 = 1. (wahre Aussage)

Bemerkung 1.10. DieAquivalenz ist besonders wichtig beim L ̈ ̈osen von Gleichungen. Multiplizieren
wir zum Beispiel beide Seiten der Gleichung

```
2 x= 4
```
(f ̈ur reellesx) mit^12 , so erhalten wir die ̈aquivalente Gleichungx= 2, also

```
2 x= 4⇔x= 2.
```
Die Aussagen sind ̈aquivalent, da wir hin und zur ̈uck kommen. Multiplizieren wir jedoch mit 0 (Null),
so erhalten wir 0·x= 0. Nun gilt keineAquivalenz mehr sondern nur noch die Folgerung ̈

```
2 x= 4⇒ 0 ·x= 0.
```
Hier kommen wir nicht zur ̈uck (Division mit 0 ist nicht definiert), und die L ̈osungsmenge hat sich
ver ̈andert: links erf ̈ullt nurx= 2 die Gleichung, rechts tun es allex∈R.

Beweistechniken:
1) Logisches Folgern (direkter Beweis): WennAwahr ist, und wirBausAfolgern, d.h.A⇒Bist
wahr, dann ist auchBwahr.
2) Kontraposition:A⇒Bist genau dann wahr, wenn¬B⇒¬Awahr ist.
Beispiel 1.11.Wir zeigen: F ̈ur allen∈Ngilt:n^2 ist gerade⇒nist gerade.
Wir zeigen dies, indem wir”nist ungerade⇒n^2 ist ungerade“ beweisen (nachrechnen):nist
ungerade, d.h.n= 2k+ 1 f ̈ur eink∈N. Dann folgt
n^2 = (2k+ 1)^2 = 4k^2 + 4k+ 1 = 2(2k^2 + 2k) + 1,
d.h.n^2 ist ungerade.
3) Beweis durch Widerspruch: Man nimmt an, dass die Aussage, die man zeigen m ̈ochte, falsch ist,
und f ̈uhrt das zu einem Widerspruch.
Beispiel 1.12.Wir zeigen, dass
√
2 keine rationale Zahl ist.
Wir fuhren einen Beweis durch Widerspruch. Dazu nehmen wir an, dass ̈
√
2 eine rationale Zahl
ist. Dann k ̈onnen wir
√
2 als gekurzten Bruch darstellen: ̈
√
2 =mnmit ganzen Zahlenm,n, die
keinen gemeinsamen Teiler haben. Dann istm=
√
2 n, und nach Quadrierenm^2 = 2n^2. Daher
istm^2 eine gerade Zahl, und dann auchmnach Beispiel 1.11. Also istm= 2kf ̈ur eine ganze
Zahlk. Nun erhalten wir 2n^2 =m^2 = (2k)^2 = 4k^2 , alson^2 = 2k^2. Damit istn^2 eine gerade Zahl,
also auch√ n(Beispiel 1.11). Somit sindmundnbeide gerade Zahlen, im Widerspruch dazu, dass
2 =mnein gek ̈urzter Bruch war. Wir haben also gezeigt: Die Annahme, dass
√
2 rational ist,
f ̈uhrt auf einen Widerspruch. Daher ist die Annahme falsch und
√
2 ist keine rationale Zahl.


Vorlesung 2

## 2 Zahlen

Wir erinnern an die nat ̈urlichen, ganzen, rationalen und reellen Zahlen, sowie an Un-
gleichungen und den Absolutbetrag. Weiter werden Summen- und Produktzeichen ein-
gef ̈uhrt.

### 2.1 Zahlen und Zahldarstellung

Eine wichtige Aufgabe ist das L ̈osen von Gleichungen der Form

```
ax+b= 0, ax^2 +bx+c= 0,
```
und allgemeiner
anxn+an− 1 xn−^1 +...+a 1 x+a 0 = 0,

wobeixgesucht unda,b,c,an,an− 1 ,...,a 0 gegeben sind. Beginnend bei den naturlichen ̈
Zahlen fuhrt dies auf die ganzen, rationalen, reellen und schließlich die komplexen Zahlen, ̈
in denen jede solche Gleichung L ̈osungen hat.
Die naturlichen Zahlen dienen dem Z ̈ ̈ahlen von Dingen. Die Menge der naturlichen ̈
Zahlen bezeichnen wir mit
N:={ 0 , 1 , 2 , 3 ,...}.

In der Literatur findet man die nat ̈urlichen Zahlen mal mit, mal ohne die Null. Nach
DIN-Norm 5473 ist 0 eine nat ̈urliche Zahl, und daran halten wir uns hier.
InNhaben manche Gleichungen L ̈osungen, etwa hatx+ 2 = 3 die L ̈osungx= 1∈N.
Hingegen hatx+ 3 = 2 die L ̈osungx=−1, die nicht inNliegt. Dies f ̈uhrt zu denganzen
Zahlen
Z:={ 0 ,± 1 ,± 2 ,± 3 ,...}=N∪{−n|n∈N},

die eine Erweiterung der nat ̈urlichen Zahlen sind:N⊆Z. Nun hat die Gleichung 2x= 1
die L ̈osungx=^12 , die keine ganze Zahl ist. Um Teilen zu k ̈onnen, wirdZerweitert zu
denrationalen Zahlen, also Br ̈uchen von ganzen Zahlen:

```
Q:=
```
```
{m
n
```
##### ∣∣

```
∣m,n∈Z,n >^0
```
##### }

##### .


Rationale Zahlen lassen sich als Dezimalzahlen schreiben, etwa

```
1
2
```
##### = 0, 5 ,

##### 1

##### 3

##### = 0, 3 ,

##### 22

##### 7

##### = 3, 142857.

Dabei bedeutet 0,3, dass die Darstellung periodisch wird mit einer sich unendlich oft
wiederholenden 3, also 0,3 = 0, 333333 .... Ebenso wird in 3,142857 die Zahl 142857
unendlich oft wiederholt. Dieses Verhalten ist typisch. Man kann zeigen, dass die Dezi-
maldarstellung einer rationalen Zahl entweder abbricht (wie f ̈ur^12 ) oder periodisch wird
(wie f ̈ur^13 und^227 ).
Nun gibt es Zahlen wie

##### √

2, π oder die Eulersche Zahle, von denen man zeigen
kann, dass sie nicht rational sind (vgl.

##### √

2 ∈/ Q in Beispiel 1.12). Erlaubt man alle
Dezimaldarstellungen (also auch solche die weder abbrechen noch periodisch werden),
so erh ̈alt man die Menge derreellen Zahlen:

```
R:={x|xist eine Dezimalzahl}.
```
Die reellen Zahlen k ̈onnen mit der Zahlengeraden identifiziert werden: Jede reelle Zahl
entspricht einem Punkt der Zahlengeraden, und jeder Punkt auf der Zahlengeraden ist
eine reelle Zahl:

```
Zahlengerade
− 3 − 2 − 1 0 1 2 3
```
Betrachten wir die rationalen Zahlen auf der Zahlengeraden, so sehen wir, dass Q

”L ̈ocher“ hat, wie zum Beispiel

##### √

2 ∈/Q. Bei den reellen Zahlen ist das nicht mehr so,
alle Punkte der Zahlengeraden sind reelle Zahlen. In diesem Sinne hatRkeine
”
L ̈ocher“,
was sehr wichtig f ̈ur die Analysis ist.
Allerdings hat nicht jede quadratische Gleichung inRL ̈osungen: Ein Beispiel ist
x^2 + 1 = 0. Um diesen Missstand zu beheben brauchen wir eine letzte Erweiterung
unseres Zahlbereichs, von den reellen zu denkomplexen ZahlenC; siehe Vorlesung 3.
Insgesamt haben wir dann die folgenden Mengen von Zahlen:

```
N⊆Z⊆Q⊆R⊆C.
```
InQ,RundCk ̈onnen wir wie gewohnt rechnen, Differenzen bilden und durch jede Zahl
(außer Null) teilen. Dies machtQ,RundCzu sogenanntenK ̈orpern.

### 2.2 Ungleichungen

InRk ̈onnen wir Zahlen der Gr ̈oße nach vergleichen: Wir haben eineOrdnungsrelation
x < y, gelesen
”
xkleinery“. Anschaulich istx < y, fallsxlinks vonyauf der Zahlenge-
raden liegt. Anstattx < yschreiben wir auchy > x, gelesen”ygr ̈oßerx“. Wir schreiben
x≤y, gelesen
”
xkleiner oder gleichy“, fallsx < yoderx=ygilt. Analog fur ̈ y≥x.
Es gelten die folgenden Axiome (grundlegende Rechengesetze), aus denen alle weite-
ren Rechenregeln folgen: Fur alle ̈ x,y,z∈Runda,b∈Rgilt:


1) Es gilt genau einer der drei F ̈alle:x < yoderx=yoderx > y.
2) Sindx < yundy < z, so folgtx < z.
3) Sindx < yunda≤b, so folgtx+a < y+b.
4) Sindx < yunda >0, so folgtax < ay.
Die Axiome 2)–4) gelten jeweils auch, wenn man ̈uberall
”
<“ durch
”
≤“ ersetzt.
Wir sammeln weitere wichtige Rechenregeln im folgenden Satz.

Satz 2.1. F ̈ur alle reellen Zahlenx,y,agelten folgende Rechenregeln:

```
1) F ̈urx < 0 ist−x > 0. F ̈urx > 0 ist−x < 0.
```
```
2) Multiplikation mit einer negativen Zahl ̈andert die”Richtung“ einer Ungleichung:
Istx < yunda < 0 , so istax > ay.
```
```
3) F ̈urx 6 = 0istx^2 > 0. Insbesondere sind Quadrate reeller Zahlen nichtnegativ.
```
```
4) Allgemeiner gilt:
```
- Istx > 0 , so folgtxn> 0 f ̈ur allen∈N.
- Istx < 0 , so istxn

##### {

```
> 0 f ̈urn∈Ngerade
< 0 f ̈urn∈Nungerade.
```
Beweis. 1) Es ist−x≤ −x(es gilt sogar =). Fur ̈ x <0 folgt mit 3):x−x < 0 −x, also 0<−x,
also−x >0 wie behauptet. Istx >0, so folgt genauso 0 =x−x > 0 −x=−x.
2) Daa <0 ist−a >0, also (−a)x <(−a)ymit 4), d.h.−ax≤−ay. Mitax≤axund 3) erhalten
wir 0 =ax−ax≤ax−ay, und mitay≤ayund 3) folgtay= 0 +ay≤ax−ay+ay=ax, also
ax≥ay.
3) Istx >0 so folgt mit 4), dassx^2 =x·x > x·0 = 0. Ist hingegenx <0, so ist−x >0 und
dann wie ebenx^2 = (−x)·(−x)>(−x)·0 = 0. Damit haben nachgerechnet dass ausx 6 = 0 folgt
x^2 >0. F ̈urx= 0 ist nat ̈urlichx^2 = 0·0 = 0≥0.
4) F ̈ur geradesn∈Nistn= 2kmitk∈N. Dann istxn=x^2 k= (xk)^2 >0, da ausx 6 = 0 folgt
xk 6 = 0. F ̈ur ungeradesn∈Nistn= 2k+ 1 mitk∈N. Dann istxn=x^2 kx. Dax^2 k>0 ist, ist
dannxn>0 fallsx >0 undxn<0 fallsx <0.

Beispiel 2.2. 1) Es ist 3<5. Multiplikation mit 2≥0 ergibt 2·3 = 6<10 = 2·5,
vergleiche 4). Hingegen ergibt Multiplikation mit−1:− 1 ·3 =− 3 >−5 =− 1 ·5,
und Multiplikation mit−2 ergibt− 2 ·3 =− 6 >−10 =− 2 ·5.
2) F ̈urx= 3 istx^2 = 9>0 undx^3 = 27>0. F ̈urx=−3 istx^2 = (−3)(−3) = 9>0,
aberx^3 = (−3)(−3)^2 =− 3 ·9 =− 27 <0.

Beispiel 2.3(Arithmetisches und geometrisches Mittel).F ̈ura >0 undb >0 gilt

```
0 ≤(a−b)^2 =a^2 − 2 ab+b^2.
```
Addieren von 4abauf beiden Seiten ergibt 4ab≤a^2 + 2ab+b^2 = (a+b)^2 , alsoab≤
(a+b
2

```
) 2
```
. Beim
Wurzelziehen bleiben Ungleichungen zwischen positiven Zahlen erhalten, und wir bekommen
√
ab≤a+ 2 b.

Dasgeometrische Mittel
√
abist also h ̈ochstens so groß wie dasarithmetische Mittela+ 2 b.


### 2.3 Reelle Wurzeln

Definition 2.4 (Quadratwurzel). Seia≥ 0. Dann ist dieQuadratwurzel oder kurz
Wurzel

##### √

```
adie nichtnegative L ̈osung der Gleichungx^2 =a.
```
Fur ̈ a <0 hatx^2 =akeine reelle L ̈osung, denn sonst w ̈are 0≤x^2 =a <0, was ein
Widerspruch ist.

Beispiel 2.5. Es ist

##### √

```
4 = 2. Man beachte, dass
```
##### √

4 =±2 falsch ist. Es ist zwarx^2 = 4
genau dann, wennx= 2 oderx=−2 (kurz:x=±2), jedoch die Quadratwurzel die
nichtnegative L ̈osung vonx^2 = 4, also

##### √

##### 4 = 2.

```
Dien-te Wurzel vona, geschrieben n
```
##### √

```
a, ist wie folgt definiert:
```
```
1) F ̈ur geradesn≥2 unda≥0 ist n
```
##### √

```
adie nichtnegative L ̈osung vonxn=a.
```
```
2) F ̈ur ungeradesn≥1 und reellesaist n
```
##### √

```
adie reelle L ̈osung vonxn=a. Ista≥ 0
ist auch n
```
##### √

```
a≥0, ist hingegena <0, so ist auch n
```
##### √

```
a <0.
```
Beispiel 2.6.Fur ̈ n= 3 unda= 8 erhalten wir^3

##### √

```
8 = 2. F ̈ura=−8 erhalten wir
```
√ (^3) −8 =−2, denn es gilt (−2) (^3) =−8.

### 2.4 Absolutbetrag

DerAbsolutbetragoder kurzBetrageiner reellen Zahl ist definiert als

```
|x|=
```
##### {

```
x, fallsx≥ 0 ,
−x, fallsx < 0.
```
Es ist also|x|≥0 f ̈ur allex∈R(vergleiche Satz 2.1). Der Betrag entfernt also negative
Vorzeichen.

Beispiel 2.7. F ̈urx= 7≥0 ist| 7 |= 7. F ̈urx=− 7 <0 ist|− 7 |=−(−7) = 7.

Der Betrag vonxist der Abstand vonxzu 0, also der Abstand vonxzum”Ur-
sprung“ der Zahlengeraden. Allgemeiner ist|x−y|der Abstand vonxundyauf der
Zahlengeraden.
Wir sammeln einige Eigenschaften des Absolutbetrags.

Satz 2.8(Eigenschaften des Betrags).F ̈ur allex,y∈Rgilt:
1)−|x|≤x≤|x|,
2)|x|= 0 ⇔ x= 0,
3)|xy|=|x||y|, also insbesondere|−x|=|(−1)·x|=|− 1 ||x|=|x|,
4)|x|=

##### √

```
x^2 ,
5) die Dreiecksungleichung:|x+y|≤|x|+|y|,
6)|x+y|≥
```
##### ∣

```
∣|x|−|y|
```
##### ∣

##### ∣.


Beweis. Weisen Sie1)–4)zurUbung nach. Um ̈ 5)zu begr ̈unden, unterscheiden wir die beiden M ̈oglich-
keitenx+y≥0 undx+y <0. Im ersten Fallx+y≥0 gilt mit1)

```
|x+y|=x+y≤|x|+|y|.
```
Im zweiten Fallx+y <0 ist

```
|x+y|=−(x+y) =−x−y≤|−x|+|−y|=|x|+|y|.
```
Eigenschaft6)rechnen wir wie folgt nach: Wenden wir die Dreicksungleichung statt aufxundyauf die
reellen Zahlenx+yund−yan, so erhalten wir

```
|x|=|x+y−y|≤|x+y|+|−y|=|x+y|+|y|,
```
also|x|−|y|≤|x+y|. Tauschen wir die Rollen vonxundy, so erhalten wir

```
|y|−|x|≤|y+x|=|x+y|.
```
Beide Ungleichungen zusammen ergeben

```
∣∣
|x|−|y|
```
```
∣∣
≤|x+y|, wie behauptet.
```
Beispiel 2.9. 1) Es ist

##### √

##### (−5)^2 =

##### √

##### 25 = 5 =|− 5 |.

```
2) Es ist 5 =|2 + 3|≤| 2 |+| 3 |= 5 und 1 =|−2 + 3|≤|− 2 |+| 3 |= 5.
3) Es ist
```
```
1 =|−2 + 3|≥||− 2 |−| 3 ||=| 2 − 3 |=|− 1 |= 1
5 =|2 + 3|≥|| 2 |−| 3 ||=|− 1 |= 1.
```
### 2.5 Beispiele zum L ̈osen von Ungleichungen

Beispiel 2.10. Fur welche ̈ x∈Rgiltx^2 +2x >1? Zuerst mussx 6 =−2 gelten, da sonst
durch Null geteilt wird. Daher gibt es zwei M ̈oglichkeiten:x >−2 oderx <−2. Um den
Bruch aufzul ̈osen, m ̈ochten wir mitx+ 2 multiplizieren, wobei das Vorzeichen wichtig
ist.

- Fall 1:x >−2, d.h.x+ 2>0. Dann gilt:
    2 x
x+ 2
> 1 ⇔ 2 x > x+ 2⇔x > 2.

```
L ̈osungen sind also diex >−2 mitx >2, also allex >2. Als L ̈osungsmenge
geschrieben ist das
```
```
L 1 ={x∈R|x >−2 undx > 2 }={x∈R|x > 2 }= ]2,∞[.
```
```
Man kann das auch direkt mit Intervallen schreiben:
```
```
L 1 = ]− 2 ,∞[∩]2,∞[ = ]2,∞[.
```
- Fall 2:x <−2, d.h.x+ 2<0. Dann gilt
    2 x
x+ 2
> 1 ⇔ 2 x < x+ 2⇔x < 2.


```
L ̈osungen sind also diex <−2 mitx <2, d.h. allex <−2. Als L ̈osungsmenge im
zweiten Fall haben wir also
```
```
L 2 ={x∈R|x <−2 undx < 2 }={x∈R|x <− 2 }= ]−∞,−2[,
```
```
oder kurz
L 2 = ]−∞,−2[∩]−∞,2[ = ]−∞,−2[.
```
Die L ̈osungsmenge der Ungleichung ist dann

```
L=L 1 ∪L 2 = ]−∞,−2[∪]2,∞[ =R\[− 2 ,2].
```
Bei Ungleichungen mit Betr ̈agen unterscheidet man, wann das Argument des Betrags
positiv oder negativ ist.

Beispiel 2.11.Bestimme allex∈Rmit

```
|x− 2 |≤ 2 x+ 5.
```
Wegen|x− 2 |unterscheiden wir die zwei F ̈allex− 2 ≥0 undx− 2 <0.

- Fall 1:x− 2 ≥0, alsox≥2. Dann ist

```
x−2 =|x− 2 |≤ 2 x+ 5⇔− 2 ≤x+ 5⇔− 7 ≤x.
```
```
Die L ̈osungsmenge in diesem Fall ist daher
```
```
L 1 ={x∈R|x≥−7 undx≥ 2 }={x∈R|x≥ 2 }= [2,∞[.
```
- Fall 2:x− 2 <0, d.h.x <2. Dann ist

```
−x+ 2 =−(x−2) =|x− 2 |≤ 2 x+ 5⇔ 2 ≤ 3 x+ 5⇔− 3 ≤ 3 x⇔− 1 ≤x.
```
```
Die L ̈osungsmenge in diesem Fall ist
```
```
L 2 ={x∈R|− 1 ≤xundx < 2 }= [− 1 ,2[.
```
Insgesamt ist die L ̈osungsmenge der Ungleichung

```
L=L 1 ∪L 2 = [2,∞[∪[− 1 ,2[ = [− 1 ,∞[.
```
### 2.6 Summenzeichen

DasSummenzeichen

##### ∑

```
ist eine n ̈utzliche Abk ̈urzung f ̈ur die Summe mehrerer Zahlen.
```
Definition 2.12(Summenzeichen). Seienm,n∈N.

```
1) F ̈urm≤nist
∑n
```
```
k=m
```
```
xk:=xm+xm+1+...+xn.
```
```
(Lies:
”
Summe derxkfur ̈ kvonmbisn“.)
```

```
2) F ̈urm > n, d.h. die obere Grenze ist kleiner als die untere Grenze der Summe, ist
```
```
∑n
```
```
k=m
```
```
xk:= 0,
```
```
und wir sprechen von derleeren Summe.
```
```
Zum Beispiel sind
```
```
∑^10
```
```
k=1
```
```
k= 1 + 2 +...+ 9 + 10 = 1 +
```
##### ∑^10

```
k=2
```
```
k,
```
```
∑^5
```
```
k=2
```
```
k^2 = 2^2 + 3^2 + 4^2 + 5^2 =
```
##### ∑^3

```
k=2
```
```
k^2 +
```
##### ∑^5

```
k=4
```
```
k^2 ,
```
```
∑^17
```
```
k=0
```
```
2 k= 2^0 + 2^1 + 2^2 + 2^3 +...+ 2^17.
```
Dabei heißtkderSummationsindex. Welchen Buchstaben wir verwenden ist unerheblich:

```
∑n
```
```
k=m
```
```
xk=
```
```
∑n
```
```
j=m
```
```
xj=
```
```
∑n
```
```
`=m
```
```
x`.
```
Ein oft verwendeter Trick ist dieIndexverschiebung, wobei im Wesentlichen ein Index
substituiert (ersetzt) wird. Zum Beispiel kann man`=k−1 (alsok=`+ 1) ersetzen:

```
∑n
```
```
k=m
```
```
xk=xm+xm+1+...+xn=
```
```
n∑− 1
```
```
`=m− 1
```
```
x`+1=
```
```
n∑− 1
```
```
k=m− 1
```
```
xk+1.
```
Im letzten Schritt haben wir`wiederkgenannt. Analog hat man mit`=k+ 1 (also
k=`−1)
∑n

```
k=m
```
```
xk=
```
```
n∑+1
```
```
`=m+1
```
```
x`− 1 =
```
```
n∑+1
```
```
k=m+1
```
```
xk− 1.
```
Beispiel 2.13.Wir wollen diegeometrische Summe

```
∑n
```
```
k=0
```
```
qk= 1 +q+q^2 +...+qn
```
berechnen, wobeiqkdiek-te Potenz vonq∈Rbezeichnet (mitq^0 := 1). Die Summe sei
S:=

```
∑n
k=0q
```
```
k. Istq= 1, so istS=n+ 1. Fur ̈ q 6 = 1 ist
```
```
qS=q+q^2 +...+qn+1=
```
```
n∑+1
```
```
k=1
```
```
qk,
```

und daher

```
(1−q)S=S−qS=
```
```
∑n
```
```
k=0
```
```
qk−
```
```
n∑+1
```
```
k=1
```
```
qk=q^0 +
```
```
∑n
```
```
k=1
```
```
qk−
```
```
(n
∑
```
```
k=1
```
```
qk+qn+1
```
##### )

##### = 1 +

```
∑n
```
```
k=1
```
```
qk−
```
```
∑n
```
```
k=1
```
```
qk−qn+1= 1−qn+1.
```
Teilen wir noch durch 1−q 6 = 0 (es warq 6 = 1), so erhalten wirS=^1 −q

```
n+1
1 −q.
```
```
Das Ergebnis halten wir als Satz fest.
```
Satz 2.14(Geometrische Summe).F ̈ur eine reelle Zahlqundn∈Ngilt

```
∑n
```
```
k=0
```
```
qk=
```
##### {

```
n+ 1, q= 1,
1 −qn+1
1 −q , q^6 = 1.
```
Beispiel 2.15.Die geometrische Summe ist ein wichtiges Beispiel einer Summenbildung, das auch im
Alltag seine Anwendung findet: Wenn man mit einer Bank einen Sparplan ̈ubernJahre abschließt,
bei dem zum Beginn eines jeden Jahres der feste BetragBeingezahlt wird bei einer Verzinsung vonp
Prozent auf den jeweils am Ende eines Jahres insgesamt vorliegenden Betrag, kann man mit Hilfe der
geometrischen Summe berechnen, wieviel Geld man nach Ablauf des Sparplanes erhalten wird.
Der zu Beginn des ersten Jahres eingezahlte Betrag wird am Ende des Jahres verzinst, d.h. mit dem
Faktorq:= 1 + 100 p multipliziert. Zu Beginn des zweiten Jahres wird wieder der BetragBeingezahlt
und die Gesamtsumme am Ende des Jahres verzinst, die dann (Bq+B)q=Bq^2 +Bqbetr ̈agt. Im
dritten Jahr wird wieder der BetragBeingezahlt, und die Gesamtsumme am Jahresende verzinst, die
dann (Bq^2 +Bq+B)qbetr ̈agt. Bei Ablauf des Sparplanes nachnJahren wird dann der folgende Betrag
ausgezahlt:

```
Bqn+Bqn−^1 +...+Bq=Bq(qn−^1 +...+q+ 1) =Bq
```
```
n∑− 1
```
```
k=0
```
```
qk=Bq^1 −q
```
```
n
1 −q.
```
```
Wir sammeln noch einige Rechenregeln:
```
```
∑n
```
```
k=m
```
```
xk+
```
```
∑n
```
```
k=m
```
```
yk=
```
```
∑n
```
```
k=m
```
```
(xk+yk),
```
```
a
```
```
∑n
```
```
k=m
```
```
xk=
```
```
∑n
```
```
k=m
```
```
(axk),
```
```
∑n
```
```
k=m
```
```
xk
```
```
∑q
```
```
j=p
```
```
yj=
```
```
∑n
```
```
k=m
```
```
∑q
```
```
j=p
```
```
xkyj=
```
```
∑q
```
```
j=p
```
```
∑n
```
```
k=m
```
```
xkyj=
```
##### ∑

```
k=m,...,n
j=p,...,q
```
```
xkyj,
```
```
∑n
```
```
k=m
```
```
xk=
```
```
∑p
```
```
k=m
```
```
xk+
```
```
∑n
```
```
k=p+1
```
```
xk, fallsm≤p≤n.
```

Beispiel 2.16.Uberlegen Sie sich, welche Regel in welchem der folgenden Rechenschritte ̈
verwendet wird:

```
∑^9
```
```
k=3
```
```
(k^2 + 4) + 4
```
##### ∑^6

```
j=3
```
```
j+
```
##### ∑^9

```
`=7
```
##### (4`) =

##### ∑^9

```
k=3
```
```
(k^2 + 4) + 4
```
##### ∑^6

```
k=3
```
```
k+
```
##### ∑^9

```
k=7
```
```
(4k)
```
##### =

##### ∑^9

```
k=3
```
```
(k^2 + 4) +
```
##### ∑^6

```
k=3
```
```
(4k) +
```
##### ∑^9

```
k=7
```
```
(4k)
```
##### =

##### ∑^9

```
k=3
```
```
(k^2 + 4) +
```
##### ∑^9

```
k=3
```
```
(4k)
```
##### =

##### ∑^9

```
k=3
```
```
((k^2 + 4) + 4k) =
```
##### ∑^9

```
k=3
```
```
(k^2 + 4k+ 4) =
```
##### ∑^9

```
k=3
```
```
(k+ 2)^2
```
##### = 5^2 + 6^2 +...+ 11^2 =

##### ∑^11

```
k=5
```
```
k^2.
```
### 2.7 Produktzeichen

Genauso wie Summen k ̈onnen wir auch Produkte betrachten.

Definition 2.17(Produktzeichen).Seienm,n∈N.

```
1) F ̈urm≤ndefiniert man
∏n
```
```
k=m
```
```
xk:=xm·xm+1·...·xn.
```
```
2) Fallsm > n, definiert man dasleere Produktals
∏n
```
```
k=m
```
```
xk:= 1.
```
```
Zum Beispiel sind
∏^4
```
```
k=1
```
```
2 k= 2^1 · 22 · 23 · 24 = 21+2+3+4= 2^10 ,
```
##### ∏^1

```
k=3
```
```
k= 1.
```
F ̈ur Produkte gelten ganz ̈ahnliche Rechenregeln wie f ̈ur Summen.Uberlegen Sie sich ̈
zurUbung, wie diese aussehen. ̈

Definition 2.18(Fakult ̈at). F ̈urn∈Ndefinieren wirnFakult ̈atdurchn!:=

```
∏n
k=1k.
Insbesondere ist 0! =
```
##### ∏ 0

```
k=1= 1 (leeres Produkt) und f ̈urn≥1 ist
n! = 1· 2 ·...·n.
```
Damit ist zum Beispiel 1! = 1, 3! = 1· 2 ·3 = 6, 4! = 24, 5! = 120, 7! = 5040 und
10! = 3628800. F ̈ur wachsendesnw ̈achstn! sehr schnell.



Vorlesung 3

## 3 Komplexe Zahlen

Wir lernen die komplexen Zahlen kennen. Diese haben viele Vorteile, zum Beispiel
ist es immer m ̈oglich, Wurzeln zu ziehen, und quadratische Gleichungen haben immer
L ̈osungen. Zudem lassen sich physikalische Schwingungsprozesse hervorragend mit kom-
plexen Zahlen beschreiben.

### 3.1 Definition und Grundrechenarten

In den reellen ZahlenRhat die Gleichungx^2 −1 = 0 die zwei L ̈osungenx= 1 und
x=−1. Hingegen hatx^2 + 1 = 0 keine reelle L ̈osung, denn f ̈urx∈Rgilt immerx^2 ≥ 0
und damitx^2 + 1≥ 1 >0. Das ist unbefriedigend. Um dennoch L ̈osungen zu haben,
definieren wir dieimagin ̈are Einheit^1 ials eine Zahl miti^2 =−1. Diekomplexen Zahlen
(komplex = zusammengesetzt) sind dann Zahlen der Formz=x+iymit reellenxund
y, und die Menge der komplexen Zahlen ist

```
C:={x+iy|x,y∈R}.
```
F ̈ur komplexe Zahlena+ibundc+idmit (a,b,c,d∈R) sei

```
(a+ib) + (c+id):= (a+c) +i(b+d),
(a+ib)·(c+id):=ac+iad+ibc+︸︷︷︸i^2
=− 1
```
```
bd= (ac−bd) +i(ad+bc).
```
Man kann nachrechnen: InCgelten f ̈ur + und·die gleichen Rechenregeln wie inR.
Wegenz =x+i0 = xgiltR⊆C, d.h.Cist eine Erweiterung vonR. Es wird
sich zeigen, dass inCnicht nurx^2 + 1 = 0 eine L ̈osung hat, sondernjedequadratische
Gleichung.

(^1) In der Elektrotechnik verwendet man oft das Symboljstatti, um nicht mit der Standardbezeichnung
fur den Strom in Konflikt zu kommen. ̈


```
Wir vereinbaren ein paar vereinfachende Schreibweisen:
```
- Anstattx+iyschreiben wir auchx+yi,
- x+i0 =xist reell,
- 0 +iy=iyist einerein imagin ̈are Zahl,
- 5 + 1i= 5 +i,
- 5 + (−2)i= 5− 2 i.
Das Rechnen mit komplexen Zahlen ist genauso einfach wie das Rechnen mit reellen
Zahlen, man beachte nur, dassi^2 =−1 gilt.

Beispiel 3.1. 1) Addition: (2 + 4i) + (1− 3 i) = 2 + 1 + 4i− 3 i= 3 +i.
2) Subtraktion: (2 + 4i)−(1− 3 i) = 2 + 4i−1 + 3i= 1 + 7i.
3) Multiplikation:

(2 + 4i)·(1− 3 i) = 2(1− 3 i) + 4i(1− 3 i) = 2− 6 i+ 4i− (^12) ︸︷︷︸i^2
=− 12
= 14− 2 i.
4) (x+iy)(x−iy) =x^2 −ixy+ixy−i^2 y^2 =x^2 +y^2 ∈R.
5) Division durch Erweitern des Bruchs und Binomische Formel (a+b)(a−b) =a^2 −b^2 :
2 + 4i
1 − 3 i

##### =

```
(2 + 4i)(1 + 3i)
(1− 3 i)(1 + 3i)
```
##### =

```
2 + 6i+ 4i+ 12i^2
1 − 9 i^2
```
##### =

```
−10 + 10i
1 + 9
=−1 +i.
```
```
6) Potenzen der imagin ̈aren Einheiti:
```
```
i^2 =− 1 ,
i^3 =i^2 i= (−1)i=−i,
i^4 =i^2 i^2 = (−1)·(−1) = 1,
i^5 =i^4 i=i,
i^101 =i^100 i= (i^4 )^25 i= 1^25 i=i.
```
### 3.2 Die Gaußsche Zahlenebene

Reelle Zahlen entsprechen den Punkten auf der Zahlengeraden. Nach Carl Friedrich Gauß
(1777–1855) kann man sich die komplexen Zahlen als die Punkte einer Ebene vorstellen,
die die reelle Zahlengerade enth ̈alt. Die komplexe Zahlz=x+iyist dabei der Punkt
mit den kartesischen Koordinaten (x,y):

```
reelle Achse
```
```
imagin ̈are Achse
```
##### 1

```
i
```
```
0 x
```
```
iy z=x+iy
```

Die Addition komplexer Zahlen entspricht geometrisch der Addition von Vektoren in der
Ebene:

```
reelle Achse
```
```
imagin ̈are Achse
```
```
z 1 =x 1 +iy 1
```
```
z 2 =x 2 +iy 2
```
```
z 1 +z 2 = (x 1 +x 2 ) +i(y 1 +y 2 )
```
Die geometrische Deutung der Multiplikation ist in kartesischen Koordinaten nicht so
einfach. Wir kommen darauf in Vorlesung 7 zur ̈uck.

### 3.3 Absolutbetrag und Konjugierte

Definition 3.2. Seiz=x+iy∈Cmitx,y∈R.

```
1) Re(z):=xheißt derRealteilvonz,
2) Im(z):=y∈Rheißt derImagin ̈arteilvonz,
```
```
3)z:=x−iyheißt die zuz(komplex) Konjugierte,
(lieszals”zkomplex konjugiert“ oder”zquer“)
```
```
4)|z|:=
```
##### √

```
x^2 +y^2 ∈Rheißt derAbsolutbetragoder kurzBetragvonz.
```
Beispiel 3.3. F ̈urz= 2 + 3iist

```
Re(z) = 2, Im(z) = 3, z= 2− 3 i, |z|=
```
##### √

##### 22 + 3^2 =

##### √

##### 4 + 9 =

##### √

##### 13.

Real- und Imagin ̈arteil vonz=x+iygeben genau die kartesischen Koordinaten
(x,y). Beachten Sie, dass Im(z) selbst reell ist, Im(z) =y, nichtiy.
Bei der Konjugiertenzwechselt das Vorzeichen des Imagin ̈arteils vonz, zum Beispiel

```
2 − 3 i= 2 + 3i, −1 + 2i=− 1 − 2 i, − 2 i−1 = 2i− 1.
```
Geometrisch beschreibt derUbergang von ̈ zzuzdie Spiegelung an der reellen Achse.

```
iy
```
```
−iy
```
```
z=x+iy
```
```
z=x−iy
```
```
x
```

Satz 3.4(Rechenregeln f ̈ur die Konjugation). F ̈urz=x+iymitx,y∈Rgilt:

```
1)z=z,
```
```
2)zz=x^2 +y^2 =|z|^2 ,
```
```
3)z 1 +z 2 =z 1 +z 2 ,
```
```
4)z 1 z 2 =z 1 ·z 2 ,
```
```
5)^1 z=^1 z,
```
```
6)Re(z) =^12 (z+z),
```
```
7)Im(z) = 21 i(z−z).
```
Beweis. Wir rechnen nur1),2)und6)nach, die anderen verbleiben alsUbungsaufgabe. ̈
Fur ̈ 1)rechnen wir
z=x−iy=x+iy=z.

Fur ̈ 2)rechnen wir

```
zz= (x+iy)(x−iy) =x^2 −ixy+ixy−i^2 y^2 =x^2 +y^2.
```
Fur ̈ 6)fangen wir mitz+zan:

```
z+z= (x+iy) + (x−iy) = 2x= 2 Re(z),
```
also Re(z) =^12 (z+z).

Der Betrag ist geometrisch der Abstand zwischenz=x+iyund 0 in der Gaußschen
Zahlenebene. Dazu betrachtet man das rechtwinklige Dreieck mit Ecken in 0,xundz
und wendet den Satz des Pythagoras an:

```
z=x+iy
```
```
x
```
```
iy
|z|=
```
##### √

```
x^2 +y^2
```
##### 0

Genauso ̈uberlegt man sich, dass|z−w|der Abstand zwischen den beiden komplexen
Zahlenzundwin der Gaußschen Zahlenebene ist.

Satz 3.5(Rechenregeln f ̈ur den Betrag).F ̈ur komplexe Zahlenz,z 1 ,z 2 gilt:

```
1)|z|=
```
##### √

```
zz,
```
```
2)|z 1 z 2 |=|z 1 |·|z 2 |,
```

##### 3)

##### ∣

```
∣∣z 1
z 2
```
##### ∣

```
∣∣=|z 1 |
|z 2 |,
4) die Dreiecksungleichung:|z 1 +z 2 |≤|z 1 |+|z 2 |.
Illustration der Dreiecksungleichung:
```
```
z 1
```
```
z 2
```
```
z 1 +z 2
```
```
|z 1 |
```
```
|z 1 +z 2 | |z^2 |
```
### 3.4 L ̈osungen quadratischer Gleichungen

Als erste Anwendung der komplexen Zahlen untersuchen wir die L ̈osungen der Gleichung

```
az^2 +bz+c= 0 (3.1)
```
mit reellena,b,cunda 6 = 0. (F ̈ura= 0 haben wirbz+c= 0, da ist alles klar.) Diese
Gleichung hat nun immer die zwei L ̈osungen

```
z 1 =
```
```
−b+
```
##### √

```
b^2 − 4 ac
2 a
und z 2 =
```
```
−b−
```
##### √

```
b^2 − 4 ac
2 a
```
##### . (3.2)

(Mit derabc- oder Mitternachstformel; mit derpq-Formel geht es ̈ahnlich, siehe unten.)
Ist die Diskriminanted:=b^2 − 4 acgr ̈oßer Null, so ist

##### √

ddie positive reelle Wurzel,
und die Gleichung (3.1) hat zwei verschiedene reelle L ̈osungen. F ̈urd= 0 ist− 2 abeine
doppelte reelle L ̈osung. Istd <0, so ist±

##### √

```
d=±i
```
##### √

−d(wobei nun−d >0), und die
Gleichung (3.1) hat zwei verschiedene komplexe L ̈osungen.
Ista 6 = 0, so k ̈onnen wir die Gleichung (3.1) durchateilen und erhalten die Gleichung

```
z^2 +pz+q= 0,
```
wobeip=b/aundq=c/agesetzt wurde. Mit derpq-Formel sind die L ̈osungen dann

```
z 1 =−
p
2
```
##### +

##### √(

```
p
2
```
##### ) 2

```
−q und z 2 =−
p
2
```
##### −

##### √(

```
p
2
```
##### ) 2

```
−q.
```
In Vorlesung 7 werden wir sehen, dass die Gleichung (3.1) auch f ̈ur komplexea,b,cdie
beiden L ̈osungen (3.2) hat. Vorher m ̈ussen wir aber komplexe Quadratwurzeln erkl ̈aren.


Beispiel 3.6.Bestimme die L ̈osungen onz^2 + 2z+ 5 = 0. Mit derpq-Formel findet man

```
z 1 , 2 =− 1 ±
```
##### √

##### 1 −5 =− 1 ±

##### √

```
−4 =− 1 ±i
```
##### √

```
4 =− 1 ± 2 i,
```
alsoz 1 =−1 + 2iundz 2 =− 1 − 2 i.


Vorlesung 4

## 4 Vollst ̈andige Induktion

In dieser Vorlesung lernen wir das Prinzip der vollst ̈andigen Induktion kennen. Als An-
wendung beweisen wir einige n ̈utzliche Summenformeln, sowie den sehr wichtigen bino-
mischen Lehrsatz.

### 4.1 Induktion

Wir werden mehrfach vor der Aufgabe stehen, eine Aussage zu beweisen, die vonn∈N
abh ̈angt. Beispiele solcher Aussagen sind:
1) Die geometrische Summe: Fur alle nat ̈ ̈urlichen Zahlenn≥0 ist

```
∑n
k=0q
k=^1 −qn+1
1 −q ,
wobeiq 6 = 1 ist.
2) F ̈ur alle nat ̈urlichen Zahlenn≥1 gilt
```
```
∑n
k=1k=
```
n(n+1)
2
3) F ̈ur alle nat ̈urlichen Zahlenn≥1 gilt:nElemente lassen sich aufn! verschiedene
Arten anordnen.
Oft lassen sich solche Aussagen mit vollst ̈andiger Induktion beweisen.
Eine Eigenschaft der nat ̈urlichen Zahlen von fundamentaler Wichtigkeit ist die fol-
gende, nicht besonders ̈uberraschende Feststellung: Wenn man bei 0 beginnend
”
immer
eins weiterz ̈ahlt“, erreicht man jede nat ̈urliche Zahl. Wenn man also eine AussageA(n)
hat, die fur ̈ n= 0 wahr ist und beim
”
Weiterz ̈ahlen“ wahr bleibt, dann gilt sie f ̈ur alle
nat ̈urlichen Zahlen. Das ist das Prinzip der vollst ̈andigen Induktion.
Das gleiche Prinzip gilt nat ̈urlich auch, wenn man mit einem
”
Startwert“n 0 ∈N
beginnt und dann”weiterz ̈ahlt“. Die Aussage gilt dann f ̈ur allen≥n 0.

Vollst ̈andige Induktion: Um eine AssageA(n) f ̈ur allen∈Nmitn≥n 0 (also ab
einem Startwertn 0 ) mit vollst ̈andiger Induktion zu beweisen, gehen wir wie folgt vor:

```
1)Induktionsanfang:Wir zeigen, dassA(n 0 ) gilt.
```
```
2)Induktionsschritt:Wir zeigen
”
A(n)⇒A(n+ 1)“, also dass wennA(n) wahr ist,
auchA(n+ 1) wahr ist.
```
Anschließend ist die Aussage f ̈ur allen≥n 0 gezeigt.


Zum besseren Verst ̈andnis teilt man den Induktionsschritt noch in zwei oder drei
Teile, n ̈amlich
1) dieInduktionsvoraussetzung:Wir nehmen an, dassA(n) fur ̈ einn≥n 0 richtig ist,
2) dieInduktionsbehauptung:”Dann gilt auchA(n+ 1)“,
3) den eigentlichenInduktionsschritt, bei dem die Induktionsbehauptung unter Be-
nutzung der Induktionsvoraussetzung bewiesen wird.
Die Induktionsbehauptung (IB) muss man nicht aufschreiben, aber es kann beim Bewei-
sen helfen, wenn man sich noch einmal klar macht, wohin man eigentlich m ̈ochte.
Fur die Schritte bei der vollst ̈ ̈andigen Induktion finden Sie manchmal auch andere
Namen: Statt Induktionsanfang (IA) sagt man auch Induktionsverankerung, statt In-
duktionsvoraussetzung (IV) auch Induktionsannahme, und statt Induktionsschritt (IS)
auch Induktionsschluss.
Als erstes Beispiel beweisen wir noch einmal die Formel f ̈ur die geometrische Summe
(siehe Satz 2.14).

Beispiel 4.1. Seiq 6 = 1 eine reelle oder komplexe Zahl. Wir zeigen mit vollst ̈andiger
Induktion, dass

```
∑n
k=0q
k=^1 −qn+1
1 −q f ̈ur alle nat ̈urlichen Zahlenngilt.
```
- Induktionsanfang: F ̈urn= 0 ist

##### ∑ 0

```
k=0q
```
k=q (^0) = 1 =^1 −q
1 −q=
1 −q0+1
1 −q.

- Induktionsvoraussetzung: Die Behauptung sei f ̈urein n≥ 0 wahr, d.h. es gilt
    ∑n
       k=0q
          k=^1 −qn+1
             1 −q.
- Induktionsschritt: Wir zeigen jetzt, dass die Behauptung auch f ̈urn+ 1 wahr ist.
    Dazu rechnen wir:
    n∑+1

```
k=0
```
```
qk=
```
```
∑n
```
```
k=0
```
```
qk+qn+1IV=
1 −qn+1
1 −q
```
```
+qn+1=
1 −qn+1
1 −q
```
##### +

```
qn+1−qn+2
1 −q
```
##### =

```
1 −qn+2
1 −q
```
##### .

Beispiel 4.2. Wir zeigen mit vollst ̈andiger Induktion, dass

```
∑n
```
```
k=1
```
```
k= 1 + 2 +...+n=
```
```
n(n+ 1)
2
```
fur alle ̈ n∈N,n≥1, gilt.
IA: F ̈urn= 1 ist

##### ∑ 1

```
k=1k= 1 =
```
```
1(1+1)
2.
IV: F ̈ur einn∈N,n≥1, gelte
```
```
∑n
k=1k= 1 + 2 +...+n=
```
```
n(n+1)
2.
IS: Wir zeigen, dass die Behauptung auch fur ̈ n+ 1 gilt:
n∑+1
```
```
k=1
```
```
k=
```
```
∑n
```
```
k=1
```
```
k+ (n+ 1)IV=
n(n+ 1)
2
```
```
+ (n+ 1) =
n(n+ 1)
2
```
##### +

```
2(n+ 1)
2
```
```
=
n(n+ 1) + 2(n+ 1)
2
```
##### =

```
(n+ 1)(n+ 2)
2
```
##### .

Nach dem Prinzip der vollst ̈andigen Induktion gilt die Formel dann f ̈ur allen≥1. Die
Formel gilt sogar f∑ ur alle ̈ n∈N, denn f ̈urn= 0 ist die leere Summe = 0 und damit
0
k=1k= 0 =

```
0(0+1)
2.
```

Beispiel 4.3. Wir zeigen mit vollst ̈andiger Induktion: F ̈ur allen∈N,n≥1, gilt

```
∏n
```
```
k=1
```
##### (

```
k+ 1
k
```
```
)k
=
```
```
(n+ 1)n
n!
```
##### .

```
IA: F ̈urn= 1 gilt
∏^1
```
```
k=1
```
##### (

```
k+ 1
k
```
```
)k
=
```
##### (

##### 1 + 1

##### 1

##### ) 1

##### = 2 =

##### 21

##### 1!

##### .

```
IV: F ̈ur einn∈N,n≥1, gelte
```
```
∏n
k=1
```
```
(k+1
k
```
```
)k
=(n+1)
```
```
n
n!.
IS: Wir zeigen, dass die Aussage dann auch f ̈urn+ 1 gilt. Es ist
n∏+1
```
```
k=1
```
##### (

```
k+ 1
k
```
```
)k
=
```
```
∏n
```
```
k=1
```
##### (

```
k+ 1
k
```
```
)k
·
```
##### (

```
n+ 1 + 1
n+ 1
```
```
)n+1
IV=(n+ 1)n
n!
```
##### (

```
n+ 2
n+ 1
```
```
)n+1
```
##### =

```
(n+ 2)n+1
n!(n+ 1)
```
##### =

```
(n+ 2)n+1
(n+ 1)!
```
##### .

Die Formel gilt auch fur ̈ n= 0, da (leeres Produkt)

##### ∏ 0

```
k=1
```
```
(k+1
k
```
```
)k
= 1 =(0+1)
```
```
0
0!.
```
Beispiel 4.4(Bernoulli-Ungleichung).Seixeine reelle Zahl mitx≥−1. Dann gilt f ̈ur
allen∈Ndie Ungleichung
(1 +x)n≥1 +nx.

Das weisen wir mit vollst ̈andiger Induktion nach.
IA: F ̈urn= 0 ist (1 +x)^0 = 1≥1 = 1 + 0·x.
IV: F ̈ur einn∈Nsei (1 +x)n≥1 +nxwahr.
IS: Wir zeigen die Behauptung f ̈urn+ 1. Es gilt 1 +x≥0, und nach Induktionsvor-
aussetzung gilt (1 +x)n≥1 +nx. Daher erhalten wir
(1 +x)n+1= (1 +x)n(1 +x)≥(1 +nx)(1 +x) = 1 +x+nx+nx^2
= 1 + (n+ 1)x+nx^2 ≥1 + (n+ 1)x.

```
Bei der letzten Absch ̈atzung haben wir verwendet, dassnx^2 ≥0 ist.
```
Beispiel 4.5.Wir beweisen mit vollst ̈andiger Induktion, dass f ̈ur jedesn∈N,n≥1, die
Anzahl der m ̈oglichen Reihenfolgen (= Anordnungen = Permutationen) vonnElementen
gleichn! ist. Wir schreiben zur Abk ̈urzung die AussageA(n):”nElemente lassen sich
auf genaun! verschiedene Weisen (linear) anordnen.“
IA: Ein einziges Element l ̈asst sich auf eine Weise anordnen, also istA(1) wahr.
IV: A(n) ist f ̈ur einn∈N,n≥1, wahr.
IS: Wir zeigen, dass dann auchA(n+ 1) wahr ist. Dazu lassen wir von denn+ 1
Elementen eines weg. Die verbleibendennElemente k ̈onnen wir nach der Induk-
tionsannahme aufn! verschiedene Weisen anordnen. In jede dieser Anordnungen
kann das weggelassene Element ann+ 1 Stellen eingef ̈ugt werden (nach einem der
nElemente oder vor dem ersten). Also ist die gesuchte Anzahl m ̈oglicher Anord-
nungen (n+ 1)·n! = (n+ 1)!.


### 4.2 Binomischer Lehrsatz

Wir beweisen nun den binomischen Lehrsatz, also die Verallgemeinerung der Binomi-
schen Formel
(a+b)^2 =a^2 + 2ab+b^2

auf (a+b)nmit beliebigemn∈N. Dazu brauchen wir einige Vorbereitungen.

Definition 4.6.Fur ̈ n,k∈Nund 0≤k≤ndefinieren wir denBinomialkoeffizienten
(
n
k

##### )

##### :=

```
n!
k!(n−k)!
```
##### .

```
Fur ̈ k 6 = 0 ergibt das
(
n
k
```
##### )

##### =

```
n(n−1)·...·(n−k+ 1)
1 · 2 ·...·k
```
##### ,

wobei Z( ̈ahler und Nenner ein Produkt auskFaktoren sind. Im Spezialfallk= 0 ist
n
0

##### )

```
=0!(nn−!0)!=nn!!= 1. Auch f ̈urk=nist
```
```
(n
n
```
##### )

##### = 1.

```
Fur die Binomialkoeffizienten gilt ̈
(
n
k
```
##### )

##### =

```
n!
k!(n−k)!
```
##### =

##### (

```
n
n−k
```
##### )

##### .

Wir rechnen weiter nach (auf Hauptnenner bringen und vereinfachen), dass
(
n
k− 1

##### )

##### +

##### (

```
n
k
```
##### )

##### =

```
n!
(k−1)!(n−k+ 1)!
```
##### +

```
n!
k!(n−k)!
```
```
=
n!k
k!(n−k+ 1)!
```
##### +

```
n!(n−k+ 1)
k!(n−k+ 1)!
```
```
=
n!(n+ 1)
k!(n−k+ 1)!
```
##### =

```
(n+ 1)!
k!(n−k+ 1)!
```
##### =

##### (

```
n+ 1
k
```
##### )

##### .

##### (4.1)

Die Formel (4.1) ist eine
”
Rekursionsformel“: Wenn man alle

```
(n
k
```
##### )

fur ein ̈ nschon be-
rechnet hat, bekommt man daraus ganz einfach die Binomialkoeffizienten

```
(n+1
k
```
##### )

. Eine
Illustration dieser Formel gibt dasPascalschen Dreieck:

```
1
1 1
1 2 1
1 3 3 1
1 4 6 4 1
```
##### ( 0

```
0
```
##### )

##### ( 1

```
0
```
##### ) ( 1

```
1
```
##### )

##### ( 2

```
0
```
##### ) ( 2

```
1
```
##### ) ( 2

```
2
```
##### )

##### ( 3

```
0
```
##### ) ( 3

```
1
```
##### ) ( 3

```
2
```
##### ) ( 3

```
3
```
##### )

##### ( 4

```
0
```
##### ) ( 4

```
1
```
##### ) ( 4

```
2
```
##### ) ( 4

```
3
```
##### ) ( 4

```
4
```
##### )

Interpretiert mankals Spalten- undnals Zeilenindex in diesem Dreieck, so ist jeder
Eintrag die Summe der beiden dar ̈uber liegenden Eintr ̈age (abgesehen von den
”
Rand-
eintr ̈agen“

```
(n
0
```
##### )

```
und
```
```
(n
n
```
##### )

##### ).


Satz 4.7(Binomischer Lehrsatz).F ̈ur allen∈Nund reelle oder komplexea,bgilt:

```
(a+b)n=
```
```
∑n
```
```
k=0
```
##### (

```
n
k
```
##### )

```
an−kbk.
```
Fur ̈ n= 2 ist das die bekannte binomische Formel (a+b)^2 =a^2 + 2ab+b^2 , f ̈urn= 3
gilt
(a+b)^3 =a^3 + 3a^2 b+ 3ab^2 +b^3.

F ̈urn= 5 haben wir die Binomialkoeffizienten (n ̈achste Zeile des Pascalschen Dreiecks):
(
5
0

##### )

##### = 1,

##### (

##### 5

##### 1

##### )

##### = 5,

##### (

##### 5

##### 2

##### )

##### = 10,

##### (

##### 5

##### 3

##### )

##### = 10,

##### (

##### 5

##### 4

##### )

##### = 5,

##### (

##### 5

##### 5

##### )

##### = 1,

daher ist
(a+b)^5 = 1a^5 + 5a^4 b+ 10a^3 b^2 + 10a^2 b^3 + 5ab^4 + 1b^5.

Beweis. Wir beweisen den binomischen Lehrsatz mit vollst ̈andiger Induktion.
IA: F ̈urn= 0 ist
∑^0
k=0

```
(
0
k
```
```
)
a^0 −kbk=
```
```
(
0
0
```
```
)
a^0 b^0 = 1 = (a+b)^0.
```
```
IV: F ̈ur einn∈Ngelte (a+b)n=
∑n
k=0
```
```
(n
k
```
```
)
an−kbk.
IS: Wir zeigen mit der Induktionsvoraussetzung, dass die Behauptung auch f ̈urn+ 1 gilt, also dass
```
```
(a+b)n+1=
```
```
n∑+1
```
```
k=0
```
```
(
n+ 1
k
```
```
)
an+1−kbk.
```
```
Dazu rechnen wir:
```
```
(a+b)n+1= (a+b)(a+b)nIV= (a+b)
```
```
∑n
k=0
```
```
(
n
k
```
```
)
an−kbk
```
```
=
```
```
∑n
k=0
```
```
(
n
k
```
```
)
an+1−kbk+
```
```
∑n
k=0
```
```
(
n
k
```
```
)
an−kbk+1.
```
```
In der zweiten Summe machen wir eine Indexverschiebung und fassen zusammen:
```
```
(a+b)n+1=
```
```
∑n
k=0
```
```
(
n
k
```
```
)
an+1−kbk+
```
```
n∑+1
k=1
```
```
(
n
k− 1
```
```
)
an+1−kbk
```
```
=an+1+
```
```
∑n
k=1
```
```
(
n
k
```
```
)
an+1−kbk+
```
```
∑n
k=1
```
```
(
n
k− 1
```
```
)
an+1−kbk+bn+1
```
```
=an+1+
```
```
∑n
k=1
```
```
((
n
k
```
```
)
+
```
```
(
n
k− 1
```
```
))
an+1−kbk+bn+1.
```
```
Mit (4.1) und
(n+1
0
```
```
)
= 1 =
(n+1
n+1
```
```
)
erhalten wir schließlich
```
```
(a+b)n+1=
```
```
(
n+ 1
0
```
```
)
an+1+
```
```
∑n
k=1
```
```
(
n+ 1
k
```
```
)
an+1−kbk+
```
```
(
n+ 1
n+ 1
```
```
)
bn+1
```
```
=
```
```
n∑+1
```
```
k=0
```
```
(
n+ 1
k
```
```
)
an+1−kbk.
```
```
Damit ist der binomische Lehrsatz bewiesen.
```


Vorlesung 5

## 5 Abbildungen

In dieser Vorlesung betrachten wir Abbildungen, Operationen mit Abbildungen und
wann Abbildungen umkehrbar sind. Zudem betrachten wir Eigenschaften reeller Funk-
tionen wie Symmetrie und Monotonie.

### 5.1 Definition

Definition 5.1(Abbildung). SeienA,BMengen. Eine AbbildungfvonAnachBist
eine Vorschrift, diejedemElementx∈Agenau einElementy=f(x)∈Bzuordnet.
Man nenntAdenDefinitionsbereichundBdenWertebereichvonf.
Schreibweisen:f:A→B,x7→y, oderf:A→B,f(x) =y.

Man nennt Abbildungen auchFunktionen, vor allem dann, wenn die Werte Zahlen
sind (Z,Q,R,C,... ).

Beispiel 5.2. 1)f(x) =x^2 , aber das ist zu ungenau, es fehlen Definitions- und Wer-
tebereich! Genauer k ̈onnen wir zum Beispiel schreiben:f:R→R,x7→x^2 , oder
f:R→R,f(x) =x^2. Der Graph der Funktion ist:

```
x
```
```
y
```
##### − 2 − 1 0 1 2

##### 1

##### 2

##### 3

##### 4

```
Da wir wissen, dassf(x) =x^2 ≥0 f ̈ur allex∈R, k ̈onnen wir auch die Abbildung
f:R→[0,∞[,x7→x^2 , betrachten.
```

```
2) Der folgende Graph zeigt keine Funktion, da zu jedemx∈]0,∞[ zweiy-Werte
vorhanden sind:
```
```
x
```
```
y
```
##### 1 2 3 4

##### − 2

##### − 1

##### 0

##### 1

##### 2

```
3) Konstante Abbildungen, zum Beispiel dieNullabbildung f :R→R,x7→0, die
jede relle Zahl auf 0 abbildet.
```
```
4)A={ 1 , 2 , 3 },B={a,b,c,d}undf:A→Bmit 17→a, 27→b, 37→a.
```
##### 1

##### 2

##### 3

##### A

```
a
```
```
b
```
```
c
d
B
```
```
f
```
```
5) SeiAeine Menge. Die Abbildung idA:A→A,a7→a, heißt dieIdentit ̈ataufA.
Dabei ist also idA(a) =af ̈ur jedesa∈A.
```
```
x
```
```
y
```
##### 0

```
Graph von idR
```
##### − 1 1

##### − 1

##### 1

Definition 5.3(Bild und Urbild). Seif:A→Beine Abbildung.

```
1) F ̈urX⊆Aheißtf(X):={f(x)|x∈X}dasBildvonXunterf. Insbesondere
istf(X)⊆B.
Speziell heißtf(A) ={f(x)|x∈A}dasBild vonf.
```
```
2) F ̈urY ⊆ B heißtf−^1 (Y) = {x ∈A |f(x) ∈ Y} dasUrbild vonY unterf.
Insbesondere istf−^1 (Y)⊆A.
```

##### X

##### A

```
f
```
##### B

```
f(X)
```
```
Bild
```
```
f−^1 (Y)
```
##### A

```
f
```
##### B

##### Y

```
Urbild
```
Beispiel 5.4. Seif:R→R,x7→x^2. Dann sind

```
f(R) = [0,∞[
```
```
x
```
```
y
```
##### − 2 − 1 0 1 2

##### 1

##### 2

##### 3

##### 4

```
f([1,2]) = [1,4]
```
```
x
```
```
y
```
##### − 2 − 1 0 1 2

##### 1

##### 2

##### 3

##### 4

```
f([− 1 ,0]) = [0,1]
```
```
x
```
```
y
```
##### − 2 − 1 0 1 2

##### 1

##### 2

##### 3

##### 4

und

```
f−^1 ({ 4 }) ={− 2 , 2 }
```
```
x
```
```
y
```
##### − 2 − 1 0 1 2

##### 1

##### 2

##### 3

##### 4

```
f−^1 ([1,4]) = [− 2 ,−1]∪[1,2]
```
```
x
```
```
y
```
##### − 2 − 1 0 1 2

##### 1

##### 2

##### 3

##### 4

Außerdem istf−^1 ([− 2 ,−1]) =∅, denn fur jede reelle Zahl ist ̈ x^2 ≥0, so dass die Werte
zwischen−2 und−1 nie erreicht werden.


### 5.2 Komposition von Abbildungen

Mehrere Abbildungen kann man nacheinander ausf ̈uhren, vorausgesetzt Werte- und De-
finitionsbereiche passen zusammen.

Definition 5.5(Komposition). F ̈ur Abbildungenf:A→Bundg:B→Cheißt die
Abbildung
g◦f:A→C, x7→g(f(x)),

dieKompositionoderVerkettungvonfundg. (Man ließtg◦fals”gnachf“ oder als

”gKringelf“.)

##### A B C

```
f g
```
```
g◦f
```
Beispiel 5.6. 1) Seienf:R→R,x7→x^2 , alsof(x) =x^2 , undg:R→R,x7→x+1,
alsog(x) =x+ 1. Dann ist

```
(g◦f)(x) =g(f(x)) =g(x^2 ) =x^2 + 1,
```
```
alsog◦f:R→R,x7→x^2 + 1. Andersherum ist
```
```
(f◦g)(x) =f(g(x)) =f(x+ 1) = (x+ 1)^2 =x^2 + 2x+ 1,
```
```
alsof◦g:R→R,x7→x^2 + 2x+ 1.
Insbesondere ist im Allgemeinenf◦g 6 =g◦f, die Reihenfolge ist wesentlich!
2) Sindf:R→[0,∞[,x7→x^2 undg: [0,∞[→[0,∞[,x7→
```
##### √

```
x, dann ist
```
```
g◦f:R→[0,∞[, x7→g(f(x)) =
```
##### √

```
x^2 =|x|.
```
Istf:A→Bundg:D→CwobeiDnur eine Teilmenge vonBist,D⊆B, so
definiert mang◦f wie vorher, fallsf(A)⊆ D. Liegt aberf(A) nicht ganz inD, so
definiert mang◦f nur fur alle ̈ x∈Amitf(x)∈D, denn nur f ̈ur diesexl ̈asst sich
g(f(x)) bilden.

Beispiel 5.7.Seienf:R→R,f(x) = 1−x^2 , undg: [0,∞[→R,g(y) =

##### √

y. Dann ist
g(f(x)) =

##### √

```
1 −x^2 nur f ̈urx∈[− 1 ,1] definiert, alsog◦f: [− 1 ,1]→R.
```
### 5.3 Umkehrbarkeit

Wir lernen, wann eine Abbildung umkehrbar (invertierbar) ist. Umkehrbarkeit einer
Abbildungferlaubt insbesondere, die Gleichungy=f(x) eindeutig nachxzu l ̈osen.


Definition 5.8(Injektiv, surjektiv, bijektiv).Seif:A→Beine Abbildung.

```
1)fheißtinjektiv, falls f ̈ur allex 1 ,x 2 ∈Agilt:f(x 1 ) =f(x 2 )⇒x 1 =x 2.
Aquivalent dazu ist: ̈ x 16 =x 2 ⇒f(x 1 ) 6 =f(x 2 ).
”Getrenntes bleibt getrennt.“
Istfinjektiv, so wird jedes Element ausBvon h ̈ochstens einem Punkt aus erreicht.
```
```
2)fheißtsurjektiv, fallsf(A) =Bist, d.h. falls zu jedemy∈Beinx∈Aexistiert
mity=f(x).
Istfsurjektiv, so wird jeder PunktyinBerreicht.
```
```
3)fheißtbijektiv, fallsfinjektiv und surjektiv ist.
```
Definition 5.9 (Umkehrabbildung). Istf :A→Binjektiv, so gibt es zu jedemy∈
f(A) genau einx∈Amitf(x) =y, f ̈ur das wirx=f−^1 (y) schreiben. Damit k ̈onnen
wir dieUmkehrabbildung(oderInverse) bilden:

```
f−^1 :f(A)→A, y7→f−^1 (y).
```
Die Umkehrabbildung macht alsof”r ̈uckg ̈angig“, und es gelten

```
f−^1 (f(x)) =x f ̈ur allex∈A,
f(f−^1 (y)) =y f ̈ur alley∈f(A).
```
Dies k ̈onnen wir noch k ̈urzer schreiben alsf−^1 ◦f= idAundf◦f−^1 = idf(A).
Man erh ̈altf−^1 (y) durch Aufl ̈osen der Gleichungy=f(x) nachx.
Istfinjektiv und surjektiv (also bijektiv), so istf(A) =B, und der Definitionsbereich
vonf−^1 ist ganzB.

Beispiel 5.10. 1)f :R→R,x7→x^2 , ist nicht injektiv, da zum Beispiel f(2) =
22 = 4 = (−2)^2 =f(−2), aber 2 6 =−2 ist.

```
2)f: [0,∞[→R,x7→x^2 , ist injektiv, denn seienx 1 ,x 2 ∈[0,∞[ mitf(x 1 ) =f(x 2 ),
dann gilt alsox^21 =x^22. Durch Wurzelziehen haben wir|x 1 |=
```
##### √

```
x^21 =
```
##### √

```
x^22 =|x 2 |,
und dax 1 ,x 2 ≥0 sind alsox 1 =x 2.
fist nicht surjektiv, daf([0,∞[) = [0,∞[, und somit die negativen reellen Zahlen
nicht erreicht werden.
Daf injektiv ist, besitztf eine Umkehrabbildung. Um diese zu berechnen, l ̈osen
wiry=f(x) =x^2 nachxauf:x=
```
##### √

```
y. Daher ist
```
```
f−^1 : [0,∞[→[0,∞[, x7→
```
##### √

```
x.
```
```
Beachten Sie den Unterschied zwischen Umkehrabbildung und Urbild:
```
- f−^1 (4) = 2 ist die Umkehrabbildung (fur Zahlen), ̈
- f−^1 ({ 4 }) ={ 2 }ist das Urbild (fur Mengen). ̈


Wir erhalten die Umkehrabbildung einer reellen Funktion durch Spiegeln des Funk-
tionsgraphen an der Winkelhalbierendeny=x. (Bei der Umkehrfunktion werden genau
die Rollen vonxundyvertauscht.)

Beispiel 5.11. Seif : [0,∞[→R,x7→x^2. Die Umkehrabbildung istf−^1 : [0,∞[→
[0,∞[,x7→

##### √

```
x.
```
```
x
```
```
y
```
##### 0

```
y=x
```
```
f(x) =x^2
```
```
f−^1 (x) =
```
##### √

```
x
```
##### 1 2 3 4

##### 1

##### 2

##### 3

##### 4

Am Beispielf:R→R,x7→x^2 , sieht man noch einmal, dass man auf Injektivit ̈at nicht
verzichten kann. Spiegelt man den Funktionsgraphen any=x, so erh ̈alt man (links)

```
x
```
```
y
```
##### 0

```
y=x
```
```
f(x) =x^2
```
##### ”

```
f−^1 “
```
##### − 4 − 3 − 2 − 1 1 2 3 4

##### − 2

##### − 1

##### 1

##### 3

```
x
```
```
y
```
##### 0

```
y=x
```
```
f(x) =x^2
```
##### −

##### √

```
x
```
##### − 4 − 3 − 2 − 1 1 2 3 4

##### − 2

##### − 1

##### 1

##### 3

und das ist keine Funktion, denn jedem Punkt wurden ̈ zweiFunktionswerte zugeordnet.
Schr ̈ankt man die Funktion auf [0,∞[ ein, so kann man sie umkehren (oben), schr ̈ankt
man sie auf ]−∞,0] ein, so kann man sie ebenfalls umkehren (rechts).

### 5.4 Eigenschaften reeller Funktionen

Wir betrachten nun noch einige Eigenschaften reeller Funktionen.

#### 5.4.1 Symmetrie

Definition 5.12(gerade und ungerade Funktionen).Die Funktionf:R→Rheißt

```
1)gerade, fallsf(−x) =f(x) f ̈ur allex∈R,
```

```
2)ungerade, fallsf(−x) =−f(x) fur alle ̈ x∈R.
```
Dies gilt genauso, wennf nur auf einer TeilmengeD⊆Rdefiniert ist, die mitxauch
−xenth ̈alt. (Zum BeispielD= [− 1 ,1], aber nicht [− 1 ,2].)

Der Graph einer geraden Funktionen ist spiegelsymmetrisch zury-Achse, der Graph
einer ungeraden Funktionen ist punktsymmetrisch zum Ursprung.

Beispiel 5.13. 1)f:R→R,f(x) =x, ist ungerade. Allgemeiner istf mitf(x) =
xkund ungerademk∈Neine ungerade Funktion.

```
2)f:R→R,f(x) =x^2 , ist gerade. Allgemeiner istf mitf(x) =xkund geradem
k∈Neine gerade Funktion.
```
```
3)f:R→R,f(x) =|x|, ist gerade.
```
```
x
```
```
y
```
##### 0

```
y=x
```
```
y=x^3
```
##### − 2 − 1 1 2

##### − 4

##### − 3

##### − 2

##### − 1

##### 0

##### 1

##### 2

##### 3

##### 4

```
x
```
```
y
```
##### 0

```
y=x^2
```
```
y=x^4
```
##### − 2 − 1 1 2

##### 1

##### 2

##### 3

##### 4

```
x
```
```
y
```
##### 0

```
y=|x|
```
##### − 2 − 1 1 2

##### 1

##### 2

#### 5.4.2 Monotonie

Definition 5.14(Monotonie).SeiA⊆Rundf:A→R. Dann heißt

```
1)fstreng monoton wachsend, wenn ausx,y∈Amitx < yfolgt, dassf(x)< f(y)
ist,
```
```
2)fmonoton wachsend, wenn ausx,y∈Amitx < yfolgt, dassf(x)≤f(y) ist,
```
```
3)fstreng monoton fallend, wenn ausx,y∈Amitx < yfolgt, dassf(x)> f(y) ist,
```
```
4)fmonoton fallend, wenn ausx,y∈Amitx < yfolgt, dassf(x)≥f(y) ist.
```
Fur differenzierbare Funktionen gibt die erste Ableitung Auskunft ̈ ̈uber das Mono-
tonieverhalten (siehe Satz 23.6).

Beispiel 5.15. 1)f :R→R,x7→x+ 1, ist streng monoton wachsend, denn sind
x < yinR, so folgtx+ 1< y+ 1, alsof(x)< f(y).

```
2) Die konstante Funktionf:R→R,x7→1, ist sowohl monoton wachsend als auch
monoton fallend.
```

```
3)f:R→R,x7→x^2 , ist weder (streng) monoton wachsend noch (streng) monoton
fallend, den z.B. f ̈urx = − 1 < 0 = y istf(−1) = 1 > 0 = f(0) aber f ̈ur
x= 0<1 =yistf(0)< f(1).
Daf ̈ur istfstreng monoton fallend auf ]−∞,0] und streng monoton wachsend auf
[0,∞[.
```
```
Strenge Monotonie impliziert Injektivit ̈at.
```
Satz 5.16.SeiA⊂Rundf:A→Rstreng monoton. Dann istf injektiv und damit
umkehrbar.

Beweis. Wir nehmen zuerst an, dassfstreng monoton wachsend ist, und zeigen, dass
finjektiv ist. Dazu seienx 6 =yausA. Dann gilt entwederx < yund dannf(x)< f(y)
(streng monoton wachsend), oderx > yund dannf(x)> f(y). In beiden F ̈allen ist also
f(x) 6 =f(y). Damit ist gezeigt, dassf injektiv ist.
Fur eine streng monoton fallende Funktion geht es genauso. ̈

#### 5.4.3 Beschr ̈anktheit

Definition 5.17(Beschr ̈anktheit). Eine Funktionf:A→Rheißt

```
1)nach oben beschr ̈ankt, wenn eine ZahlM ∈Rexistiert mitf(x)≤ M fur alle ̈
x∈A,
```
```
2)nach unten beschr ̈ankt, wenn eine Zahlm∈Rexistiert mitf(x)≥mfur alle ̈
x∈A,
```
```
3)beschr ̈ankt, wennfnach oben und nach unten beschr ̈ankt ist, also wenn esm,M∈
Rgibt mitm≤f(x)≤Mf ̈ur allex∈A.
```
Fur Beschr ̈ ̈anktheit reicht es nachzurechnen, dass es einM≥0 gibt mit|f(x)|≤M
fur alle ̈ x∈A. Dann ist n ̈amlich−M ≤f(x)≤Mf ̈ur allex∈A. Ist andersherum
f beschr ̈ankt, d.h.m≤f(x)≤M, so ist|f(x)| ≤max{|m|,|M|}, wobei max{...}die
gr ̈oßere der beiden Zahlen|m|und|M|bezeichnet.

Beispiel 5.18. 1) Die Funktionf :R→R,x7→x^2 , ist nach unten durchm= 0
beschr ̈ankt, aber nicht nach oben beschr ̈ankt.
2) Die Funktionf: [0,4]→R,x7→x^2 , ist beschr ̈ankt: Es ist 0≤x^2 ≤16 f ̈ur alle
x∈[0,4].
3) Die Funktionf: ]− 1 ,0[→R,x7→^1 x, ist nach oben beschr ̈ankt:f(x)≤ −1, aber
nicht nach unten beschr ̈ankt.


Vorlesung 6

## 6 Elementare Funktionen

In dieser Vorlesung lernen wir dieelementaren Funktionen exp, ln, cos, sin und tan
kennen. Weitere elementare Funktionen werden wir in den Vorlesungen 8, 9, 26 und 27
behandeln.

### 6.1 Exponentialfunktion und Logarithmus

Wir beginnen mit derExponentialfunktionexp :R→R. Wir schreiben auch exp(x) =ex.
Der Funktionsgraph der Exponentialfunktion sieht wie folgt aus:

```
x
```
```
y
```
##### − 4 − 3 − 2 − 1 0 1 2

##### 1

##### 2

##### 3

##### 4

##### 5

##### 6

##### 7

Satz 6.1(Eigenschaften der Exponentialfunktion). F ̈ur die Exponentialfunktion gilt:
1) e^0 = 1.
2) Funktionalgleichung:ex+y=exeyf ̈ur allex,y∈R.
3) ex 6 = 0unde−x=e^1 x f ̈ur allex∈R.
4) ex> 0 f ̈ur allex∈R.
5) expist streng monoton wachsend.
6) exp :R→]0,∞[ist bijektiv.

Einen Nachweis der Eigenschaften erbringen wir sp ̈ater in Vorlesung 26, wenn wir
die Exponentialfunktion vollst ̈andig definiert haben. Vorerst begn ̈ugen wir uns mit den
im Satz angegebenen Eigenschaften.


Da exp : R → ]0,∞[ eine bijektive Funktion ist, existiert eine Umkehrfunktion.
Diese heißt dernat ̈urliche Logarithmus, oder nurLogarithmusund wird mit ln oder log
bezeichnet. Insbesondere gilt

```
ln(ex) =x f ̈ur allex∈R, (6.1)
eln(x)=x f ̈ur allex∈]0,∞[. (6.2)
```
Da der Logarithmus die Umkehrfunktion zur Exponentialfunktion ist, erhalten wir seinen
Graphen durch Spiegeln des Graphen von exp an der Winkelhalbierendeny=x:

```
x
```
```
y
```
##### 0

```
y=x
```
```
Graph von ln
```
```
Graph von exp
```
##### − 4 − 3 − 2 − 1 1 2 3 4

##### − 4

##### − 3

##### − 2

##### − 1

##### 1

##### 2

##### 3

##### 4

Eigenschaften des Logarithmus erhalten wir direkt aus denen der Exponentialfunktion.

Satz 6.2(Eigenschaften des Logarithmus).F ̈ur den Logarithmus gilt:
1)ln(1) = 0.
2) Funktionalgleichung:ln(xy) = ln(x) + ln(y)f ̈ur allex,y > 0.
3)ln(^1 x) =−ln(x)f ̈ur allex > 0.
4)ln(x/y) = ln(x)−ln(y)f ̈ur allex,y > 0.
5)ln(xn) =nln(x)f ̈ur allex > 0 undn∈Z.
6)lnist streng monoton wachsend.
7)ln : ]0,∞[→Rist bijektiv.

Beweis. Durch Nachrechnen:
1)Es gilt ln(1) = ln(e^0 ) = 0.
2)F ̈urx,y >0 giltx=eln(x)undy=eln(y), siehe (6.2). Damit ist
ln(xy) = ln(eln(x)eln(y)) = ln(eln(x)+ln(y)) = ln(x) + ln(y),
wobei wir die Funktionalgleichung fur die Exponentialfunktion und (6.1) verwendet ̈
haben.
3)F ̈urx >0 ist 0 = ln(1) = ln(x/x) = ln(x) + ln(1/x), also ln(1/x) =−ln(x).
4)und5)folgen ebenfalls aus der Funktionalgleichung von ln.6)und7)gelten, da der
Logarithmus die Umkehrfunktion der Exponentialfunktion ist.


### 6.2 Die trigonometrischen Funktionen Sinus und Cosinus

Sinus und Cosinus stammen aus der Geometrie am Einheitskreis (die Punkte mitx^2 +
y^2 = 1), und werden auch alsKreisfunktionenbezeichnet.
Ein Punkt auf dem Einheitskreis ist durch Angabe eines Winkelsφzur positiven
reellen Achse (x-Achse) festgelegt:

##### 0 1

##### 1

```
φ=π 6
0 1
```
##### 1

```
φ=−π 2
```
Winkel entgegen dem Uhrzeigersinn sind (im mathematischen Sinn) positiv, und Winkel
im Uhrzeigersinn negativ. Der Winkel wird dabei im Bogenmaß gemessen, also z.B.
π= 180̂ ◦und 2π= 360̂ ◦. Bezeichnetφden Winkel im Bogenmaß undαden Winkel in
Grad, so gilt

```
φ=
```
```
π
180
α, α=
```
##### 180

```
π
φ.
```
Sinus und Cosinus: Vom Einheitskreis zur Funktion. F ̈ur einen Winkelφhaben
wir ein rechtwinkliges Dreieck mit dem Punkt (x,y) auf dem Einheitskreis und einer
Hypothenuse der L ̈ange 1:

##### 0 1

##### 1

```
(x,y)
```
```
x
```
```
y
```
```
φ
cos(φ)
```
```
sin(φ)
```
Dann sind
cos(φ) =
x
1

```
=x, sin(φ) =
y
1
```
```
=y,
```

d.h. Cosinus und Sinus sind derx-Wert und dery-Wert des Punktes auf dem Einheits-
kreis mit Winkelφ. Damit sind cos und sin f ̈ur jedesφ∈Rdefiniert. Abtragen als
Funktion liefert die folgenden Graphen:

```
φ
```
```
y
sin(φ) cos(φ)
```
```
π
2 π^32 π^2 π^52 π^3 π^72 π^4 π
− 1
```
##### 0

##### 1

Aus der Definition ersehen wir folgende Eigenschaften von Sinus und Cosinus. Hier schrei-
ben wirxstattφ.

Satz 6.3(Eigenschaften von Sinus und Cosinus).Es gelten f ̈ur allex∈R:
1)− 1 ≤cos(x)≤ 1 und− 1 ≤sin(x)≤ 1.
2)cos(−x) = cos(x), d.h. der Cosinus ist eine gerade Funktion.
3)sin(−x) =−sin(x), d.h. der Sinus ist eine ungerade Funktion.
4) Trigonometrischer Pythagoras:cos(x)^2 + sin(x)^2 = 1.
5)cos(x+ 2πk) = cos(x)undsin(x+ 2πk) = sin(x)f ̈urk∈Z, d.h.cosundsinsind
2 π-periodische Funktionen.
6)cos(x) = 0genau dann, wennx=±π 2 ,±^32 π,±^52 π,...
7)sin(x) = 0genau dann, wennx= 0,±π,± 2 π,....

Bemerkung 6.4.Wir vereinbaren folgende Schreibweise:

```
cos^2 (x) = (cos(x))^2 = cos(x)^2 , sin^2 (x) = (sin(x))^2 = sin(x)^2.
```
Zum Beispiel sieht man h ̈aufig cos^2 (x) + sin^2 (x) = 1 f ̈ur den trigonometrischen Pytha-
goras.

Satz 6.5(Additionstheoreme).F ̈ur allex,y∈Rgilt

```
cos(x+y) = cos(x) cos(y)−sin(x) sin(y),
sin(x+y) = sin(x) cos(y) + cos(x) sin(y).
```
Aus den beiden Additionstheoremen lassen sich unz ̈ahlige weitere als Spezialf ̈alle
herleiten. Setzen wir zum Beispiely=π 2 ein, so folgt

```
cos
```
##### (

```
x+
π
2
```
##### )

```
= cos(x) cos
```
```
(π
2
```
##### )

```
−sin(x) sin
```
```
(π
2
```
##### )

```
=−sin(x),
```
```
sin
```
##### (

```
x+
π
2
```
##### )

```
= sin(x) cos
```
```
(π
2
```
##### )

```
+ cos(x) sin
```
```
(π
2
```
##### )

```
= cos(x).
```
Weitere typische Anwendungen sind mity=xundy=π.


Bekannte Werte. F ̈ur einige Winkel ist der Wert von cos(x) und sin(x) exakt bekannt:

```
x 0 π 6 π 4 π 3 π 2
cos(x) 1
```
```
√
3
2
```
```
√
2
2
```
```
1
2 0
```
sin(x) (^012)
√
2
2
√
3
2 1
Wie Winkel und Werte zueinander passen kann man sich leicht am Einheitskreis merken:

##### 0

##### 1

```
φ=π 6
```
```
φ=π 4
```
```
φ=π 3
```
```
1
2
```
```
√ 2
2
```
```
√ 3
2
```
##### 1 / 2

##### √

##### 2 / 2

##### √

##### 3 / 2

Die Werte f ̈ur die entsprechenden Winkel in den anderen drei Quadranten kann man
sich mit den Symmetrieeigenschaften von Sinus und Cosinusuberlegen. ̈

Amplitude, Frequenz und Phasenverschiebung. Wir betrachten nun einige ein-
fache Modifikationen von Sinus und Cosinus, die eine herausragende Rolle spielen: Mul-
tiplikation von Sinus und Cosinus mit einer Zahl, Multiplikation des Arguments mit
einer Zahl, sowie Addition einer Konstanten zum Argument. Dabei beschr ̈anken wir uns
auf den Sinus, f ̈ur den Cosinus geht es ganz genau so. Wir betrachten eine sogenannte
Sinusschwingung, d.h. eine Funktion der Form

```
f:R→R, x7→Asin(ωx−φ).
```
Dabei heißtAdieAmplitude,ωdieFrequenz, undφdiePhasenverschiebungder Sinus-
schwingung.

Sinusschwingungen mit verschiedenen Amplituden: Die FunktionAsin(x) ist
eine Streckung oder Stauchung von sin(x) iny-Richtung. DieAmplitudeAgibt die gr ̈oße
der Streckung oder Stauchung an. Wir vergleichen sin(x) mit 2 sin(x) und^12 sin(x):


```
x
```
```
y
```
```
sin(x)
```
```
2 sin(x)
```
```
1
2 sin(πx)
2 π^32 π^2 π^52 π^3 π^72 π^4 π
```
##### − 2

##### − 1

##### 0

##### 1

##### 2

Sinusschwingungen mit verschiedenen Frequenzen: Wir betrachten die Funk-
tionen sin(x), sin(2x) und sin(3x), die wie folgt aussehen:

```
x
```
```
y
sin(x) sin(3x)sin(2x)
```
```
π
2
π 3 π
2
2 π^5 π
2
3 π^7 π
2
4 π
− 1
```
##### 0

##### 1

Die Funktion sin(2x) schwingt doppelt so schnell wie sin(x), und sin(3x) schwingt dreimal
so schnell wie sin(x). Weiter haben sin(2x) und sin(3x) die Periodenπund^23 π.
Allgemein beschreiben sin(ωx) und cos(ωx) Schwingungen mit derFrequenzω > 0
und derPeriodeT=^2 ωπ>0. Die Frequenz bewirkt also eine Streckung oder Stauchung
des Funktionsgraphen inx-Richtung.

Sinusschwingungen mit verschiedenen Phasenverschiebungen: Schließlich be-
trachten wir die Funktion sin(x−φ), bei der der Parameterφzu einer Verschiebung
des Funktionsgraphen entlang derx-Achse f ̈uhrt. Istφ >0 wird der Graph umφnach
rechts verschoben, istφ <0, wird der Graph nach links verschoben. F ̈ur diePhasenver-
schiebungenφ=π 4 undφ=−π 2 erh ̈alt man zum Beispiel folgende Funktionen:

```
x
```
```
y
sin(x) sin(x−π 4 ) sin(x+π 2 ) = cos(x)
```
```
π
2
π 3 π
2 2 π
```
```
5 π
2 3 π
```
```
7 π
2 4 π
− 1
```
##### 0

##### 1


Beispiel 6.6(Harmonische Schwingung). Die Funktion

```
f(x) =acos(ωx) +bsin(ωx) (6.3)
```
ist dieUberlagerung von zwei Schwingungen mit Frequenz ̈ ω. Gilta^2 +b^2 >0 so k ̈onnen
wir schreiben

```
f(x) =
```
##### √

```
a^2 +b^2
```
##### (

```
a
√
a^2 +b^2
```
```
cos(ωx) +
b
√
a^2 +b^2
```
```
sin(ωx)
```
##### )

Wegen (
a
√
a^2 +b^2

##### ) 2

##### +

##### (

```
b
√
a^2 +b^2
```
##### ) 2

##### = 1

ist (a/

##### √

```
a^2 +b^2 ,b/
```
##### √

a^2 +b^2 ) ein Punkt auf dem Einheitskreis und es existiert ein Win-
kelφmit

```
cos(φ) =
a
√
a^2 +b^2
```
```
und sin(φ) =
b
√
a^2 +b^2
```
Also ist zusammen mit dem Additionstheorem (Satz 6.5)

```
f(x) =
```
##### √

```
a^2 +b^2 (cos(φ) cos(ωx) + sin(φ) sin(ωx)) =
```
##### √

```
a^2 +b^2 cos(ωx−φ).
```
Daher l ̈asst sich jede Schwingung der Form (6.3) als eine reine Cosinusschwingung mit
einer Phasenverschiebung schreiben. Wegen cos(x) = sin(x+π 2 ) l ̈asst sich die Schwingung
auch als reine Sinusschwingung schreiben (mit anderer Phasenverschiebung).

### 6.3 Tangens

DerTangensist definiert als

```
tan(x) =
sin(x)
cos(x)
```
uberall dort, wo der der Cosinus nicht Null ist, also f ̈ ur alle ̈ x∈R\{π 2 +kπ|k∈Z}:

```
x
```
```
y
```
```
tan
```
```
−^32 π −π 2 π 2 32 π
```
Aus den Eigenschaften von Sinus und Cosinus folgen viele Eigenschaften des Tangens.


```
Insbesondere erhalten wir dasAdditionstheorem
```
```
tan(x+y) = 1 tan(−tan(x) + tan(x) tan(yy)).
```
Rechnen Sie das zurUbung nach! F ̈ ̈ur welchexundygilt diese Gleichung?


Vorlesung 7

## 7 Komplexe Zahlen

Wir lernen eine zweite Darstellung der komplexen Zahlen kennen, die besonders g ̈unstig
f ̈ur das Multiplizieren und Wurzelziehen ist.

### 7.1 Polardarstellung

In Vorlesung 3 haben wir komplexe Zahlen in kartesischer Darstellung kennengelernt:
z=x+iymitx,y∈R. Dabei sindxundygenau die kartesischenx- undy-Koordinaten
in der Gaußschen Zahlenebene.

```
z=x+iy
```
```
x
```
```
iy
r=|z|
```
##### 0

```
φ
```
Im eingezeichneten rechtwinkligen Dreieck haben wir

```
x=rcos(φ),
y=rsin(φ),
```
##### (7.1)

und damit
z=r(cos(φ) +isin(φ)).

Das ist diePolardarstellungvonz. Dabei sind also
1)r=|z|der Betrag vonz(Abstand zu 0) und
2)φ= arg(z) dasArgumentvonz. Das ist der Winkel zwischenzund der positiven
reellen Achse (im Bogenmaß gemessen).
Beachten Sie: Vergr ̈oßern wir das Argument um 2π, so erhalten wir den gleichen Punkt in
der Ebene. Allgemeiner gilt: Mitφist auchφ+ 2πkmitk∈Zein Argument vonz.Das
Argument ist eindeutig bis auf ganzzahlige Vielfache von 2 π.Das macht die Bezeichnung


arg(z) etwas problematisch, wir m ̈ussen uns eigentlich auf einen Bereich f ̈ur den Winkel
einigen, etwa [0, 2 π[ oder ]−π,π] oder [−π 2 ,^32 π[, ist ansonsten aber nicht schlimm.
Fur ̈ z= 0 ist das Argument nicht bestimmt. Trotzdem k ̈onnen wir dann

```
z= 0(cos(φ) +isin(φ))
```
mit einem willkurlich gew ̈ ̈ahltenφschreiben.
Wie lassen sich die kartesische und die Polardarstellung ineinander umrechnen? Sind
r,φgegeben, so berechnen wirx,y durch (7.1). Sind umgekehrtx,ygegeben, so istr
der Betrag vonz=x+iy:
r=|z|=

##### √

```
x^2 +y^2.
```
Um den Winkelφzu berechnen, teilen wir die beiden Gleichungen in (7.1) und erhalten

```
y
x
```
##### =

```
sin(φ)
cos(φ)
```
```
= tan(φ),
```
fallsx 6 = 0. (Fur ̈ x= 0 ist es ganz einfach:φ=π 2 wenny >0, undφ=−π 2 wenny <0.)
Der Tangens ist aufR\{π 2 +kπ|k∈Z}definiert undπ-periodisch, tan(φ+kπ) = tan(φ)
fur ̈ k∈Z, insbesondere also nicht injektiv:

```
x
```
```
y
```
```
tan
```
```
−^32 π −π 2 π 2 32 π
```
Eingeschr ̈ankt auf ]−π 2 ,π 2 [ ist der Tangens hingegen injektiv und besitzt daher eine Um-
kehrabbildung, denArcus Tangens

```
arctan = tan−^1 :R→
```
##### ]

##### −

```
π
2
```
##### ,

```
π
2
```
##### [

##### ,

mit dem wir das Argumentφbestimmen k ̈onnen. Es ist

```
φ= arctan
```
```
(y
x
```
##### )

##### ∈

##### ]

##### −

```
π
2
```
##### ,

```
π
2
```
##### [

##### ,

fallszin der rechten Halbebene Re(z)>0 liegt (erster und vierter Quadrant). F ̈urzin
der linken Halbebene Re(z)<0 (zweiter und dritter Quadrant) mussen wir ̈ πaddieren.


Insgesamt erhalten wir f ̈urz=x+iy 6 = 0 somitz=r(cos(φ) +isin(φ)) mit

```
r=
```
##### √

```
x^2 +y^2 ,
```
```
φ=
```
##### 

##### 

##### 

##### 

##### 

```
arctan
```
```
(y
x
```
##### )

```
fallsx > 0 ,
arctan
```
```
(y
x
```
##### )

```
+π fallsx < 0 ,
+π 2 fallsx= 0,y > 0 ,
−π 2 fallsx= 0,y < 0.
```
Das Argument ist dabei in [−π 2 ,^32 π[. M ̈ochte man das Argument in einem anderen Inter-
vall w ̈ahlen, so kann man geeignet 2πaddieren oder abziehen. W ̈ahlt man z.B. [0, 2 π[,
so ersetzt man f ̈urx >0 undy <0 die Formel durchφ= arctan(y/x) + 2π.

Beispiel 7.1. 1) Fur ̈ z 1 = 2 + 2iist

```
r=
```
##### √

##### 22 + 2^2 =

##### √

##### 4 + 4 =

##### √

##### 8 ,

```
und
tan(φ) =
y
x
```
##### =

##### 2

##### 2

##### = 1,

```
also
arctan(1) =
π
4
```
##### .

```
Wegenx= Re(z 1 ) = 2>0, ist dies das Argument vonz 1. Damit ist
```
```
z 1 = 2 + 2i=
```
##### √

##### 8

##### (

```
cos
```
```
(π
4
```
##### )

```
+isin
```
```
(π
4
```
##### ))

##### .

```
2) F ̈urz 2 =− 1 −iist
```
```
r=
```
##### √

##### (−1)^2 + (−1)^2 =

##### √

```
2 , tan(φ) =
```
```
y
x
```
##### =

##### − 1

##### − 1

##### = 1,

```
und wieder ist arctan(1) =π 4. Da Re(z 2 ) =− 1 <0 ist, mussen wir ̈ πaddieren, das
Argument vonz 2 ist alsoφ=π 4 +π=^54 π.
```
```
3) F ̈urz 3 = 3iistr=
```
##### √

02 + 3^2 = 3. Wegenx= 0 undy= 3>0 istφ=π 2.
Die Punkte in der Gaußschen Zahlenebene:

##### 1

```
i
```
```
z 1
```
```
z 2
```
```
z 3
```

```
Zur Abk ̈urzung der Schreibweise wird oft dieEuler-Formel
```
```
eiφ= cos(φ) +isin(φ)
```
verwendet, die eine Verbindung zwischen der (komplexen) Exponentialfunktion und Si-
nus und Cosinus herstellt. Sp ̈ater k ̈onnen wir das nachrechnen (Abschnitt 25.4), vorerst
nehmen wir das als Vereinfachung der Schreibweise hin. Es gelten die ̈ublichen Potenzge-
setze, insbesondere (eiφ)n=einφ. Mit der Euler-Formel k ̈onnen wir eine komplexe Zahl
schreiben als
z=x+iy=r(cos(φ) +isin(φ)) =reiφ.

Die Schreibweisereiφheißt auchEulerdarstellungvonz.

Beispiel 7.2. F ̈ur die komplexen Zahlen aus Beispiel 7.1 gilt

```
z 1 =
```
##### √

```
8 ei
```
```
π
```
(^4) , z 2 =

##### √

```
2 ei
```
```
5 π
```
(^4) , z 3 = 3ei
π
(^2).

### 7.2 Vorteile der Euler- und Polardarstellungen

Mit der Polar- und Eulerdarstellung wird die Multiplikation komplexer Zahlen ganz
einfach. Sind z 1 = r 1 (cos(φ 1 ) +isin(φ 1 )) undz 2 = r 2 (cos(φ 2 ) +isin(φ 2 )) komplexe
Zahlen in Polardarstellung, so ist

```
z 1 z 2 =r 1 (cos(φ 1 ) +isin(φ 1 ))r 2 (cos(φ 2 ) +isin(φ 2 ))
=r 1 r 2
```
##### (

```
cos(φ 1 ) cos(φ 2 )−sin(φ 1 ) sin(φ 2 ) +i(cos(φ 1 ) sin(φ 2 ) + sin(φ 1 ) cos(φ 2 ))
```
##### )

```
=r 1 r 2 (cos(φ 1 +φ 2 ) +isin(φ 1 +φ 2 )),
```
wobei wir im letzten Schritt die Additionstheoreme von Cosinus und Sinus verwendet
haben (Satz 6.5). Noch einfacher ist es mit der Eulerdarstellung:

```
z 1 z 2 =r 1 eiφ^1 r 2 eiφ^2 =r 1 r 2 ei(φ^1 +φ^2 ).
```
Damit haben wir folgenden Satz bewiesen.

Satz 7.3.Bei der Multiplikation komplexer Zahlen
1) multiplizieren sich die Betr ̈age,
2) addieren sich die Argumente.
Insbesondere lassen sich Potenzen komplexer Zahlen hervorragend in der Eulerdar-
stellung berechnen. Versuchen Sie

```
z^42 = (x+iy)^42 =...
```
zu berechnen. Selbst mit dem binomischen Satz 4.7 ist das nicht so einfach. In der
Eulerdarstellung hingegen ist es ganz einfach:

```
z^42 = (reiφ)^42 =r^42 ei^42 φ.
```

### 7.3 Komplexe Wurzeln

Wir wollen die folgende Aufgabe l ̈osen.

Aufgabe:Finden Sieallekomplexen L ̈osungenz∈Cder Gleichungzn=awobeia∈C
undn∈N,n≥1, gegeben sind.

```
Zur L ̈osung stellen wirzundain der Eulerdarstellung dar,
z=reiφ und a=seiα,
```
mit noch unbekanntem Betragr≥0 und Argumentφ, und bekanntems=|a|≥0 und
α∈R. Dann ist
rneinφ=zn=a=seiα,

und es m ̈ussen gelten:
1)rn=s, da gleiche Zahlen den gleichen Betrag haben, alsor= n

##### √

```
s.
2)nφ=α+ 2πkmitk∈Z, da das Argument eindeutig bis auf ganzzahlige Vielfache
von 2πist, also
φ=
α
n
```
##### +

```
2 π
n
```
```
k, k∈Z.
```
Das ergibt die L ̈osungen

```
zk= n
```
##### √

```
sei(
```
αn+ (^2) nπk)
= n

##### √

```
s
```
##### (

```
cos
```
##### (

```
α
n
```
##### +

```
2 π
n
k
```
##### )

```
+isin
```
##### (

```
α
n
```
##### +

```
2 π
n
k
```
##### ))

```
, k∈Z.
```
Weil aber das Argument^2 nπ(k+n) =^2 nπk+ 2πdie gleiche komplexe Zahl wie^2 nπkergibt,
gibt es nurnverschiedene L ̈osungen, und wir k ̈onnenz 0 ,z 1 ,...,zn− 1 aussuchen.
Insgesamt hatzn=amita=seiαalso genau dienL ̈osungen
zk= n

##### √

```
sei(
```
```
α
n+
2 π
nk), k= 0, 1 ,...,n− 1 ,
```
die ein regelm ̈aßigesn-Eck mit Mittelpunkt in 0 bilden. (F ̈ura= 0 istz= 0 einen-fache
L ̈osung.)

Beispiel 7.4. Wir l ̈osenzn = 1. Dazu schreiben wirz =reiφ undc= 1 = 1ei^0 in
Eulerdarstellung. Dann istrneinφ=zn= 1 = 1ei^0. Vergleich der Betr ̈age liefertrn= 1,
alsor= 1. Fur das Argument haben wir ̈ nφ= 0 + 2πk, alsoφ=^2 nπk. Dies ergibt dien
verschiedenen L ̈osungen

```
zk=ei
```
(^2) nπk
, k= 0, 1 , 2 ,...,n− 1.
Speziell fur ̈ n= 4 sind das
z 0 = 1ei^0 = 1,
z 1 = 1ei
24 π 1
=ei
π 2
=i,
z 2 = 1ei
2 π
(^42) =eiπ=− 1 ,
z 3 = 1ei
24 π 3
=ei
32 π
=−i.
F ̈urn= 6 sind die Argumenteφ=^26 πk=π 3 kmitk= 0, 1 ,...,5, also 0,π 3 ,^23 π,π,^43 π,^53 π.


```
n= 4
```
```
z 0
```
```
z 1
```
```
z 2
```
```
z 3
```
```
n= 5
```
```
z 0
```
```
z 1
z 2
```
```
z 3
z 4
```
```
n= 6
```
```
z 0
```
```
z 2 z 1
```
```
z 3
```
```
z 4 z 5
```
Die L ̈osungen sind die Eckpunkte eines regelm ̈aßigenn-Ecks.

Beispiel 7.5. Wir l ̈osenz^2 = 4i. Wie oben machen wir den Ansatz z = reiφ und
schreibenc= 4i= 4ei

```
π
```
(^2). Dann ist alsor^2 = 4, alsor= 2, und 2φ= π 2 + 2πk, also
φ=π 4 +πk,k= 0,1. Das ergibt die beiden L ̈osungen
z 0 = 2ei
π
(^4) = 2

##### (

```
cos
```
```
(π
4
```
##### )

```
+isin
```
```
(π
4
```
##### ))

##### = 2

##### (√

##### 2

##### 2

##### +

##### √

##### 2

##### 2

##### )

##### =

##### √

```
2 +i
```
##### √

##### 2 ,

```
z 1 = 2ei
54 π
= 2ei(
π 4 +π)
= 2ei
π 4
eiπ=eiπz 0 =−z 0.
```
### 7.4 L ̈osungen quadratischer Gleichungen

In Abschnitt 3.4 hatten wir die L ̈osung von quadratischen Gleichungen mit reellen Ko-
effizienten besprochen. Auf diese Einschr ̈ankung k ̈onnen wir nun verzichten, und ganz
allgemein die quadratische Gleichung

```
az^2 +bz+c= 0
```
mit komplexen Koeffizientena,b,cl ̈osen. Dabei setzen wira 6 = 0 voraus. (Fur ̈ a= 0 ist
nurbz+c= 0 zu l ̈osen.) Die Gleichung hat immer die beiden L ̈osungen

```
z 1 , 2 =
−b±
```
##### √

```
b^2 − 4 ac
2 a
```
##### .

Dabei ist das Wurzelzeichen wie folgt zu verstehen. Ist die Diskriminanted=b^2 − 4 ac
reell mitd≥0, so ist

##### √

ddie gew ̈ohnliche reelle Quadratwurzel. Andernfalls sind die
beiden Wurzeln”±

##### √

d“ die beiden L ̈osungen der Gleichung z^2 = d, also (vergleiche
Beispiel 7.5)
√
|d|eiarg(d)/^2 und

##### √

```
|d|ei(arg(d)/2+π)=−
```
##### √

```
|d|eiarg(d)/^2 ,
```
die sich nur um das Vorzeichen unterscheiden. Ist die Diskriminanted= 0, so istz 1 =z 2
und diese Nullstelle wird doppelt gez ̈ahlt.


Vorlesung 8

## 8 Polynome

Polynome kennen Sie aus der Schule als reelle Funktionen, die aus Potenzen x^0 =
1 ,x,x^2 ,...zusammengesetzt sind.
Es stellt sich heraus, dass insbesondere die Frage nach der Existenz von Nullstellen
viel einfacher wird, wenn man Polynome als Funktionen einer komplexen Zahlzbetrach-
tet. F ̈ur quadratische Polynomeaz^2 +bz+chaben wir das bereits gesehen: Im Reellen
kann es 0, 1 oder 2 reelle Nullstellen geben (Abschnitt 3.4), aber es gibt immer zwei
Nullstellen im Komplexen (Abschnitt 7.4).

Definition 8.1(Polynom, Grad). 1) EinPolynomphat die Form

```
p(z):=a 0 +a 1 z+a 2 z^2 +...+anzn=
```
```
∑n
```
```
k=0
```
```
akzk,
```
```
wobeia 0 ,a 1 ,...,an∈Cundn∈N. Dieakheißen dieKoeffizientendes Polynoms.
Dadurch wird eine Funktionp:C→C,z7→p(z) =
```
```
∑n
k=0akz
kgegeben.
```
```
Sind alleakreell, nennt manpeinreelles Polynom. Dann betrachten wir wahlweise
pals Funktionp:C→Coderp:R→R.
```
```
2) Ist der h ̈ochste Koeffizientan 6 = 0, so heißtnderGradvonp, in Zeichenn= deg(p).
Sind alle Koeffizienten null, alsopdas Nullpolynom, so setzt man deg(p) =−∞.
```
Polynome vom Grad 0 sind also konstante Funktionenp(z) =a 06 = 0. Beim Rechnen
mit dem Grad von Polynomen vereinbart man, dass−∞< nf ̈ur jedesn∈Nist, und
−∞+n=−∞. Das ist nutzlich f ̈ ̈ur die Gradformel (8.1) unten.

Beispiel 8.2. 1)p(z) = 1 +i+ 3z+ (5i−1)z^2 ist ein Polynom vom Grad 2.

```
2)p(z) = 1 + 3z− 5 z^2 +z^42 ist ein reelles Polynom (die Koeffizienten sind alle reell)
vom Grad 42.
```
```
3)p(z) = 3 ist ein konstantes Polynom vom Grad 0, undp(z) = 0 ist das Nullpolynom
mit Grad−∞.
```

### 8.1 Rechenoperationen

Mit Polynomen kann man rechnen
”
wie man es erwartet“. Seien zum Beispiel

```
p(z) = 2 + 3z+ 4z^2 und q(z) = 1 +z.
```
Addition: Bei der Additionp+qwerden die Koeffizienten vor gleichen Potenzen von
zaddiert:

```
p(z) +q(z) = (2 + 3z+ 4z^3 ) + (1 +z) = (2 + 1) + (3 + 1)z+ 4z^3 = 3 + 4z+ 4z^3.
```
Skalarmultiplikation: Bei der Skalarmultiplikationλpmitλ∈Coderλ∈Rwerden
alle Koeffizienten vonpmitλmultipliziert:

```
2 p(z) = 2(2 + 3z+ 4z^3 ) = 4 + 6z+ 8z^3.
```
Multiplikation: Die Multiplikationpqist genauso einfach, man multipliziert aus (Dis-
tributivgesetz) und sortiert nach Potenzen vonz:

```
p(z)q(z) = (2 + 3z+ 4z^3 )(1 +z) = 2 + 2z+ 3z+ 3z^2 + 4z^3 + 4z^4
= 2 + 5z+ 3z^2 + 4z^3 + 4z^4.
```
Bei der Multiplikation gilt dieGradformel

```
deg(pq) = deg(p) + deg(q). (8.1)
```
Beispiel 8.3. Es sind

- (z^2 +z+ 1) + (3z^4 + 5z^2 + 2z−4) = 3z^4 + 6z^2 + 3z−3,
- (z^2 +z+ 1) + (−z^2 + 2z−5) = 3z−4,
- 2(z^2 + 3z−1) = 2z^2 + 6z+−2,
- (1 +z)(2 + 3z) = 2 + 3z+ 2z+ 3z^2 = 2 + 5z+z^2 ,
- (2z^2 + 3z)(z^3 −z+ 2) = 2z^5 − 2 z^3 + 4z^2 + 3z^4 − 3 z^2 + 6z= 2z^5 + 3z^4 − 2 z^3 +z^2 + 6z.

Division: Dividiert man zwei Polynome so erh ̈alt man im Allgemeinen kein Polynom,
sondern einerationale Funktion, z.B. 1/z, die wir in Vorlesung 9 behandeln. Daf ̈ur gibt
es bei Polynomen die Division mit Rest (Polynomdivision), die ̈ahnlich wie die Division
mit Rest in den ganzen Zahlen funktioniert.

Satz 8.4(Division mit Rest).Sindp,qzwei Polynome undq 6 = 0, so existieren Polynome
r,smit
p=sq+r und deg(r)<deg(q).

Dabei heißtrderDivisionsrestoder kurz Rest.

Ist der Rest gleich null, so istpq =sein Polynom und man sagt, dieDivision geht
auf, undqteiltp.


Beispiel 8.5.Polynomdivision vonp(z) = 2z^3 +z^2 + 3 durchq(z) =z+ 1. Wir rechnen

```
(2z^3 +z^2 + 3)÷(z+ 1) = 2z^2 −z+ 1
2 z^3 + 2z^2
−z^2
−z^2 −z
z+ 3
z+ 1
2
```
Also ists(z) = 2z^2 −z+ 1 und der Rest istr(z) = 2 mit deg(r) = 0<1 = deg(q). Es
gilt dann

```
2 z^3 +z^2 + 3 = (2z^2 −z+ 1)(z+ 1) + 2 oder
```
```
2 z^3 +z^2 + 3
z+ 1
= 2z^2 +z+ 1 +
```
##### 2

```
z+ 1
```
### 8.2 Nullstellen von Polynomen

Eine wichtige Frage ist die nachNullstellenvon Polynomen, also von Zahlenz 0 ∈Cmit
p(z 0 ) = 0. Nullstellen von Polynomen lassen sich mit der Polynomdivision”abspalten“:
Die Division durch denLinearfaktorz−z 0 geht auf.

Satz 8.6. Istpein Polynom vom Grad n≥ 1 mit Nullstelle z 0 ∈C, so gibt es ein
Polynomsvom Gradn− 1 mit

```
p(z) = (z−z 0 )s(z).
```
Beweis. Division mit Rest f ̈urpundq(z) =z−z 0 ergibtp(z) = (z−z 0 )s(z) +r(z) mit
Polynomensundr, wobei deg(r)<deg(q) = 1, d.h.r(z) =cist konstant. Einsetzen
vonz 0 ergibt dann 0 =p(z 0 ) = (z 0 −z 0 )s(z 0 ) +c=c. Also giltp(z) = (z−z 0 )s(z). Mit
der Gradformel (8.1) folgt deg(s) =n−1.

Beispiel 8.7.Das Polynomp(z) = 2z^3 −16 hat die Nullstelle 2, dennp(2) = 2· 23 −16 =

0. Polynomdivision vonpdurchz−2 ergibt:

```
(2z^3 +0z^2 + 0z−16)÷(z−2) = 2z^2 + 4z+ 8
2 z^3 − 4 z^2
4 z^2
4 z^2 − 8 z
8 z− 16
8 z− 16
0
```
alsop(z) = (z−2)(2z^2 + 4z+ 8).


Definition 8.8(Vielfachheit einer Nullstelle).DieVielfachheitder Nullstellez 0 ist die
Zahlk∈N, so dass
p(z) = (z−z 0 )kq(z),

wobeiqein Polynom mitq(z 0 ) 6 = 0 ist.
Nullstellen mit Vielfachheit 1 nennt man aucheinfache Nullstellen, Nullstellen mit
Vielfachheitk≥2 nennt man auchmehrfache Nullstellen.

Beispiel 8.9. Das Polynomp(z) =z^3 +z^2 =z^2 (z+ 1) hat die Nullstelle 0 mit Viel-
fachheit 2 und die Nullstelle−1 mit Vielfachheit 1. Damit ist−1 eine einfache Nullstelle
und 0 eine mehrfache Nullstelle vonp.

Nun stellt sich die Frage, ob es ̈uberhaupt Nullstellen gibt. Reelle Nullstellen gibt es
nicht immer, zum Beispiel habenz^2 +1 undz^2 +z+1 keine reellen Nullstellen. Sucht man
nach komplexen Nullstellen ist das anders, dort gibt es immer Nullstellen. Das besagt
der Fundamentalsatz der Algebra.

Satz 8.10 (Fundamentalsatz der Algebra). Seip(z) =

```
∑n
k=0akz
k ein Polynom vom
```
Gradn≥ 1 , dann gibt es eine bis auf die Reihenfolge eindeutigeLinearfaktorzerlegung

```
p(z) =an(z−z 1 )(z−z 2 )·...·(z−zn) =an
```
```
∏n
```
```
k=1
```
```
(z−zk)
```
mitz 1 ,z 2 ,...,zn∈C. Daher hatpgenaunNullstellen, die aber nicht unbedingt vonein-
ander verschieden sein m ̈ussen: mehrfache Nullstellen werden entsprechend ihrer Viel-
fachheit gez ̈ahlt.

Beispiel 8.11.Das Polynomp(z) = 2z^3 −16 = (z−2)(2z^2 +4z+8) = 2(z−2)(z^2 +2z+4)
aus Beispiel 8.7 hat die Nullstellez 1 = 2. Weiter hatz^2 + 2z+ 4 die Nullstellen (pq-
Formel):

```
z 2 , 3 =−
```
##### 2

##### 2

##### ±

##### √

##### 1 −4 =− 1 ±

##### √

##### −3 =− 1 ±

##### √

```
i^2 3 =− 1 ±i
```
##### √

##### 3.

Damit hatpdie Nullstellenz 1 = 2,z 2 =−1 +i

##### √

```
3,z 3 =− 1 −i
```
##### √

3, und die Linearfak-
torzerlegung
p(z) = 2(z−2)(z−(−1 +i

##### √

```
3))(z−(− 1 −i
```
##### √

##### 3)).

Aus dem Fundamentalsatz sehen wir: ein Polynom vom Grad≤nmitn+1 Nullstellen
ist das Nullpolynom. Daraus folgt der wichtige Koeffizientenvergleich.

Satz 8.12(Koeffizientenvergleich). Sindp(z) =

```
∑n
k=0akz
k undq(z) =∑m
k=0bkz
kund
```
giltp(z) =q(z)f ̈ur allez, so sind die Koeffizienten gleich:ak=bkf ̈ur allek.
Dabei gen ̈ugt Gleichheit inN+ 1Stellen, wennNder gr ̈oßere der beiden Graden
undmist.

Bemerkung 8.13.Wir haben nun zwei Darstellungen f ̈ur Polynome kennen gelernt:

```
1)
```
```
∑n
k=0akz
```
```
k, die gut zum Rechnen ist (Addition, Subtraktion, Polynomdivision)
```
```
2)an
```
∏n
k=1(z−zk) bei der die Nullstellen sofort abzulesen sind.
Tipp:Wenn Sie Nullstellen suchen und bereits einen Faktorz−z 0 ausgeklammert haben,
multiplizieren Sie diesen auf keinen Fall aus!


### 8.3 Reelle Polynome

Nun betrachten wir noch einmal Polynome mit reellen Koeffizienten. Nach dem Funda-
mentalsatz der Algebra k ̈onnen wir diese inkomplexeLinearfaktoren zerlegen.

Beispiel 8.14. 1) Das reelle Polynomp(z) =z^2 + 1 hat die Nullstelleniund−i=i.

```
2) Das reelle Polynomp(z) =z^2 +z+ 1 hat die Nullstellenz 1 , 2 =−^1 ±i
```
√ 3
2 , d.h. auch
hier giltz 2 =z 1.
In beiden Beispiel sind die nichtreellen Nullstellen komplex konjugiert zueinander.
F ̈ur reelle Polynome ist das immer der Fall, wie wir gleich nachrechnen werden. F ̈ur
komplexe Polynome ist das nicht immer der Fall, zum Beispiel hatp(z) = (z−i)(z+2i) =
z^2 +iz+ 2 die Nullstelleniund− 2 i, hat aber keine reellen Koeffizienten.

Fur ein Polynom mit reellen Koeffizienten ( ̈ ak ∈ R, alsoak = ak) gilt mit den
Rechenregeln fur die Konjugation: ̈

```
p(z) =
```
```
∑n
```
```
k=0
```
```
akzk=
```
```
∑n
```
```
k=0
```
```
akzk=
```
```
∑n
```
```
k=0
```
```
akzk=p(z).
```
Ist dann p(z 0 ) = 0, so ist auch p(z 0 ) = p(z 0 ) = 0, d.h.die nichtreellen Nullstellen
kommen in komplex konjugierten Paarenz 0 undz 0 vor. Schreiben wirz 0 =α+iβmit
α,β∈R, so ist

```
(z−z 0 )(z−z 0 ) =z^2 −(z 0 +z 0 )z+|z 0 |^2 =z^2 − 2 αz+α^2 +β^2 = (z−α)^2 +β^2.
```
und das ist wieder ein Polynom mit reellen Koeffizienten. Fassen wir so die komplexen
Linearfaktoren zusammen, erhalten wir eine reelle Zerlegung fur reelle Polynome. ̈

Satz 8.15(Reelle Zerlegung). Seip(z) =

```
∑n
k=0akz
k ein Polynom mit reellen Koeffizi-
```
enten vom Gradn≥ 1 , dann gibt es eine bis auf die Reihenfolge eindeutige Zerlegung in
reelle Linearfaktoren und in reelle quadratische Faktoren ohne reelle Nullstellen:

```
p(z) =an(z−x 1 )·...·(z−xk)·((z−α 1 )^2 +β 12 )·...·((z−αm)^2 +β^2 m),
```
mit reellenan,x 1 ,...,xk,α 1 ,...,αmund reellenβ 16 = 0,...,βm 6 = 0.

Beispiel 8.16.Die Zerlegung in komplexe Linearfaktoren vonz^4 −1 ist

```
z^4 −1 = (z^2 −1)(z^2 + 1) = (z−1)(z+ 1)(z−i)(z+i).
```
Die entsprechende reelle Zerlegung ist

```
z^4 −1 = (z−1)(z+ 1)(z^2 + 1).
```
```
Als letzte Folgerung notieren wir:
```
Satz 8.17. Ein reelles Polynom mit ungeradem Grad hat mindestens eine reelle Null-
stelle.


### 8.4 Nullstellen berechnen

Wie findet man Nullstellen? F ̈ur Polynomep(z) =a 1 z+a 0 vom Grad 1 ist die einzige
Nullstelle−a 0 /a 1. Fur Polynome vom Grad 2 berechnet man die Nullstellen mit der ̈ pq-
oderabc-Formel (Abschnitte 3.4 und 7.4). F ̈ur Gradn= 3 undn= 4 gibt es (kom-
plizierte) Formeln, um die Nullstellen zu berechnen. F ̈ur Gradn≥5 gibt es beweisbar
keine solche Formeln. Manchmal kann man die Nullstellen trotzdem finden (z.B. f ̈ur
zn=a; Abschnitt 7.3), oder man kann eine Nullstelle raten und dann abspalten (mit
Satz 8.6). Ansonsten ist man auf numerische Methoden zum finden von Nullstellen ange-
wiesen. Zwei solche Methode werden wir sp ̈ater kennen lernen: das Bisektionsverfahren
(Abschnitt 20.1) und das Newtonverfahren (Abschnitt 22.2).


Vorlesung 9

## 9 Rationale Funktionen

Rationale Funktionen sind Br ̈uche von Polynomen, also

```
f(z) =
p(z)
q(z)
```
##### ,

wobeipundqPolynome sind. Daher ist

```
f:D→C mitD=C\{z|q(z) = 0}.
```
Fallsp,qreelle Polynome sind, k ̈onnen wirfauch als reelle Funktion betrachten, also

```
f:D→R mitD=R\{z|q(z) = 0}.
```
An einer Nullstelle des Nenners, alsoz 0 mitq(z 0 ) = 0, istfzun ̈achst nicht definiert.

```
1) Istp(z 0 ) 6 = 0, so heißtz 0 einPoloder einePolstellevonf. Wir sagen,z 0 ist ein
Pol der Ordnungkvonf, wennz 0 eine Nullstelle vonqder Vielfachheitkist. Ein
Pol der Ordnung 1 heißt aucheinfacher Pol, und ein Pol h ̈oherer Ordnung (≥2)
heißtmehrfacher Pol.
2) Ist auchp(z 0 ) = 0, so k ̈urzez−z 0 so oft wie m ̈oglich in Z ̈ahler und Nenner.
```
Beispiel 9.1. 1) Polynome sind rationale Funktionen (mitq(z) = 1 f ̈ur allez).
2)f(z) =z−^13 ist eine rationale Funktion mit einem einfachen Pol in 3.
3)f(z) =z

(^4) +3z+5
z^2 +1 ist eine rationale Funktion mit Polstelleniund−i. Wegenz
(^2) + 1 =
(z−i)^1 (z+i)^1 sind beide Polstellen einfach.
4)f(z) =z
(^4) +3z+5
z^2 +2z+1=
z^4 +3z+5
(z+1)^2 ist eine rationale Funktion mit einem Pol der Ordnung
2 inz=−1.
5)f(z) =((zz−−1)(1)(zz+1)+2)= pq((zz))hat einen einfachen Pol in−2. F ̈urz= 1 sindq(1) = 0
undp(1) = 0, und wir k ̈onnenz−1 k ̈urzen. Wir erhaltenf(z) =zz+1+2, so dassf
anschließend auch inz= 1 definiert ist.
6)f(z) =((zz−−1)1)( (^2) (zz+1)+2)=pq((zz)) hat ebenfalls einen einfachen Pol in−2. F ̈urz= 1 sind
q(1) = 0 undp(1) = 0, und wir k ̈onnenz−1 k ̈urzen. Wir erhaltenf(z) =(z−z1)(+1z+2).
Nun ist 1 immer noch eine Nullstelle des Nenners und ein einfacher Pol vonf.


Verschiedene rationale Funktionen addieren wir, indem wir sie auf den Hauptnenner
bringen, zum Beispiel

```
1
z− 1
```
##### +

##### 1

```
z− 2
```
##### =

```
(z−2) + (z−1)
(z−1)(z−2)
```
##### =

```
2 z− 3
(z−1)(z−2)
```
Umgekehrt kann man eine rationale Funktion zerlegen als Summe eines Polynoms und
einfacher rationaler Funktionen der Form

```
A
z−a
, oder allgemeiner
```
##### A

```
(z−a)k
```
Diese Darstellung heißt diePartialbruchzerlegung der rationalen Funktion. Wie man
diese berechnet lernen wir in dieser Vorlesung.

Anwendungen. Die Partialbruchzerlegung spielt eine wichtige Rolle

- bei der Integration von rationalen Funktionen (Abschnitt 31.3), und
- im Zusammenhang mit der sogenannten Laplace-Transformation, die in der Re-
    gelungstechnik, in der Netzwerktheorie, bei der Signalverarbeitung und in vie-
    len anderen Anwendungsbereichen eine zentrale Rolle spielt. (Mehr zur Laplace-
    Transformation lernen Sie in den Vorlesungen”Differentialgleichungen f ̈ur Inge-
    nieurwissenschaften“ und
       ”
          Integraltransformationen und partielle Differentialglei-
    chungen f ̈ur Ingenieurwissenschaften“.)

### 9.1 Komplexe Partialbruchzerlegung

Wir beginnen mit der komplexen Partialbruchzerlegung (PBZ).

Satz 9.2 (Komplexe Partialbruchzerlegung). Seif(z) = pq((zz)) eine rationale Funktion.
Hatqdie einfachen Nullstellenz 1 ,...,zn, also

```
q(z) =an
```
```
∏n
```
```
k=1
```
```
(z−zk) =an(z−z 1 )(z−z 2 )...(z−zn),
```
so hatfdiePartialbruchzerlegung

```
f(z) =s(z) +
```
```
∑n
```
```
k=1
```
```
Ak
z−zk
```
```
=s(z) +
```
##### A 1

```
z−z 1
```
##### +...+

```
An
z−zn
```
##### (9.1)

mit einem Polynoms(z)und KoeffizientenA 1 ,...,An∈C. Polynom und Koeffizienten
sind eindeutig bestimmt.
Istzk einemk-fache Nullstelle vonq, so wird der Summand zA−kzk ersetzt duch die
mkSummanden
Ak, 1
z−zk

##### +

```
Ak, 2
(z−zk)^2
```
##### +...+

```
Ak,mk
(z−zk)mk
```
##### , (9.2)

wobei die Koeffizienten auch eindeutig bestimmt sind.


Berechnung der Partialbruchzerlegung.
Schritt 1:Polynomdivision. Polynomdivision vonpdurchqergibtp(z) =s(z)q(z) +
r(z) mit eindeutig bestimmten Polynomensundrmit deg(r)<deg(q), also

```
f(z) =
p(z)
q(z)
=s(z) +
r(z)
q(z)
mit deg(r)<deg(q).
```
Falls deg(p)<deg(q) ist, braucht keine Polynomdivision durchgef ̈uhrt werden, dann ist
direkts(z) = 0 undr(z) =p(z). Damit ist das Polynomsin der Partialbruchzerle-
gung (9.1) berechnet.
Schritt 2:Partialbruchzerlegung vonrq((zz)). Hatqnur einfache Nullstellen, so machen
wir den Ansatz
r(z)
q(z)

##### =

```
∑n
```
```
k=1
```
```
Ak
z−zk
```
##### . (9.3)

Hatqmehrfache Nullstellen, so modifizieren wir den Ansatz wie folgt: Istzkeine Null-
stelle mit Vielfachheitmk, so ersetzen wir zA−kzk durch diemkSummanden (9.2). Die
Koeffizienten k ̈onnen wir wie folgt berechnen:
M ̈oglichkeit 1: Koeffizientenvergleich. Wir multiplizieren den Ansatz f ̈ur rq((zz)) mit
dem Nennerq(z), was eine Gleichung von Polynomen ergibt. Ein Koeffizientenvergleich
(Satz 8.12) ergibt ein lineares Gleichungssystem, aus dem wir die Koeffizienten berechnen
k ̈onnen.
M ̈oglichkeit 2: Einsetzen spezieller Werte.Wir k ̈onnen in den Ansatz f ̈urrq((zz))spezielle
Werte f ̈urzeinsetzen, um einen Teil oder alle Koeffizienten zu bestimmen.
Besonders hilfreich ist es, den Ansatz mit (z−zk)mkzu multiplizieren und anschlie-
ßendz=zkin die Gleichung einzusetzen. Das ergibt genau den KoeffizientenAk,mk. Da

man dabei nur (z−zk)mkinrq((zz))zuhalten braucht und in den Restz=zkeinsetzt, ist
diese Methode zur Bestimmung vonAk,mkauch alsZuhaltemethodebekannt.

```
Fur einfache Nullstellen sieht das wie folgt aus: Multiplizieren wir den Ansatz ̈
```
```
r(z)
(z−z 1 )(z−z 2 )...(z−zn)
```
##### =

##### A 1

```
z−z 1
```
##### +

##### A 2

```
z−z 2
```
##### +...+

```
An
z−zn
```
mitz−z 1 , so folgt

```
r(z)
(z−z 2 )...(z−zn)
=A 1 + (z−z 1 )
```
##### A 2

```
z−z 2
+...+ (z−z 1 )
```
```
An
z−zn
```
und Einsetzen vonz=z 1 ergibt

```
r(z 1 )
(z 1 −z 2 )...(z 1 −zn)
```
##### =A 1.

Dabei erhalten wirA 1 , indem wirz−z 1 inrq((zz))zuhalten, und in den Restz 1 einsetzen.
Genauso berechnet manA 2 ,...,An.


Beweis, dass die Partialbruchzerlegung existiert.Wir brauchen nur zeigen, dass der Ansatz in Schritt 2
immer m ̈oglich ist. Dies geht mit den Ergebnissen der Vorlesungen 10 und 11, auf die wir hier vorgreifen.
Wir nehmen zuerst an, dassqeinfache Nullstellen hat und wollen zeigen, dass
r(z)
q(z)=

```
∑n
k=1
```
```
Ak
z−zk (9.4)
```
gilt, oder ̈aquivalent dazu, dass

```
r(z) =
```
```
∑n
k=1
```
```
Akzq−(zz)
k
(9.5)
```
gilt. Beachten Sie, dassqk(z):=zq−(zz)k=
∏n
j=1,j 6 =k(z−zj) nach k ̈urzen des Faktorsz−zkein Polynom
vom Grad∑ n−1 ist. DiesenPolynome sind linear unabh ̈angig, denn: Sindλ 1 ,...,λn∈Cmit 0 =
n
k=1λkqk(z), so folgt durch Einsetzen vonz=z^1 , dass 0 =λ^1

∏n
j=1,j 6 =k(z^1 −zj) + 0, alsoλ^1 = 0.
Anschließend setzt man nacheinanderz=z 2 ,z 3 ,...,znein und findet, dassλ 1 =λ 2 =...=λn= 0
ist. Somit sind dienPolynomeq 1 ,...,qnlinear unabh ̈angig und bilden eine Basis desn-dimensionalen
Vektorraums der Polynome vom Grad≤n−1 (Satz 11.9). Wegen deg(r)≤n−1 istrin diesem
Vektorraum, so dass (9.5) mit eindeutig bestimmten KoeffizientenA 1 ,...,Angilt.
Hatqmehrfache Nullstellen, so modifizieren wir den Ansatz (9.4) wie folgt: F ̈ur einemk-fache
NullstellezkwirdzA−kzk ersetzt durch (9.2). Nun wird es technisch komplizierter, die Idee bleibt aber
gleich: Nach Multiplikation mitqerhalten wir wieder eine Basis vonC[z]<deg(q). Wir verzichten auf die
genaue Ausfuhrung. ̈

Beispiel 9.3. 1) Wir berechnen die Partialbruchzerlegung von

```
f(z) =
z^2 +z+ 4
(z−3)(z^2 + 1)
```
##### =

```
z^2 +z+ 4
(z−3)(z−i)(z+i)
```
##### .

```
Wegen deg(p) = 2<3 = deg(q) entf ̈allt Schritt 1 (Polynomdivision). Dafdie drei
einfachen Polstellen 3,iund−ihat, machen wir den Ansatz
```
```
f(z) =
```
##### A

```
z− 3
```
##### +

##### B

```
z−i
```
##### +

##### C

```
z+i
```
##### .

```
Die Koeffizienten k ̈onnen wir mit der Zuhaltemethode bestimmen:
```
```
A=
```
##### 32 + 3 + 4

##### 32 + 1

##### =

##### 16

##### 10

##### =

##### 8

##### 5

##### ,

##### B=

```
i^2 +i+ 4
(i−3)(i+i)
```
##### =−

##### 1

##### 2

```
3 +i
1 + 3i
```
##### =−

##### 1

##### 2

```
(3 +i)(1− 3 i)
(1 + 3i)(1− 3 i)
```
##### =−

##### 3

##### 10

##### +

##### 2

##### 5

```
i,
```
##### C=

```
(−i)^2 −i+ 4
(−i−3)(−i−i)
```
##### =

```
3 −i
6 i− 2
```
##### =−

##### 3

##### 10

##### −

##### 2

##### 5

```
i.
```
```
Die Partialbruchzerlegung ist also
```
```
f(z) =
```
```
8
5
z− 3
```
##### +

```
− 103 +^25 i
z−i
```
##### +

```
− 103 −^25 i
z+i
```
##### .

```
Statt mit der Zuhaltemethode kann man die Koeffizienten auch ̈uber einen Koef-
fizientenvergleich bestimmen: Multipliziert man den Ansatz mitqso ergibt sich
z^2 +z+ 4 =A(z^2 + 1) +B(z−3)(z+i) +C(z−3)(z−i)
= (A+B+C)z^2 + ((−3 +i)B+ (− 3 −i)C)z+ (A− 3 iB+ 3iC),
```

```
und ein Koeffizientenvergleich liefert das lineare Gleichungssystem
```
```
A+B+C= 1,
(−3 +i)B+ (− 3 −i)C= 1
A− 3 iB+ 3iC= 4.
```
```
L ̈osen wir das lineare Gleichungssystem, so erhalten wir die gleichen Koeffizienten
wie oben. (Wie man lineare Gleichungssysteme effizient l ̈ost, lernen wir in Vorle-
sung 13.)
```
```
2) Wir berechnen die Partialbruchzerlegung der rationalen Funktion
```
```
f(z) =
5 z^2 − 4 z+ 7
z^3 +z^2 − 5 z+ 3
```
##### =

```
5 z^2 − 4 z+ 7
(z−1)^2 (z+ 3)
Damit ist 1 ein doppelter Pol vonf und−3 ist ein einfacher Pol vonf. Wieder
gilt deg(p) = 2<3 = deg(q) und wir machen den Ansatz
```
```
f(z) =
```
##### A

```
z− 1
```
##### +

##### B

```
(z−1)^2
```
##### +

##### C

```
z+ 3
Die KoeffizientenBundCk ̈onnen wir mit der Zuhaltemethode bestimmen:
```
```
C=
```
##### 5 ·(−3)^2 − 4 ·(−3) + 7

##### (− 3 −1)^2

##### = 4,

##### B=

##### 5 −4 + 7

##### 1 + 3

##### = 2.

```
F ̈ur den KoeffizientenAfunktioniert dies nicht: Multiplizieren wir den Ansatz mit
z−1, so bleibt auf der rechten Seite ein einfacher Pol in 1, so dass wir nichtz= 1
einsetzen k ̈onnen. Dafur k ̈ ̈onnen wir einen anderen Wert fur ̈ zeinsetzen, oder mit
dem Nennerqmultiplizieren:
```
```
5 z^2 − 4 z+ 7 =A(z−1)(z+ 3) + 2(z+ 3) + 4(z−1)^2.
```
```
Nun hilft entweder ein Koeffizientenvergleich (vorz^2 finden wir 5 =A+ 4, also
A= 1) oder das Einsetzen eines weiteren Wertes, z.B. mitz= 0 ist 7 =− 3 A+6+4,
alsoA= 1.
```
### 9.2 Reelle Partialbruchzerlegung

Wir betrachten nun den Fall einer reellen rationalen Funktion f(x) = pq((xx)), d.h. die
Polynomepundqhaben reelle Koeffizienten. Sind alle Pole vonf reell, so funktioniert
die Partialbruchzerlegung genau wie im komplexen, und die Koeffizienten sind dann auch
alle reell.
Allerdings kann der Nenner auch komplexe Nullstellen besitzen, die dann in komplex-
konjugierten Paarenλ=a+ib,λ=a−ibauftreten (mit reellena,bundb 6 = 0). Nat ̈urlich


funktioniert die komplexe Partialbruchzerlegung wie gehabt und enth ̈alt dann komplexe
Koeffizienten und Polstellen (siehe Beispiel 9.3).
M ̈ochte man statt der komplexen eine reelle Partialbruchzerlegung, so kann man das
durch zusammenfassen der zugeh ̈origen Terme erreichen ( ̈ahnlich zur reellen Zerlegung
von reellen Polynomen).
Im Falle eines einfachen komplexen Pols kann man die zugeh ̈origen Linearfaktoren
inq(x) zusammenfassen als

```
(x−λ)(x−λ) = (x−(a+ib))(x−(a−ib)) = (x−a)^2 +b^2.
```
Die entsprechenden komplexen Summanden in der Partialbruchzerlegung kann man
ebenfalls zusammenfassen

```
A
x−(a+ib)
```
##### +

##### B

```
x−(a−ib)
```
##### =

```
Cx+D
(x−a)^2 +b^2
```
##### .

und in dieser Form ansetzen. SindAundB schon bekannt, so k ̈onnen wirC undD
direkt berechnen, indem wir die linke Seite auf den Hauptnenner bringen.
Fur Polstellen h ̈ ̈oherer Ordnung geht das so ̈ahnlich, aber es ist meist einfacher, die
Terme komplex zu lassen.Istλeine nichtreelle Polstelle der Ordnungm, so ist auchλeine nichtreelle
Polstelle der Ordnungm, und man ersetzt

```
A 1
z−λ+
```
```
A 2
(z−λ)^2 +...+
```
```
Am
(z−λ)m+
```
```
B 1
z−λ
+ B^2
(z−λ)^2
+...+ Bm
(z−λ)m
```
im Ansatz fur die Partialbruchzerlegung durch ̈

```
C 1 z+D 1
(z−a)^2 +b^2 +
```
```
C 2 z+D 2
((z−a)^2 +b^2 )^2 +...+
```
```
Cmz+Dm
((z−a)^2 +b^2 )m
```
mit reellenC 1 ,D 1 ,C 2 ,D 2 ,...,Cm,Dm.

Beispiel 9.4.Seif(z) = z

(^2) +z+4
(z−3)(z^2 +1)aus Beispiel 9.3. Wir machen nun den reellen Ansatz
z^2 +z+ 4
(z−3)(z^2 + 1)

##### =

##### A

```
z− 3
```
##### +

```
Cz+D
z^2 + 1
```
##### .

Mutliplikation mit dem Nenner ergibt

```
z^2 +z+ 4 =A(z^2 + 1) + (Cz+D)(z−3) = (A+C)z^2 + (− 3 C+D)z+A− 3 D,
```
also (Koeffizientenvergleich)

```
A+C= 1
− 3 C+D= 1
A− 3 D= 4.
```
L ̈ost man das lineare Gleichungssystem, so findet manA=^85 ,C=−^35 ,D=−^45. Hat
man schon die komplexe Zerlegung, kann man auch die komplexen Pole zusammenfassen.


### 9.3 Zusammenfassung

Wir fassen das Vorgehen zur Berechnung der Partialbruchzerlegung zusammen.

Schritt 1: Polynomdivision. Nur erforderlich, falls deg(p)≥deg(q): Polynomdivi-
sionp(z) =s(z)q(z) +r(z) mit deg(r)<deg(q).
Falls deg(p)<deg(q) ists(z) = 0 undr(z) =p(z).

Schritt 2: Zerlegung des Nenners. Bestimme die Nullstellen des Nennersq(z) und
zerlegeqso weit wie m ̈oglich in Faktoren.

- F ̈ur eine komplexe PBZ in Linearfaktoren
- F ̈ur eine reelle PBZ in Linearfaktoren und quadratische Faktoren ohne reelle Null-
    stellen.
Anschließend werden gleiche Faktoren zu Potenzen zusammengefasst.

Schritt 3: Ansatz zur Partialbruchzerlegung.

- Der Ansatz bestimmt sich allein aus den Faktoren des Nenners.
- Die gesamte Anzahl der Koeffizienten in den Ansatztermen stimmt mit dem Grad
    des Nenners ̈uberein.

```
Faktor des Nenners Ansatzterm
```
```
z−z 0
```
##### A

```
z−z 0
```
```
(z−z 0 )m
```
##### A 1

```
z−z 0
```
##### +

##### A 2

```
(z−z 0 )^2
```
##### +...+

```
Ak
(z−z 0 )m
```
```
(z−a)^2 +b^2
```
```
Cz+D
(z−a)^2 +b^2
```
```
((z−a)^2 +b^2 )m
C 1 z+D 1
(z−a)^2 +b^2
```
##### +

```
C 2 z+D 2
((z−a)^2 +b^2 )^2
```
##### +...+

```
Cmz+Dm
((z−a)^2 +b^2 )m
```
Schritt 4: Koeffizienten bestimmen. Mehrere M ̈oglichkeiten:

- Koeffizientenvergleich (immer m ̈oglich)
- Zuhaltemethode: Gibt die Koeffizienten von einfachen Polstellen bzw. den Koeffi-
    zient bei der h ̈ochsten Potenz eines Pols.
    Bestimme verbleibende Koeffizienten durch Koeffizientenvergleich oder durch Ein-
    setzen weiterer Zahlen.



Vorlesung 10

## 10 Vektorr ̈aume

Wir lernen Vektorr ̈aume kennen, die Grundstruktur der linearen Algebra.

### 10.1 Vektorr ̈aume

Vektorr ̈aume sind Mengen, in denen wir addieren und mit Skalaren multiplizieren k ̈on-
nen, so dass die
” ̈
ublichen“ Rechenregeln gelten. Die genaue Definition geben wir in
Definition 10.1. Beispiele sind die EbeneR^2 und der euklidische RaumR^3 , in denen wir
Vektoren addieren k ̈onnen und mit einer (reellen) Zahl multiplizieren k ̈onnen. Man kann
Vektorr ̈aume ̈uber den reellen oder den komplexen Zahlen betrachten. Die Grundlegen-
den Eigenschaften sind die gleichen. Im Folgenden schreiben wirKf ̈urRoderC, um
nicht jede Definition und jeden Satz doppelt zu schreiben (einmal f ̈urRund einmal f ̈ur
C). Die Elemente vonKheißenSkalare.Ubliche Bezeichnungen sind ̈ λ(Lambda) undμ
(My).

Definition 10.1(Vektorraum).EinK-Vektorraumist eine MengeV mit einer Addition
+ und einer skalaren Multiplikation·, so dass

```
v+w∈V und λ·v∈V
```
f ̈ur allev,w∈V undλ∈Kgilt und folgende Rechenregeln f ̈ur allev,w,x∈V und
λ,μ∈Kgelten:
1) + ist assoziativ: (v+w) +x=v+ (w+x),
2) + ist kommutativ:v+w=w+v,
3) es gibt einenNullvektor 0 ∈V mit 0 +v=v,
4) zu jedem Vektorv∈V gibt es−v∈V mitv+ (−v) = 0,
5) es gilt: (λμ)·v=λ·(μ·v),
6) Distributivgesetz:λ·(v+w) =λ·v+λ·w
7) Distributivgesetz: (λ+μ)·v=λ·v+μ·v.
8) 1·v=v.
EinVektorist ein Element eines Vektorraums.


Insbesondere enth ̈alt jederK-Vektorraum den Nullvektor und ist somit nicht leer:
V 6 =∅. DasKbei”K-Vektorraum“ sagt, aus welchem Zahlbereich die Zahlen (=Skalare)
kommen, mit denen multipliziert wird.
Zur Vereinfachung schreiben wir oftλvanstattλ·v. Auch schreiben wir oft nur
Vektorraum anstattK-Vektorraum.
Die geometrische Anschauung zu den Vektorraumroperationen ist die aus demR^2 :

```
v
```
```
w
v+w
v
```
```
2 v
1
2 v
−v= (−1)v
```
##### 0

Beispiel 10.2. 1) Die EbeneR^2 mit der Vektoraddition und Skalarmultiplikation

```
[
x 1
x 2
```
##### ]

##### +

##### [

```
y 1
y 2
```
##### ]

##### =

##### [

```
x 1 +y 1
x 2 +y 2
```
##### ]

```
, λ
```
##### [

```
x 1
x 2
```
##### ]

##### =

##### [

```
λx 1
λx 2
```
##### ]

##### ,

```
ist einR-Vektorraum.
```
```
2) Der dreidimensionale RaumR^3 mit der Vektoraddition und Skalarmultiplikation
ist einR-Vektorraum.
```
```
3)VektorraumKn:Allgemeiner ist fur ̈ n∈N,n≥1, die Menge
```
```
Kn=
```
##### 

##### 

##### 

##### 

##### 

```
x 1
..
.
xn
```
##### 

##### 

##### ∣∣

##### ∣

##### ∣∣

##### ∣∣

```
x 1 ,...,xn∈K
```
##### 

##### 

##### 

```
einK-Vektorraum, wenn man die Vektoren (wie imR^2 ) eintragsweise addiert und
mit Skalaren multipliziert:



```
```
x 1
..
.
xn
```
##### 

##### 

##### +

##### 

##### 

##### 

```
y 1
..
.
yn
```
##### 

##### 

##### :=

##### 

##### 

##### 

```
x 1 +y 1
..
.
xn+yn
```
##### 

##### 

```
 und λ·
```
##### 

##### 

##### 

```
x 1
..
.
xn
```
##### 

##### 

##### =

##### 

##### 

##### 

```
λx 1
..
.
λxn
```
##### 

##### 

##### .

```
Der Nullvektor ist~0 =
```
##### [ 0

##### ..

##### . 0

##### ]

. Insbesondere sindRneinR-Vektorraum undCnein
C-Vektorraum.
Vektoren ausRnoderCnwerden oft (aber nicht immer) mit einem Pfeil daruber ̈
geschrieben:~x.

```
4)Vektorraum der Polynome:Die Menge der Polynome
```
```
V ={p(z) =a 0 +a 1 z+...+anzn|wobein∈Nunda 0 ,a 1 ,...,an∈C}
```

```
mit derublichen Addition und Skalarmultiplikation von Polynomen (Abschnitt 8.1) ̈
bildet einenC-Vektorraum. Dieser wird oft mitC[z] bezeichnet. Die Menge der
Polynome mit reellen Koeffizienten wird oft mitR[z] bezeichnet und bildet ebenfalls
einen Vektorraum ( ̈uber den reellen Zahlen, d.h. wenn wir nur mit reellen Zahlen
multiplizieren). Der Nullvektor ist das Nullpolynom:p 0 (z) = 0.
```
```
5) IstDeine Menge, so ist die Menge der Funktionenf:D→Rmit der punktweisen
Addition und Skalarmultiplikation
```
```
(f+g)(x):=f(x) +g(x), (λf)(x):=λ·f(x),
```
```
einR-Vektorraum. (Rechts von den Gleichheitszeichen haben wir dabei + und·
von reellen Zahlen.) Der Nullvektor ist die Nullfunktionf 0 :D→R,f 0 (x) = 0,
denn f ̈ur diese giltf 0 +f=f, denn (f 0 +f)(x) =f 0 (x) +f(x) = 0 +f(x) =f(x)
f ̈ur jedesx∈D.
```
```
6) Auch Funktionenf :D→V, wobeiV ein beliebigerK-Vektorraum ist, bilden
einenK-Vektorraum (wieder mit punktweiser Addition und Multiplikation).
```
### 10.2 Teilr ̈aume

Teilr ̈aume sind Teilmengen von Vektorr ̈aumen, die selbst wieder ein Vektorraum sind.

Definition 10.3(Teilraum). SeiV einK-Vektorraum. Eine TeilmengeT ⊆V ist ein
TeilraumvonV, fallsTselbst einK-Vektorraum ist (mit dem gleichen + und·wieV).

Statt Teilraum sagt man auchUnterraumoderUntervektorraum. Es gilt die folgende
nutzliche Charakterisierung. ̈

Satz 10.4(Teilraumkriterium).SeiV einK-Vektorraum. Dann istT⊆V ein Teilraum
vonV, genau dann wenn
1) 0 ∈T,
2) f ̈ur allev,w∈T istv+w∈T,
3) f ̈ur allev∈T undλ∈Kistλv∈T.

Beweis. IstTein Teilraum vonV, so istTselbst ein Vektorraum und die drei Eigenschaften sind erf ̈ullt.
SeiT⊆Vso dass1)–3)erf ̈ullt sind. Dann”bleiben + und·inT“, wie es f ̈ur einenK-Vektorraum
sein muss. Wir m ̈ussen also noch die Rechenregeln in einem Vektorraum nachweisen. Es ist 0∈T, und
dann gilt 0+v=vf ̈ur allev∈T⊆V, weil das sogar f ̈ur allev∈Vgilt. Istv∈T, so ist (−1)·v=−v∈T,
und dannv−v= 0. Die anderen Rechenregeln ̈ubertragen sich vonVauf die TeilmengeT.

Der Satz sagt uns im Wesentlichen, dass eine Teilmenge ein Teilraum ist, wenn wir
bei + und·in der Teilmenge bleiben.

Beispiel 10.5. 1) IstV einK-Vektorraum, so sind{ 0 }undV Teilr ̈aume vonV.

```
2) SeiV =R^2. Dann ist die GeradeG={t[^12 ]|t∈R}ein Teilraum vonR^2 , denn:
(a) Mitt= 0 ist 0 [^12 ] = [^00 ]∈G,
```

```
(b) Sind~v, ~w ∈ G, so sind~v = s[^12 ] und ~w = t[^12 ] mits,t ∈ R, und dann
~v+~w= (s+t) [^12 ]∈G,
(c) Ist~v=t[^12 ]∈Gundλ∈R, so istλ~v= (λt) [^12 ].
Genauso sieht man: jede Gerade durch~0 ist ein Teilraum vonR^2.
```
```
3) SeiV=R^2 undKder Einheitskreis in der Ebene. Dieser enth ̈alt nicht die~ 0 ∈R^2 ,
ist also kein Teilraum.
```
```
4) InV =R^3 sind Geraden und Ebenen durch~0 Teilr ̈aume vonR^3.
```
```
5)Vektorraum der Polynome von beschr ̈anktem Grad:SeiV=C[z] derC-Vektorraum
der Polynome. Die MengeT :=C[z]≤ 2 :={p∈C[z]|deg(p)≤ 2 }der Polynome
vom Grad h ̈ochstens 2 ist ein Teilraum vonV, denn:
(a) das Nullpolynom liegt inT, denn deg(0) =−∞≤2,
(b) sindp,q∈T, d.h. habenpundqden Grad h ̈ochstens 2, so ist auch der Grad
vonp+qh ̈ochstens 2,
(c) istp∈T undλ∈C, so ist auch der Grad vonλ·ph ̈ochstens 2.
Genauso rechnet man f ̈ur jedesn∈Nnach:
```
```
C[z]≤n={p∈C[z]|deg(p)≤n},
```
```
die Menge der Polynome vom Grad h ̈ochstensn, ist ein Teilraum vonV =C[z].
Dies gilt ebenso f ̈ur Polynome mit reellen Koeffizienten:R[z]≤n = {p ∈ R[z] |
deg(p)≤n}ist ein Teilraum vonR[z].
```
### 10.3 Linearkombinationen

Idee: Linearkombination von zwei Vektoren:

v (^1) λ 1 v 1
v 2
λ 2 v 2 λ 1 v 1 +λ 2 v 2
Definition 10.6(Linearkombination).SeiVeinK-Vektorraum. Ein Vektorv∈Vheißt
Linearkombinationder Vektorenv 1 ,...,vk∈V, wenn Zahlenλ 1 ,...,λk∈Kexistieren,
so dass
v=λ 1 v 1 +λ 2 v 2 +...+λkvk=
∑k
j=1
λjvj.
Man sagt,vl ̈asst sich ausv 1 ,...,vklinear kombinieren. Dieλ 1 ,...,λkheißenKoeffizi-
entender Linearkombination.


Beispiel 10.7. 1) InV =R^2 ist
[
2
1

##### ]

##### = 2

##### [

##### 1

##### 0

##### ]

##### + 1

##### [

##### 0

##### 1

##### ]

##### ,

```
also ist~v= [^21 ] eine Linearkombination von~v 1 = [^10 ] und~v 2 = [^01 ]. Weiter ist
[
2
1
```
##### ]

##### = 1

##### [

##### 1

##### 0

##### ]

##### + 1

##### [

##### 1

##### 1

##### ]

##### .

```
2) Ein Polynomp(z) =a 0 +a 1 z+...+anznist eine Linearkombination der Monome
1 ,z,...,zn.
3) InV ={f:R→R}ist
```
```
sin
```
##### (

```
x+
π
3
```
##### )

```
= sin(x) cos
```
```
(π
3
```
##### )

```
+ cos(x) sin
```
```
(π
3
```
##### )

##### =

##### 1

##### 2

```
sin(x) +
```
##### √

##### 3

##### 2

```
cos(x),
```
```
d.h., die Funktionx7→sin(x+π 3 ) ist eine Linearkombination von sin und cos.
```
Definition 10.8(Span).SeiVeinK-Vektorraum. Die Menge aller Linearkombinationen
von v 1 ,...,vk ∈ V heißt der Span (oder dielineare H ̈ulle oder dasErzeugnis) von
v 1 ,...,vk:
span{v 1 ,...,vk}:={λ 1 v 1 +...+λkvk|λ 1 ,...,λk∈K}.

Beispiel 10.9. 1) SeiV =R^2 und~v 1 = [^12 ]∈V. Dann ist

```
span{~v 1 }={λ 1 ~v 1 |λ 1 ∈R},
und das ist die Gerade durch~0 mit dem Richtungsvektor~v 1.
2) SeiV =R^3 und~v 1 =
```
##### [ 1

```
0
0
```
##### ]

```
und~v 2 =
```
##### [ 0

```
1
0
```
##### ]

```
, dann ist
```
```
span{~v 1 ,~v 2 }={λ 1 ~v 1 +λ 2 ~v 2 |λ 1 ,λ 2 ∈R}=
```
##### 

##### 

##### 

##### 

##### 

```
λ 1
λ 2
0
```
##### 

##### 

##### ∣∣

##### ∣∣

##### ∣

##### ∣

```
λ 1 ,λ 2 ∈R
```
##### 

##### 

##### 

```
diex-y-Ebene imR^3.
```
Satz 10.10.SeiV einK-Vektorraum undv 1 ,...,vk∈V. Dann istspan{v 1 ,...,vk}ein
Teilraum vonV.

Beweis. Das rechnen wir mit dem Teilraumkriterium nach:
1) Mitλ 1 =...=λk= 0∈Kist 0V= 0v 1 +...+ 0vk∈span{v 1 ,...,vk}.
2) Sindv=λ 1 v 1 +...+λkvkundw=μ 1 v 1 +...+μkvk∈span{v 1 ,...,vk}, dann ist
v+w=λ 1 v 1 +...+λkvk+μ 1 v 1 +...+μkvk= (λ 1 +μ 1 )v 1 +...+ (λk+μk)vk,
alsov+w∈span{v 1 ,...,vk}.
3) Sindv=λ 1 v 1 +...+λkvk∈span{v 1 ,...,vk}undλ∈K, dann ist
λv=λ(λ 1 v 1 +...+λkvk) = (λλ 1 )v 1 +...+ (λλk)vk∈span{v 1 ,...,vk}.

Daher ist span{v 1 ,...,vk}ein Teilraum vonV.


### 10.4 Erzeugendensysteme

Definition 10.11(Erzeugendensystem).SeiV einK-Vektorraum undTein Teilraum
vonV. Eine Menge{v 1 ,...,vk} ⊆T heißtErzeugendensystem vonT, falls ihr Span
gleichTist:
span{v 1 ,...,vk}=T.

Beispiel 10.12. 1) SeiV =R^2 undG={t[^12 ]|t∈R}die Gerade durch [^00 ] und
[^12 ]. Anders geschrieben ist also

```
G= span
```
##### {[

##### 1

##### 2

##### ]}

##### ,

```
und damit ist{[^12 ]}ein Erzeugendensystem vonG. Weitere Erzeugendensysteme
vonGsind zum Beispiel
{[
2
4
```
##### ]}

##### ,

##### {[

##### 1

##### 2

##### ]

##### ,

##### [

##### 0

##### 0

##### ]}

##### ,

##### {[

##### 2

##### 4

##### ]

##### ,

##### [

##### − 3

##### − 6

##### ]}

##### ,

##### {[

##### 0

##### 0

##### ]

##### ,

##### [

##### 2

##### 4

##### ]

##### ,

##### [

##### − 1

##### − 2

##### ]}

##### .

```
Hingegen sind{[^12 ],[^11 ]}und{[^11 ]}keine Erzeugendensysteme vonG, da [^11 ]∈/G.
2) SeiT=V =R^2. Dann ist zum Beispiel{[^10 ],[^01 ]}ein Erzeugendensystem vonT.
Dass span{[^10 ],[^01 ]}⊆Tgilt ist klar, und wir m ̈ussen nachrechnen, dass sich jeder
Vektor inR^2 als Linearkombination von [^10 ],[^01 ] darstellen l ̈asst. Das ist aber ganz
einfach:
~x=
```
##### [

```
x 1
x 2
```
##### ]

##### =

##### [

```
x 1
0
```
##### ]

##### +

##### [

##### 0

```
x 2
```
##### ]

```
=x 1
```
##### [

##### 1

##### 0

##### ]

```
+x 2
```
##### [

##### 0

##### 1

##### ]

##### .

```
3) Genauso gilt, dass 






```
##### 

##### 

##### 

##### 

##### 

##### 1

##### 0

##### 0

##### ..

##### .

##### 0

##### 

##### 

##### 

##### 

##### 

##### ,

##### 

##### 

##### 

##### 

##### 

##### 0

##### 1

##### 0

##### ..

##### .

##### 0

##### 

##### 

##### 

##### 

##### 

##### ,...,

##### 

##### 

##### 

##### 

##### 

##### 0

##### 0

##### ..

##### .

##### 0

##### 1

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

```
ein Erzeugendensystem vonKnist.
4) SeiV =C[z] der Vektorraum der Polynome undT = C[z]≤ 2 der Teilraum der
Polynome vom Grad h ̈ochstens 2. Fur ̈ p(z)∈T ist
```
```
p(z) =a 0 +a 1 z+a 2 z^2 =a 0 ·1 +a 1 ·z+a 2 ·z^2 ,
```
```
also ist{ 1 ,z,z^2 }ein Erzeugendensystem vonT. Hingegen ist{ 1 ,z}kein Erzeugen-
densystem vonT, da zum Beispiel 1 +z^2 ∈Tkeine Linearkombination von 1 und
zist.
Allgemeiner ist
```
```
C[z]≤n={p|pist ein Polynom mit deg(p)≤n}= span{ 1 ,z,...,zn},
```

```
d.h. 1,z,...,znbilden ein Erzeugendensystem der Polynome vom Grad h ̈ochstens
n. F ̈ur reelle Polynome gilt das ebenso.
Wir beobachten:C[z] (undR[z]) haben kein Erzeugendensystem aus endlich vielen
Elementen, da der Grad der Polynome beliebig groß werden kann.
5) Sei wiederT=V =R^2. Dann ist{[^10 ],[^01 ],[^11 ]}ein Erzeugendensystem vonR^2 ,
denn es ist zum Beispiel
```
```
~x=
```
##### [

```
x 1
x 2
```
##### ]

```
=x 1
```
##### [

##### 1

##### 0

##### ]

```
+x 2
```
##### [

##### 0

##### 1

##### ]

##### + 0

##### [

##### 1

##### 1

##### ]

##### .

```
Den Vektor~xk ̈onnen wir aber auch anders darstellen, zum Beispiel als
```
```
~x=
```
##### [

```
x 1
x 2
```
##### ]

```
= (x 1 −x 2 )
```
##### [

##### 1

##### 0

##### ]

##### + 0

##### [

##### 0

##### 1

##### ]

```
+x 2
```
##### [

##### 1

##### 1

##### ]

##### .

Ein Erzeugendensystem erlaubt es mit wenigen Grundvektoren (v 1 ,...,vk) s ̈amtliche
Vektoren eines Vektorraums durch Linearkombinationen zu rekonstruieren. Das ist sehr
nutzlich, insbesondere da die Koeffizienten Zahlen sind, mit denen sich bestens Rechnen ̈
l ̈asst, auch und insbesondere auf dem Computer.
Diese Rekonstruktion ist im Allgemeinen nicht eindeutig, wie das obige Beispiel zeigt.
Um Eindeutigkeit der Koeffizienten zu erhalten (und ein minimales Erzeugendensystem
zu erhalten), brauchen wir den Begriff der linearen Unabh ̈angigkeit, den wir in Vorle-
sung 11 kennen lernen werden.



Vorlesung 11

## 11 Basis und Dimension

In dieser Vorlesung lernen wir die Begriffe Basis und Dimension eines Vektorraums ken-
nen. Eine Basis ist ein Erzeugendensystem, bzgl. dem sich Vektoren eindeutig darstellen
lassen. Die Dimension ist ein Maß fur die Gr ̈ ̈oße eines Vektorraums.

### 11.1 Lineare Unabh ̈angigkeit

Ob sich Vektoren eindeutig durch ein Erzeugendensystem darstellen lassen, h ̈angt davon
ab, ob die Vektoren linear unabh ̈angig sind.

Definition 11.1(linear unabh ̈angig, linear abh ̈angig).

```
1) Die Vektorenv 1 ,...,vk desK-VektorraumsV heißen linear unabh ̈angig genau
dann, wenn die Gleichung
```
```
λ 1 v 1 +λ 2 v 2 +...+λkvk= 0
```
```
f ̈ur die Unbekanntenλ 1 ,λ 2 ,...,λk ∈Knur die L ̈osungλ 1 =λ 2 =...=λk= 0
hat. (D.h. die L ̈osungλ 1 =λ 2 =...=λk= 0 ist eindeutig.)
```
```
2) Die Vektorenv 1 ,...,vkheißenlinear abh ̈angig, wenn sie nicht linear unabh ̈angig
sind. D.h. sie sind linear abh ̈angig genau dann, wenn die Gleichung
```
```
λ 1 v 1 +λ 2 v 2 +...+λkvk= 0
```
```
neben der L ̈osungλ 1 =...=λk= 0 noch weitere L ̈osungen besitzt (also wenn die
L ̈osung nicht eindeutig ist).
```
Beispiel 11.2. 1) Die Vektoren~v 1 = [^10 ] und~v 2 = [^01 ] sind linear unabh ̈angig inK^2 ,
denn sindλ 1 ,λ 2 ∈Kmit
[
0
0

##### ]

```
=λ 1 ~v 1 +λ 2 ~v 2 =
```
##### [

```
λ 1
λ 2
```
##### ]

##### ,

```
so folgtλ 1 =λ 2 = 0. Auch [^10 ] und [^11 ] sind linear unabh ̈angig.
```

```
2) Die Vektoren~v 1 = [^10 ],~v 2 = [^01 ],~v 3 = [^11 ] sind linear abh ̈angig, da z.B.
[
0
0
```
##### ]

##### = 1

##### [

##### 1

##### 0

##### ]

##### + 1

##### [

##### 0

##### 1

##### ]

##### − 1

##### [

##### 1

##### 1

##### ]

##### ,

```
wobei die Koeffizienten nicht alle Null sind. In Beispiel 10.12 hatten wir gesehen,
dass~v 1 ,~v 2 ,~v 3 ein Erzeugendensystem bilden, bei dem die Darstellung aber nicht
eindeutig ist.
3) Die Vektoren [^10 ] und [^20 ] sind nicht linear unabh ̈angig, denn: Sindλ 1 ,λ 2 ∈Kmit
[
0
0
```
##### ]

```
=λ 1
```
##### [

##### 1

##### 0

##### ]

```
+λ 2
```
##### [

##### 2

##### 0

##### ]

##### =

##### [

```
λ 1 + 2λ 2
0
```
##### ]

##### ,

```
so folgtλ 1 + 2λ 2 = 0, was zum Beispiel auch f ̈urλ 2 = 1 undλ 1 =−2 erf ̈ullt ist.
Ebenso sind [^11 ] und
```
##### [− 1

```
− 1
```
##### ]

```
linear abh ̈angig.
4)Der Nullvektor ist immer linear abh ̈angig:Sindv 1 ,...,vk∈V mit einemvj= 0,
so kann man∑ λj = 1 6 = 0 und alle anderen Koefiizienten = 0 w ̈ahlen, so dass
k
i=1λivi= 0 gilt. Daher sindv^1 ,...,vklinear abh ̈angig.
5) Sei V = C[z] der Vektorraum der Polynome. F ̈ur n ∈ Nsind die Polynome
1 ,z,z^2 ,...,znlinear unabh ̈angig: Sindλ 0 ,λ 1 ,...,λn∈Cmit
λ 0 1 +λ 1 z+...+λnzn= 0 (Nullpolynom),
so sind nach dem Koeffizientenvergleich (Satz 8.12) alle Koeffizientenλ 0 =λ 1 =
...=λn= 0. Daher sind 1,z,...,znlinear unabh ̈angig.
Anschaulich gesehen sind Vektoren linear unabh ̈angig, wenn jeder in eine neue eigene
```
”
Richtung“ zeigt. ImR^2 undR^3 kann man sich das durch eine Skizze veranschaulichen.
ImR^2 :

```
linear unabh ̈angig
v 1
v 2
```
```
linear abh ̈angig
v 1
v 2
```
```
linear abh ̈angig
v 1
```
```
v 2
```
```
linear abh ̈angig
v 1
v 2
```
```
v 3
```
ImR^3 :

```
linear unabh ̈angig
```
```
v 1
```
```
v 2
```
```
v 3
```
```
linear abh ̈angig
```
```
v 1
```
```
v 2
v 3
```

Die Vektorenv 1 ,...,vksind linear abh ̈angig, wenn es einen gibt, der sich als Linearkom-
bination der anderen schreiben l ̈asst. Daher sindv 1 ,...,vk linear unabh ̈angig, wenn es
keinen Vektor gibt, der Linearkombination der anderen Vektoren ist. Das ist gut fur die ̈
Vorstellung, aber schlecht zum Nachrechnen. Leichter ist es mit der Definition.
Um nachzurechnen, obv 1 ,...,vklinear unabh ̈angig sind oder nicht, nehmen wir die
Unbekanntenλ 1 ,...,λk∈K, stellen die Gleichung

```
0 =λ 1 v 1 +λ 2 v 2 +...+λkvk
```
auf und machen daraus aus lineares Gleichungssystem f ̈urλ 1 ,...,λk:

- Ist dieses eindeutig l ̈osbar (alle λ 1 = ... = λk = 0), so sindv 1 ,...,vk linear
    unabh ̈angig.
- Ist die L ̈osung des linearen Gleichungssystems nicht eindeutig, so sindv 1 ,...,vk
    linear abh ̈angig.

### 11.2 Basis und Dimension

Wir hatten gesehen, dass bei einem Erzeugendensystem die Darstellung von Vektoren im
Allgemeinen nicht eindeutig sein braucht (siehe Beispiel 10.12). Verlangt man zus ̈atzlich
lineare Unabh ̈angigkeit, so f ̈uhrt uns das auf den Begriff der Basis, f ̈ur die die Darstellung
von Vektoren eindeutig wird.

Definition 11.3 (Basis). Sei V 6 = { 0 } einK-Vektorraum. Ein endliches linear un-
abh ̈angiges Erzeugendensystem{v 1 ,...,vn}vonV heißtBasisvonV.
Ausfuhrlich bedeutet das: ̈ {v 1 ,...,vn}heißtBasisvonV, falls gilt

```
1)v 1 ,...,vnsind linear unabh ̈angig
```
```
2){v 1 ,...,vn}ist ein Erzeugendensystem: span{v 1 ,...,vn}=V.
```
F ̈ur den NullvektorraumV ={ 0 }definiert man die leere Menge als Basis.

Bemerkung 11.4. 1) Singular: die Basis, Plural: die Basen. Die Base (ohne”n“)
kommt in der Chemie vor, und ist eine veraltete Bezeichnung f ̈ur Cousine.

```
2)Basen sind immer geordnet.Auch wenn Basen wie Mengen geschrieben werden
(Mengen sind ungeordnet), sind Basen immer geordnet, d.h. inB={v 1 ,...,vn}
istv 1 der erste Basisvektor,v 2 der zweite Basisvektor,... Das wird oft wichtig sein.
```
Beispiel 11.5. 1) Die Vektoren~v 1 = [^10 ] und~v 2 = [^11 ] bilden eine Basis vonR^2. Die
Vektoren~v 1 ,~v 2 sind linear unabh ̈angig, denn: Sindλ 1 ,λ 2 ∈Rmit
[
0
0

##### ]

```
=λ 1
```
##### [

##### 1

##### 0

##### ]

```
+λ 2
```
##### [

##### 1

##### 1

##### ]

##### =

##### [

```
λ 1 +λ 2
λ 2
```
##### ]

##### ,


```
so folgt im zweiten Eintragλ 2 = 0, und dann im ersten Eintrag 0 =λ 1 +λ 2 =λ 1.
Außerdem bilden~v 1 ,~v 2 ein Erzeugendensystem, denn f ̈ur einen beliebigen Vektor
inR^2 gilt
[
x 1
x 2
```
##### ]

##### =

##### [

```
x 1 −x 2 +x 2
x 2
```
##### ]

```
= (x 1 −x 2 )
```
##### [

##### 1

##### 0

##### ]

```
+x 2
```
##### [

##### 1

##### 1

##### ]

```
= (x 1 −x 2 )~v 1 +x 2 ~v 2.
```
```
Daher ist{~v 1 ,~v 2 }eine Basis vonR^2.
```
```
2) Die Vektoren~v 1 = [^10 ] und~v 2 = [^01 ] sind linear unabh ̈angig (Beispiel 11.2) und ein
Erzeugendensystem (Beispiel 10.12), daher ist{[^10 ],[^01 ]}eine Basis vonR^2.
```
```
3) Allgemein ist 






```
##### 

##### 

##### 

##### 

##### 

##### 1

##### 0

##### 0

##### ..

##### .

##### 0

##### 

##### 

##### 

##### 

##### 

##### ,

##### 

##### 

##### 

##### 

##### 

##### 0

##### 1

##### 0

##### ..

##### .

##### 0

##### 

##### 

##### 

##### 

##### 

##### ,...,

##### 

##### 

##### 

##### 

##### 

##### 0

##### 0

##### ..

##### .

##### 0

##### 1

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

```
eine Basis vonKn, die so genannteStandardbasis.
4) Hingegen ist{[^10 ],[^01 ],[^11 ]}keine Basis: Dies ist zwar ein Erzeugendensystem (Bei-
spiel 10.12), aber die Vektoren sind linear abh ̈angig (Beispiel 11.2).
5) Der Vektorraum der Polynome vom Grad h ̈ochstensn,
```
```
C[z]≤n= span{ 1 ,z,...zn}={a 0 +a 1 z+...+anzn|a 0 ,a 1 ,...,an∈C},
```
```
hat die Basis{ 1 ,z,...,zn}, denn 1,z,...,znsind linear unabh ̈angig (Beispiel 11.2)
und{ 1 ,z,...,zn}ist ein Erzeugendensystem (Beispiel 10.12).
```
Satz 11.6. Alle Basen eines Vektorraums haben die gleiche Anzahl an Elementen.

```
Das ist nicht so schwer zu beweisen und erm ̈oglicht die folgende Definition.
```
Definition 11.7 (Dimension). DieDimension eines Vektorraums ist die Anzahl der
Elemente einer Basis vonV und wird mit dim(V) bezeichnet. HatV eine endliche Basis,
so ist dim(V)∈N, und wir nennenV endlichdimensional. HatV keine endliche Basis,
so schreiben wir dim(V) =∞und nennenV unendlichdimensinal.

Beispiel 11.8.Mit den Basen aus Beispiel 11.5 k ̈onnen wir die Dimension einiger Vek-
torr ̈aume bestimmen.
1) Es ist dim(R^2 ) = 2, denn{[^10 ],[^01 ]}ist eine Basis vonR^2.
2) Allgemein ist dim(Kn) =n, da die Standardbasis vonKngenaunElemente hat.
3) Der Vektorraum der Polynome vom Grad h ̈ochstensn,

```
C[z]≤n= span{ 1 ,z,...zn}={a 0 +a 1 z+...+anzn|a 0 ,a 1 ,...,an∈C},
```
```
hat die Dimension dim(C[z]≤n) =n+ 1, da{ 1 ,z,...,zn}eine Basis ist. Gleiches
gilt f ̈ur reelle Polynome.
```

```
4) Der Vektorraum der Polynome
```
```
C[z] ={p(z) =a 0 +a 1 z+...+anzn|wobein∈Nunda 0 ,a 1 ,...,an∈C}
```
```
ist unendlichdimensional: dim(C[z]) =∞. Um jedes m ̈ogliche Polynom darzustel-
len, braucht man alle Monome 1,z,z^2 ,.... Gleiches gilt f ̈ur den Vektorraum der
Polynome mit reellen KoeffizientenR[z].
```
Kennt man die Dimension eines Vektorraums und hat die passende Anzahl Vektoren
f ̈ur eine Basis, braucht man nur noch lineare Unabh ̈angigkeit oder Erzeugendensystem
zu prufen, also nur noch eine der beiden Bedingungen f ̈ ̈ur eine Basis.

Satz 11.9. SeiV ein Vektorraum mit Dimensionn∈N,n≥ 1. Dann gilt:

```
1) Sindv 1 ,...,vn∈V linear unabh ̈angig, so ist{v 1 ,...,vn}eine Basis vonV.
```
```
2) Ist{v 1 ,...,vn}ein Erzeugendensystem vonV, so ist{v 1 ,...,vn}eine Basis von
V.
```
Konstruktion von Basen WirdV von endlich vielen Vektoren aufgespannt,V =
span{v 1 ,...,vk}, so kann man aus diesen eine Basis vonV konstruieren. Sindv 1 ,...,vk
linear unabh ̈angig, so bilden sie eine Basis und wir sind fertig. Andernfalls sindv 1 ,...,vk
also linear abh ̈angig und wir k ̈onnen schreiben

```
0 =λ 1 v 1 +λ 2 v 2 +...+λkvk
```
mitλ 1 ,...,λk∈K, die nicht alle gleich Null sind. Ist zum Beispielλk 6 = 0, k ̈onnen wir
nachvkaufl ̈osen:

```
vk=−
```
##### 1

```
λk
```
```
(λ 1 v 1 +...+λk− 1 vk− 1 ),
```
d.h. wir k ̈onnenvkdurch die anderen Vektorenv 1 ,...,vk− 1 schreiben, und es ist

```
V = span{v 1 ,...,vk}= span{v 1 ,...,vk− 1 }.
```
Sind jetztv 1 ,...,vk− 1 linear unabh ̈angig, haben wir eine Basis gefunden. Andernfalls
entfernen wir wie eben einen n ̈achsten Vektor, und f ̈uhren dies so lange fort, bis wir ein
linear unabh ̈angiges Erzeugendensystem, also eine Basis, haben.
Also:

```
1) Ist{v 1 ,...,vk} ein Erzeugendensystem aber sind die Vektoren nicht linear un-
abh ̈angig,so entferne so lange geeignete Vektoren, bis eine Basisubrig bleibt. ̈
```
```
2) Sindv 1 ,...,vk linear unabh ̈angig, aber{v 1 ,...,vk}kein Erzeugendensystem, so
nimm geeignete Vektorenvk+1 ∈/ span{v 1 ,...,vk}hinzu, bis eine Basis vonV
entsteht (falls dim(V)<∞).
```

### 11.3 Koordinaten

Hat man eine Basis, l ̈asst sich jeder Vektor auf eindeutige Art und Weise als Linearkom-
bination der Basisvektoren darstellen. Das wird sich als sehr n ̈utzlich erweisen, da man
dann jeden Vektorv∈V mit einem Spaltenvektor ausKnidentifizieren kann, mit dem
man bestens rechnen kann.

Satz 11.10.SeiV einK-Vektorraum mit BasisB={v 1 ,...,vn}. Dann l ̈asst sich jeder
Vektorv∈V schreiben als

```
v=
```
```
∑n
```
```
j=1
```
```
λjvj=λ 1 v 1 +λ 2 v 2 +...+λnvn,
```
wobei die Koeffizientenλ 1 ,...,λn∈Keindeutig bestimmtsind undKoordinatenvonv
heißen. Der Vektor

```
~vB=
```
##### 

##### 

##### 

```
λ 1
..
.
λn
```
##### 

##### 

```
∈Kn
```
heißt derKoordinatenvektorvonvbzgl.B.

Beweis. Als Basis ist{v 1 ,...,vn}insbesondere ein Erzeugendensystem vonV, so dass
sichvals Linearkombination vonv 1 ,...,vnschreiben l ̈asst:

```
v=λ 1 v 1 +...+λnvn
```
mit Koeffizientenλ 1 ,...,λn ∈ K. Wir zeigen jetzt, dass die Koeffizienten eindeutig
bestimmt sind. Dazu nehmen wir an, wir k ̈onnten den Vektorvauch schreiben als

```
v=μ 1 v 1 +...+μnvn,
```
mit Koeffizientenμ 1 ,...,μn, die m ̈oglicherweise von den λ 1 ,...,λn verschieden sind.
Wir m ̈ochten nun sehen, dass sie gleich sind. Dazu rechnen wir

0 =v−v=λ 1 v 1 +...+λnvn−(μ 1 v 1 +...+μnvn) = (λ 1 −μ 1 )v 1 +...+ (λn−μn)vn.

Dav 1 ,...,vnlinear unabh ̈angig sind, sind die Koeffizienten dieser Linearkombination
alle Null, d.h. es sindλ 1 −μ 1 = 0,... ,λn−μn= 0, anders gesagtλ 1 =μ 1 ,... ,λn=μn.
Damit sind die Koeffizienten vonveindeutig bestimmt.

Die Koordinaten sind eine”Bauanleitung“, wie man einen Vektor aus den Basis-
vektoren (den
”
Bausteinen“) zusammensetzt (genauer: linear kombiniert). Hat man die

Koordinaten (die”Bauanleitung“), so ist es ganz einfach: Ist~vB=

```
[λ
1
..
.
λn
```
##### ]

```
, dann ist
```
```
v=λ 1 v 1 +λ 2 v 2 +...+λnvn=
```
```
∑n
```
```
j=1
```
```
λjvj,
```

was wir direkt ausrechnen k ̈onnen. Hat z.B.vdie Koordinaten 2 und 3 bez ̈uglich~v 1 = [^11 ]
und~v 2 =

##### [ 1

```
− 1
```
##### ]

```
, so brauchen wir 2-mal den Vektor [^11 ] und 3-mal den Vektor
```
##### [ 1

```
− 1
```
##### ]

um
den Vektor~vzu schreiben:

```
~v= 2
```
##### [

##### 1

##### 1

##### ]

##### + 3

##### [

##### 1

##### − 1

##### ]

##### =

##### [

##### 5

##### − 1

##### ]

##### .

Haben wir hingegen den Vektorvund m ̈ochten seine Koordinaten finden, ist in der Regel
ein Gleichungssystem zu l ̈osen.
Ist schonV =Kn, so sieht der Koordinatenvektor nicht spektakul ̈ar anders aus, wird
aber dennoch n ̈utzlich sein. Ist hingegenVirgendein anderer Vektorraum, so erlaubt uns
der letzte Satz den allgemeinen Vektorvmit einem Spaltenvektor inKnzu identifizieren.

Beispiel 11.11. 1) SeiV=R^2 mit der StandardbasisB={[^10 ],[^01 ]}. F ̈ur~v= [xx^12 ]∈
R^2 ist
~v=

##### [

```
x 1
x 2
```
##### ]

```
=x 1
```
##### [

##### 1

##### 0

##### ]

```
+x 2
```
##### [

##### 0

##### 1

##### ]

##### .

```
Damit ist der Koordinatenvektor~vB= [xx^12 ] und sieht wie~vselbst aus.
Allgemein gilt~vB=~vfallsBdie Standardbasis vonKnist.
2) SeiV =R^2 mit der BasisB 2 ={[^10 ],[^11 ]}. F ̈ur~v= [xx^12 ]∈R^2 rechnen wir
```
```
~v= [xx^12 ] = (x 1 −x 2 ) [^10 ] +x 2 [^11 ],
```
```
und erhalten den Koordinatenvektor von~vbez ̈uglich der BasisB 2 :
```
```
~vB 2 =
```
##### [

```
x 1 −x 2
x 2
```
##### ]

##### .

```
Zum Beispiel ist fur ̈ ~v= [^35 ] =−2 [^10 ] + 5 [^11 ], also~vB 2 =
```
##### [− 2

```
5
```
##### ]

##### .

```
Der Koordinatenvektor~vB 2 bzgl.B 2 ist verschieden von dem Koordinatenvektor
bez ̈uglich der Standardbasis. Wie Koordinatenvektoren bzgl. verschiedener Basen
zusammenh ̈angen, werden wir in Vorlesung 16 sehen.
3) Der Vektorraum der Polynome vom Grad h ̈ochstens 2,
```
```
V =C[z]≤ 2 ={p(z) =a 0 +a 1 z+a 2 z^2 |a 0 ,a 1 ,a 2 ∈C},
```
```
hat die BasisB={ 1 ,z,z^2 }(Beispiel 11.5). Bez ̈uglich der BasisBhat ein Polynom
```
```
p(z) =a 0 +a 1 z+a 2 z^2 =a 0 ·1 +a 1 ·z+a 2 ·z^2
```
```
dem Koordinatenvektor
```
```
~pB=
```
##### 

##### 

```
a 0
a 1
a 2
```
##### 

##### .


4) SeiV ={f: [0,1]→R}derR-Vektorraum der Funktionen von [0,1] nachR. Sein∈Nund
h=n^1 +1. Definiere die Hutfunktionenfj∈Vdurch

```
fj(x):=
```
```



```
```
1
h(x−(j−1)h), (j−1)h≤x≤jh,
1 −^1 h(x−jh), jh≤x≤(j+ 1)h,
0 , sonst.
Diese sehen wie folgt aus:
1
```
```
(j−1)h jh (j+ 1)h
Wir rechnen nach, dassf 0 ,...,fn+1linear unabh ̈angig sind: Seienλ 0 ,...,λn+1∈Rmit
λ 0 f 0 (x) +λ 1 f 1 (x) +...+λn+1fn+1(x) = 0
f ̈ur allex∈[0,1]. Im Punktx=jhistfjeins und die anderen Funktionenfksind null (k 6 =j).
Setzen wir daher den Punktx=jhein, so erhalten wir
λj·1 = 0,
alsoλj= 0. Da wirjhf ̈urj= 0, 1 ,...,n+ 1 einsetzen k ̈onnen, sind alle Koeffizientenλj= 0,
und damitf 0 ,f 1 ,...,fn+1linear unabh ̈angig.
Insbesondere istB={f 0 ,...,fn+1}eine Basis vonT= span{f 0 ,...,fn+1}, so dass dim(T) =
n+2. Jede Funktion ausTl ̈asst sich dann durch Angabe vonn+2 Zahlen durch die Hutfunktionen
rekonstruieren.
Zum Beispiel l ̈asst sich (mitn= 5) die Funktion
```
(^000). 2 0. 4 0. 6 0. 8 1
0. 2
0. 4
0. 6
0. 8
1
durch Angabe ihres Koordinatenvektors komplett beschreiben. Dieser ist
f~B=






0. 8
0. 95
0. 1
0. 6
0. 5
0. 05
0. 35






∈R^7.
Die Hutfunktionen werden bei der Finite-Elemente-Methode beim L ̈osen gewisser partieller Dif-
ferentialgleichungen verwendet. Dies ist Thema der Vorlesung”Numerik II f ̈ur Ingenieurwissen-
schaften“.


Vorlesung 12

## 12 Matrizen

Nach der Bereitstellung der Grundbegriffe in den ersten Vorlesungen, ist unser erstes
großes Ziel das L ̈osen linearer Gleichungssysteme, also von mehreren linearen Gleichun-
gen in mehreren Unbekannten, zum Beispiel

```
x 1 + 3x 2 = 1,
2 x 1 + 4x 2 = 2.
```
Um lineare Gleichungssysteme effizient (von Hand oder im Rechner) zu l ̈osen, ben ̈otigen
wir den Begriff der Matrix, den wir in dieser Vorlesung einfuhren. Die L ̈ ̈osung linearer
Gleichungssysteme ist dann Thema der n ̈achsten Vorlesungen.

### 12.1 Definition von Matrizen

In dieser und den n ̈achsten Vorlesungen ist unerheblich, ob wir mit reellen oder komple-
xen Zahlen rechnen. Daher schreiben wirKf ̈urRoderC.

Definition 12.1(Matrix). F ̈ur Zahlenai,j∈K,i= 1,...,m,j= 1,...,n, heißt das
Zahlenschema

##### A=

##### 

##### 

##### 

##### 

##### 

```
a 1 , 1 a 1 , 2 ... a 1 ,n
a 2 , 1 a 2 , 2 ... a 2 ,n
..
.
```
##### ..

##### .

##### ... ..

##### .

```
am, 1 am, 2 ... am,n
```
##### 

##### 

##### 

##### 

##### 

##### =

##### [

```
ai,j
```
##### ]

##### =

##### [

```
ai,j
```
##### ]

```
i=1,...,m
j=1,...,n
```
einem×n-Matrixmit Eintr ̈agen inK. Die Menge allerm×n-Matrizen mit Eintr ̈agen
inKwird mitKm,nbezeichnet. Eine Matrix heißtquadratisch, fallsm=ngilt.

Bemerkung 12.2. 1) Die Indizesi,jsetzt man nach der Regel
”
Zeile, Spalte“.
2) Singular: die Matrix, Plural: die Matrizen. Eine”Matrize“ bezeichnet u.a. eine
Gußform, Druckvorlage, oder auch ein Hilfsmittel beim Legen einer Zahnf ̈ullung.
3) Man sagt auch kurzer ̈ ”m×n-Matrix ̈uberK“, lies”mkreuznMatrix ̈uberK“.
4) Man schreibt auchaijstattai,j, wenn keine Verwechslungsgefahr besteht.


Beispiel 12.3.Es ist

```
A=
```
##### [

##### 1 2 3

##### 4 5 6

##### ]

eine 2×3-Matrix mit Eintr ̈agen inR(oderC). Der Eintrag (2,1) ista 2 , 1 = 4. Die Matrix

```
B=
```
##### [

```
1 +i 1 −i
2 i 42
```
##### ]

ist eine 2×2-Matrix ̈uberC.

```
Zwei besonders wichtige Matrizen sind dieNullmatrix, deren Eintr ̈age alle 0 sind:
```
```
0 = 0m,n=
```
##### [

##### 0

##### ]

##### =

##### 

##### 

##### 

##### 

##### 0 0 ... 0

##### 0 0 ... 0

##### ..

##### .

##### ..

##### .

##### ... ..

##### .

##### 0 0 ... 0

##### 

##### 

##### 

##### 

```
∈Km,n,
```
sowie dieEinheitsmatrix(auchIdentit ̈at) die quadratisch ist (m=n)

```
In:=
```
##### 

##### 

##### 

##### 

##### 

##### 1 0 ... 0

##### 0 1

##### ... ..

##### .

##### ..

##### .

##### ... ...

##### 0

##### 0 ... 0 1

##### 

##### 

##### 

##### 

##### 

```
∈Kn,n.
```
```
Istm= 1, also
A=
```
##### [

```
a 1 , 1 a 1 , 2 ... a 1 ,n
```
##### ]

```
∈K^1 ,n,
```
so nennt manAeinenZeilenvektor. Ist hingegenn= 1, also

##### A=

##### 

##### 

##### 

##### 

```
a 1 , 1
a 2 , 1
..
.
am, 1
```
##### 

##### 

##### 

##### 

```
∈Km,^1 ,
```
so nennt manAeinenSpaltenvektoroder kurzVektorinKm,^1 , und man schreibt k ̈urzer

```
Km:=Km,^1.
```
Beispiel 12.4.Zeilenvektoren:

```
A=
```
##### [

```
1 1 −i
```
##### ]

##### ∈C^1 ,^3 , B=

##### [

##### −1 1

##### ]

##### ∈R^1 ,^2.

Spaltenvektoren:

```
[
1
0
```
##### ]

##### ,

##### [

##### 0

##### 1

##### ]

##### ,

##### [

##### 2

##### − 5

##### ]

##### ∈R^2 ,

##### 

##### 

##### − 1

##### 6

##### 2

##### 

##### ∈R^3 ,

##### 

##### 

```
1 +i
√^4
2 i
```
##### 

##### ∈C^3.

Spaltenvekoren inR^2 sind genau die Punkte der Ebene. Vektoren inR^3 sind die Punkte
des dreidimensionalen Raums.


### 12.2 Addition und Skalarmultiplikation

Addition und Skalarmultiplikation von Vektoren sind Eintragsweise definiert, zum Bei-
spiel [
1
2

##### ]

##### +

##### [

##### − 2

##### 1

##### ]

##### =

##### [

##### − 1

##### 3

##### ]

##### , 2

##### [

##### 1

##### 3

##### ]

##### =

##### [

##### 2

##### 6

##### ]

##### .

F ̈ur Matrizen definieren wir das ganz genauso.

Definition 12.5 (Addition und Skalarmultiplikation von Matrizen).SeienA=

##### [

```
ai,j
```
##### ]

##### ,

##### B=

##### [

```
bi,j
```
##### ]

∈Km,nzweim×n-Matrizen undλ∈K. Dann ist dieSummevonAundB
die Matrix
A+B=

##### [

```
ai,j+bi,j
```
##### ]

```
∈Km,n
```
und dieMultiplikation mit einem Skalar(kurz:Skalarmultiplikation) ist die Matrix

```
λA=
```
##### [

```
λai,j
```
##### ]

∈Km,n.
Beachten Sie, dass nur Matrizen gleicher Gr ̈oße addiert werden k ̈onnen. Bei der
Summe und Skalarmultiplikation ist das Ergebnis wieder einem×n-Matrix, also von
der gleichen Gr ̈oße wie die urspr ̈unglichen Matrizen.

Beispiel 12.6.Fur ̈

```
A=
```
##### [

##### 1 2 3

##### 4 5 6

##### ]

```
und B=
```
##### [

##### 3 1 − 1

##### 2 0 1

##### ]

sind

```
A+B=
```
##### [

##### 1 + 3 2 + 1 3 + (−1)

##### 4 + 2 5 + 0 6 + 1

##### ]

##### =

##### [

##### 4 3 2

##### 6 5 7

##### ]

und

```
(−1)·A=
```
##### [

##### − 1 − 2 − 3

##### − 4 − 5 − 6

##### ]

##### , 2 ·B=

##### [

##### 6 2 − 2

##### 4 0 2

##### ]

##### .

Da die Addition und Skalarmultiplikation von Matrizen eintragsweise definiert sind,
gelten die gleichen Rechenregeln wie f ̈ur reelle oder komplexe Zahlen.

Satz 12.7(Rechenregeln f ̈ur die Addition).F ̈urA,B,C∈Km,ngilt

```
1) (A+B) +C=A+ (B+C),
2) A+B=B+A.
```
Satz 12.8(Rechenregeln f ̈ur die Skalarmultiplikation). F ̈urA,B∈Km,nundα,β∈K
gilt

1) α(βA) = (αβ)A,
2) α(A+B) =αA+αB,
3) (α+β)A=αA+βA,
Daher istKm,nmit der Addition und Skalarmultiplikation einK-Vektorraum. Dieser
hat die Basis{Ei,j|i= 1,...,m;j= 1,...,n}, wobeiEi,j∈Km,neine 1 in Eintrag
(i,j) hat und alle anderen Eintr ̈age Null sind. Daher ist dim(Km,n) =mn.


### 12.3 Matrizenmultiplikation

Nach der Addition und Skalarmultiplikation fuhren wir nun die Multiplikation zweier ̈
Matrizen ein. Wir beginnen mit dem Spezialfall eines Zeilen- und eines Spaltenvektors.
Fur ̈

##### A=

##### [

```
a 1 , 1 a 1 , 2 ... a 1 ,n
```
##### ]

```
∈K^1 ,n und B=
```
##### 

##### 

##### 

##### 

```
b 1 , 1
b 2 , 1
..
.
bn, 1
```
##### 

##### 

##### 

##### 

```
∈Kn,^1
```
definieren wir das ProduktABdurch

```
AB:=
```
##### [

```
a 1 , 1 b 1 , 1 +a 1 , 2 b 2 , 1 +...+a 1 ,nbn, 1
```
##### ]

##### =

```
[∑n
k=1a^1 ,kbk,^1
```
##### ]

##### ∈K^1 ,^1.

Das Resultat ist eine 1×1-Matrix. Ihr Eintrag entsteht, indem wir die ersten Elemente
vonAundBmultiplizieren, dann die zweiten, dritten,...Elemente und alles addieren.
Damit das Produkt definiert ist, mussAgenau so viele Spalten haben, wieBZeilen hat.

Beispiel 12.9.Es sind

```
[
2 1
```
##### ]

##### [

##### − 1

##### 3

##### ]

##### =

##### [

##### 2 ·(−1) + 1· 3

##### ]

##### =

##### [

##### 1

##### ]

##### ∈K^1 ,^1 ,

##### [

##### 1 2 3

##### ]

##### 

##### 

##### 7

##### − 1

##### 5

##### 

##### =[ 1 ·7 + 2·(−1) + 3· 5 ]=[ 20 ]∈K^1 ,^1.

Fur zwei allgemeine Matrizen ̈ AundBmacht man dies analog f ̈ur jede Zeile vonA
und jede Spalte vonB: Multiplikation von Zeileiund Spaltejergibt Eintrag (i,j) des
Produkts.

Definition 12.10(Matrizenmultiplikation).SeienA=

##### [

```
ai,j
```
##### ]

```
∈Km,nundB=
```
##### [

```
bi,j
```
##### ]

##### ∈

Kn,p. Dann ist

```
AB:=
```
```
[∑n
k=1ai,kbk,j
```
##### ]

##### =

##### [

```
ai, 1 b 1 ,j+ai, 2 b 2 ,j+...+ai,nbn,j
```
##### ]

```
∈Km,p.
```
Ausgeschrieben bedeutet das

##### AB=

##### 

##### 

##### 

##### 

```
∑n
k=1a^1 ,kbk,^1
```
```
∑n
k=1a^1 ,kbk,^2 ...
```
```
∑n
∑n k=1a^1 ,kbk,p
k=1a^2 ,kbk,^1
```
```
∑n
k=1a^2 ,kbk,^2 ...
```
```
∑n
k=1a^2 ,kbk,p
..
.
```
##### ..

##### .

##### ... ..

##### ∑.

```
n
k=1am,kbk,^1
```
```
∑n
k=1am,kbk,^2 ...
```
```
∑n
k=1am,kbk,p
```
##### 

##### 

##### 

##### 

##### .

Beispiel 12.11. 1) Es ist

```
[
1 2
0 3
```
##### ][

##### 7

##### − 1

##### ]

##### =

##### [

##### 1 ·7 + 2·(−1)

##### 0 ·7 + 3·(−1)

##### ]

##### =

##### [

##### 5

##### − 3

##### ]

##### .


```
2) Es sind




```
##### 1 1

##### 1 0

##### 0 1

##### 1 − 1

##### 

##### 

##### 

##### 

##### [

##### 5 1 1

##### 4 0 1

##### ]

##### =

##### 

##### 

##### 

##### 

##### 1 ·5 + 1·4 1·1 + 1·0 1·1 + 1· 1

##### 1 ·5 + 0·4 1·1 + 0·0 1·1 + 0· 1

##### 0 ·5 + 1·4 0·1 + 1·0 0·1 + 1· 1

##### 1 · 5 − 1 ·4 1· 1 − 1 ·0 1· 1 − 1 · 1

##### 

##### 

##### 

##### =

##### 

##### 

##### 

##### 

##### 9 1 2

##### 5 1 1

##### 4 0 1

##### 1 1 0

##### 

##### 

##### 

##### 

```
und
[
1 2 3 − 1
0 1 0 2
```
##### ]

##### 

##### 

##### 

##### 

##### 1 − 1 0

##### 1 2 − 1

##### 0 0 3

##### 1 0 1

##### 

##### 

##### 

##### =

##### [

##### 2 3 6

##### 3 2 1

##### ]

##### .

```
3) Das Produkt
AB=
```
##### [

##### 1 2

##### 0 3

##### ]

##### [

##### 1 2

##### ]

```
ist nicht definiert, da die Anzahl der Spalten vonAverschieden von der Anzahl
der Zeilen vonBist.
```
```
4) Es ist [
1 0
0 1
```
##### ][

```
a b
c d
```
##### ]

##### =

##### [

```
1 ·a+ 0·c 1 ·b+ 0·d
0 ·a+ 1·c 0 ·b+ 1·d
```
##### ]

##### =

##### [

```
a b
c d
```
##### ]

##### .

```
Multiplikation mit derEinheitsmatrixI 2 =
```
##### [

##### 1 0

##### 0 1

##### ]

```
̈andert die zweite Matrix nicht.
Das ist genauso wie 1·x=xf ̈ur reelle oder komplexexgilt.
Wir fassen einige Rechenregeln fur die Matrizenmultiplikation zusammen. ̈
```
Satz 12.12(Rechenregeln f ̈ur die Matrizenmultiplikation).F ̈ur MatrizenA,B,Cmit
geeigneter Gr ̈oße und f ̈urα∈Kgilt

```
1) A(BC) = (AB)C,
```
```
2) A(B+C) =AB+AC,
```
```
3) (A+B)C=AC+BC,
```
```
4) α(AB) = (αA)B=A(αB),
```
```
5) ImA=A=AInf ̈urA∈Km,n.
```
Im Allgemeinen istAB 6 =BA.

Ein wesentlicher Unterschied zur Multiplikation von reellen oder komplexen Zahlen
ist, dass die Matrizenmultiplikationnicht kommutativist, d.h. im Allgemeinen sindAB
undBAverschieden. Zum Beispiel ist

```
AB=
```
##### [

##### 1 1

##### 0 1

##### ][

##### 1 0

##### 1 1

##### ]

##### =

##### [

##### 2 1

##### 1 1

##### ]

##### 6 =

##### [

##### 1 1

##### 1 2

##### ]

##### =

##### [

##### 1 0

##### 1 1

##### ][

##### 1 1

##### 0 1

##### ]

##### =BA.


Das war auch bei der Komposition von Abbildungen in Abschnitt 5.2 so, und in der
Tat gibt es hier einen tieferen Zusammenhang: Eine MatrixA∈ Km,n definiert eine
AbbildungA : Kn → Km,~x 7→ A~x. Daher entspricht das ProduktAB gerade der
Komposition der beiden Abbildungen~x7→B~xund~x7→A~x.

### 12.4 Inverse

Wir betrachten nunquadratischeMatrizen, das sind MatrizenA∈Kn,n(mitm=n)
und fragen, wann eine Matrix eine Inverse besitzt. Dies entspricht der Frage, wann die
AbbildungA:Kn→Kn,~x7→A~x, eine Umkehrabbildung (= Inverse) besitzt; vergleiche
Definition 5.9.

Definition 12.13(Inverse).Eine quadratische MatrixA∈Kn,nheißtinvertierbar, falls
es eine MatrixB∈Kn,ngibt mit

```
BA=In und AB=In.
```
Die MatrixB ist dann eindeutig bestimmt, wird dieInverse vonAgenannt und mit
A−^1 bezeichnet.

Beweis der Eindeutigkeit. Wir nehmen an, dassAzwei Inversen hat, und zeigen, dass
diese gleich sind. SindB,C∈Kn,nmitBA=In=ABundCA=In=AC, so folgt

```
B=BIn=B(AC) = (BA)C=InC=C,
```
alsoB=C. Daher ist die Inverse, falls sie existiert, eindeutig bestimmt.

Per Definition gelten alsoA−^1 A=InundAA−^1 = In. Man kann zeigen, dass es
fur quadratische Matrizen ausreicht, nur ̈ eineder beiden Gleichungen BA=In und
AB=Inzuuberpr ̈ ̈ufen, die andere gilt dann automatisch.

Beispiel 12.14.Die MatrixA= [1 10 1] ist invertierbar mit InversenA−^1 =

##### [ 1 − 1

```
0 1
```
##### ]

```
, denn
```
```
A−^1 A=
```
##### [

##### 1 − 1

##### 0 1

##### ][

##### 1 1

##### 0 1

##### ]

##### =

##### [

##### 1 ·1 + (−1)·0 1·1 + (−1)· 1

##### 0 ·1 + 1· 0 0 ·1 + 1· 1

##### ]

##### =

##### [

##### 1 0

##### 0 1

##### ]

##### =I 2.

Satz 12.15.SindA,B∈Kn,ninvertierbar, so gelten

```
1)A−^1 ist invertierbar mit(A−^1 )−^1 =A.
2)ABist invertierbar mit(AB)−^1 =B−^1 A−^1.
```
Beweis. Eigenschaft1)folgt direkt aus der Definition: DaAA−^1 =InundA−^1 A=In
istA−^1 nach Definition invertierbar mit (A−^1 )−^1 =A.
Eigenschaft2)geht ̈ahnlich: Da wir schon einen Kandidaten fur die Inverse haben, ̈
rechnen wir die Definition nach:

```
(B−^1 A−^1 )(AB) =B−^1 A−^1 AB=B−^1 InB=B−^1 B=In.
```
Genauso findet man (AB)(B−^1 A−^1 ) =In. Daher istABinvertierbar mit (eindeutiger)
Inversen (AB)−^1 =B−^1 A−^1.


Beispiel 12.16.IstA= [1 11 1]∈R^2 ,^2 invertierbar? Wenn ja, dann gibt esB=

```
[a b
c d
```
##### ]

```
mit
[
1 0
0 1
```
##### ]

##### =BA=

##### [

```
a b
c d
```
##### ][

##### 1 1

##### 1 1

##### ]

##### =

##### [

```
a+b a+b
c+d c+d
```
##### ]

Aus der ersten Zeile sehen wir 1 =a+bund 0 =a+b, d.h. den Widerspruch 1 = 0.
Also istAnicht invertierbar.

Im Beispiel konnten wir direkt nachrechnen, obA invertierbar ist oder nicht. Im
Allgemeinen ist es nicht so einfach, die Inverse einer gegebenen Matrix zu berechnen
(wenn sie denn existiert). Wir werden darauf in Vorlesung 14 n ̈aher eingehen.

### 12.5 Transposition

Zum Abschluss dieser Vorlesung erkl ̈aren wir eine weitere Operation f ̈ur Matrizen.

Definition 12.17(Transponierte).DieTransponierteder MatrixA=

##### [

```
ai,j
```
##### ]

∈Km,nist
dien×m-Matrix
AT:=

##### [

```
bi,j
```
##### ]

```
∈Kn,m, wobeibi,j=aj,i.
```
Bei der Transposition werden also die Zeilen vonAzu den Spalten vonAT. (Lies:AT
als”Atransponiert“.)

Beispiel 12.18. 1) Die Transponierte von

##### A=

##### [

##### 1 2 3

##### 4 5 6

##### ]

```
∈R^2 ,^3 ist AT=
```
##### 

##### 

##### 1 4

##### 2 5

##### 3 6

##### 

##### ∈R^3 ,^2.

```
2) Beim Transponieren werden Zeilenvektoren zu Spaltenvektoren, und Spaltenvek-
toren zu Zeilenvektoren:


```
##### 2

##### 3

##### − 1

##### 

##### 

```
T
=
```
##### [

##### 2 3 − 1

##### ]

```
und
```
##### [

##### 2 3 − 1

##### ]T

##### =

##### 

##### 

##### 2

##### 3

##### − 1

##### 

##### .

Satz 12.19 (Rechenregeln fur die Transponierte) ̈. F ̈ur A,B ∈ Km,n,C ∈Kn,` und
α∈Kgilt:

```
1) (AT)T=A,
```
```
2) (A+B)T=AT+BT,
```
```
3) (αA)T=αAT,
```
```
4) (AC)T=CTAT.
```

Fur Matrizen ̈ uber ̈ Cspielt die Kombination von Transponieren und komplexer Kon-
jugation eine wichtige Rolle.

Definition 12.20.DieAdjungierteder MatrixA=

##### [

```
ai,j
```
##### ]

```
∈Cm,nist dien×m-Matrix
```
```
AH:=
```
##### [

```
bi,j
```
##### ]

```
∈Cn,m, wobeibi,j=aj,i.
```
(Lies:AHals”Ahermitesch“ oder”Aadjungiert“.)

Die Adjungierte vonAist also die Transponierte, wo zus ̈atzlich alle Eintr ̈age komplex
konjugiert werden. StattAHwird auch die BezeichnungA∗verwendet.

Beispiel 12.21.Es ist

```
[
1
1 +i
```
##### ]H

##### =

##### [

```
1 1−i
```
##### ]

##### ,

##### [

```
1 +i 2 2 + 3i
− 1 0 − 2 i
```
##### ]H

##### =

##### 

##### 

```
1 −i − 1
2 0
2 − 3 i 2 i
```
##### 

##### .

Satz 12.22(Rechenregeln f ̈ur die Adjungierte).F ̈urA,B∈Cm,n,C∈Cn,`undα∈C
gilt:

```
1)(AH)H=A,
```
```
2)(A+B)H=AH+BH,
```
```
3)(αA)H=αAH,
```
```
4)(AC)H=CHAH.
```

Vorlesung 13

## 13 Lineare Gleichungssysteme

In dieser Vorlesung lernen wir ein Verfahren, um lineare Gleichungssysteme zu l ̈osen.

### 13.1 Matrixschreibweise eines linearen Gleichungssystems

Einlineares Gleichungssystem(LGS) mitmGleichungen innUnbekannten hat die Form

```
a 1 , 1 x 1 +a 1 , 2 x 2 +...+a 1 ,nxn=b 1
a 2 , 1 x 1 +a 2 , 2 x 2 +...+a 2 ,nxn=b 2
..
.
am, 1 x 1 +am, 2 x 2 +...+am,nxn=bm.
```
##### (13.1)

DiexjheißenUnbekannteoderVariablenund sind gesucht. DieKoeffizientenai,j∈K
undbi∈Ksind gegeben. Wir sprechen von einemreellen LGS, fallsK=Rist (alsoai,j
undbireell sind), und von einemkomplexen LGS, fallsK=Cist. Sind allebi= 0, so
heißt das LGShomogen, andernfalls heißt das LGSinhomogen(mindestens einbi 6 = 0).
Das lineare Gleichungssystem (13.1) k ̈onnen wir auch schreiben als





```
a 1 , 1 a 1 , 2 ... a 1 ,n
a 2 , 1 a 2 , 2 ... a 2 ,n
..
.
```
##### ..

##### .

##### ... ..

##### .

```
am, 1 am, 2 ... am,n
```
##### 

##### 

##### 

##### 

##### ︸ ︷︷ ︸

```
=A
```
##### 

##### 

##### 

##### 

```
x 1
x 2
..
.
xn
```
##### 

##### 

##### 

##### 

##### ︸︷︷︸

```
=x
```
##### =

##### 

##### 

##### 

##### 

```
b 1
b 2
..
.
bm
```
##### 

##### 

##### 

##### 

##### ︸︷︷︸

```
=b
```
##### ,

also als
Ax=b.

Dabei heißt

- A∈Km,ndieKoeffizientenmatrixdes LGS,
- b∈Kmdierechte Seiteoder dieInhomogenit ̈atdes LGS.


EineL ̈osungdes linearen Gleichungssystems ist ein Vektorx∈KnmitAx=b. Aus-
geschrieben erfullen die Eintr ̈ ̈age einer L ̈osung die Gleichungen (13.1). Die Menge aller
L ̈osungen heißtL ̈osungsmenge,

```
L=L(A,b):={x∈Kn|Ax=b}.
```
IstA∈Kn,ninvertierbar, dann multiplizieren wir das LGSAx=bmit der Inversen
A−^1 und erhalten
x=A−^1 Ax=A−^1 b.

Die L ̈osung ist dann eindeutig:L={A−^1 b}. Im Allgemeinen istAaber nicht invertierbar,
oder wir kennen die Inverse nicht. Dann brauchen wir einen anderen Weg, um das LGS
zu l ̈osen. Dazu betrachten wir zun ̈achst ein Beispiel.

Beispiel 13.1.Wir betrachten das lineare Gleichungssystem

```
x 1 + 2x 2 −x 3 = 3
x 2 + 2x 3 = 5
2 x 3 = 4
```
```
, also
```
##### 

##### 

##### 1 2 − 1

##### 0 1 2

##### 0 0 2

##### 

##### 

##### 

##### 

```
x 1
x 2
x 3
```
##### 

##### =

##### 

##### 

##### 3

##### 5

##### 4

##### 

##### .

Dieses LGS k ̈onnen wir einfach von
”
unten nach oben“ l ̈osen: Die letzte Gleichung ist
2 x 3 = 4, woraus wirx 3 = 2 erhalten. Setzen wirx 3 = 2 in die zweite Gleichung ein, so ist
nur nochx 2 unbekannt und wir findenx 2 = 1. Mitx 2 = 1 undx 3 = 2 k ̈onnen wir ganz
einfachx 1 = 3− 2 x 2 +x 3 = 3 berechnen. Die Rechnung ist einfach, da die Matrix eine
Dreiecksform hat und so beim L ̈osen
”
von unten nach oben“ immer nur eine Variable
auf einmal zu berechnen ist. Dieses Vorgehen nennt manR ̈uckw ̈artssubstitution.
Die L ̈osung des LGS

```
x 1 + 2x 2 −x 3 = 3
x 1 + 3x 2 +x 3 = 8
x 1 + 2x 2 +x 3 = 7
```
```
, also
```
##### 

##### 

##### 1 2 − 1

##### 1 3 1

##### 1 2 1

##### 

##### 

##### 

##### 

```
x 1
x 2
x 3
```
##### 

##### =

##### 

##### 

##### 3

##### 8

##### 7

##### 

##### ,

ist hingegen nicht so einfach zu bestimmen, da in jeder Gleichung alle Variablen vor-
kommen.

Um ein lineares GleichungssystemAx=bzu l ̈osen, bringen wir die Matrix auf eine
obere Dreiecksform. Dies gelingt mit dem Gauß-Algorithmus.

### 13.2 Der Gauß-Algorithmus

Im Gauß-Algorithmus sind nur die folgendenelementaren Zeilenoperationenerlaubt:
1) Vertauschen von zwei Zeilen,
2) Multiplizieren einer Zeile mit einer Zahlλ 6 = 0 (wobeiλ∈K), und
3) Addition des Vielfachen einer Zeile zu einer anderen Zeile.


Der Gauß-Algorithmus SeiA∈Km,n. Dann kannAdurch elementare Zeilenopera-
tionen aufZeilenstufenform(ZSF) gebracht werden, d.h. auf die Form

##### C=

##### 

##### 

##### 

##### 

##### 

##### 

```
0 c 1 ,j 1 ∗ ∗ ∗ ∗ ∗ ∗ ∗ ∗
0 0 0 c 2 ,j 2 ∗ ∗ ∗ ∗ ∗ ∗
0 0 0 0 0 c 3 ,j 3 ∗ ∗ ∗ ∗
..
.
```
##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### . 0 0

##### ...

##### ∗ ∗

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

. 0 cr,jr ∗
0 0 0 0 0 0 0 0 0 0

##### 

##### 

##### 

##### 

##### 

##### 

wobeici,ji 6 = 0 fur ̈ i= 1,...,rund∗f ̈ur beliebige Eintr ̈age steht (null oder ungleich
null)^1. Die Eintr ̈ageci,ji bei den”Stufen“ werden auch Pivotelementegenannt. Wir
k ̈onnen wie folgt vorgehen, umA 6 = 0 in Zeilenstufenform zu bringen:
1) Suche die erste von Null verschiedene Spaltej 1.
2) Suche in dieser Spalte den ersten Eintrag ungleich Null (Pivotelement) und tausche
ihn ggf. in die erste Zeile. Wir haben nun eine Matrix der Form
[
0 c 1 ,j 1 ∗
0 ∗ ∗

##### ]

```
mit c 1 ,j 16 = 0.
```
```
3) Unter dem Pivotelementc 1 ,j 1 werden alle Eintr ̈age eliminiert, indem geeignete
Vielfache der ersten Zeile von den anderen Zeilen abgezogen werden.
4) Rekursion: Ist die Matrix in Zeilenstufenform, so sind wir fertig. Andernfalls haben
wir die Form [
0 c 1 ,j 1 ∗
0 0 A 1
```
##### ]

```
mit c 1 ,j 16 = 0.
```
Die erste Zeile und die ersten Spalten (bis Spaltej 1 ) bleiben wie sie sind, und wir
wenden das gleiche Verfahren auf die kleinere MatrixA 1 an.
Die Zeilenstufenform der NullmatrixA= 0 istC= 0.

Beispiel 13.2.Wir bringen die Matrix

##### A=

##### 

##### 

##### 0 0 1 − 2

##### 0 2 1 1

##### 0 4 3 3

##### 

##### ∈R^3 ,^4

in Zeilenstufenform. DaA 6 = 0 suchen wir die erste von Null verschiedene Spalte:j 1 = 2.
Der erste Eintrag in Spalte 2 ist Null, der erste Eintrag ungleich Null ist in Zeile 2, daher

(^1) Genauer ist die MatrixC∈Km,nin Zeilenstufenform, falls gilt:
1) Der erste Nichtnulleintrag einer Zeile ist weiter rechts als die ersten Nichtnulleintr ̈age der vorhe-
rigen Zeilen.
2) Alle Zeilen mit nur Nullen sind unter den Zeilen mit Nichtnulleintr ̈agen.


tauschen wir die erste und zweite Zeile:

```


```
##### 0 0 1 − 2

##### 0 2 1 1

##### 0 4 3 3

##### 

##### I↔→II

##### 

##### 

##### 0 2 1 1

##### 0 0 1 − 2

##### 0 4 3 3

##### 

##### III→−^2 I

##### 

##### 

##### 0 2 1 1

##### 0 0 1 − 2

##### 0 0 1 1

##### 

##### 

Im zweiten Schritt haben wir die Eintr ̈age unterc 1 ,j 1 eliminiert, indem wir zwei mal
die erste Zeile von der dritten Zeile abgezogen haben. Die Matrix ist noch nicht in
Zeilenstufenform. Die erste Zeile bleibt ab jetzt unver ̈andert und wir fahren mit der
Matrix rechts unten fort: 

```

```
##### 0 2 1 1

##### 0 0 1 − 2

##### 0 0 1 1

##### 

##### 

Hier m ̈ussen wir keine Zeilen tauschen, da gleich der erste Eintrag von Null verschieden
ist. Wir eliminieren direkt unter dem Eintrag (2,j 2 ) = (2,3):

```


```
##### 0 2 1 1

##### 0 0 1 − 2

##### 0 0 1 1

##### 

##### III→−II

##### 

##### 

##### 0 2 1 1

##### 0 0 1 − 2

##### 0 0 0 3

##### 

##### 

Die Matrix ist in Zeilenstufenform, so dass wir fertig sind.

Bemerkung 13.3. 1) Der Gauß-Algorithmus gibt einen Rechenweg an, umAin ZSF
zu bringen. Man kannAaber auch auf andere ArtAmit elementaren Zeilenope-
rationen in Zeilenstufenform bringen:Der Rechenweg ist egal!

```
2) Die Zeilenstufenform einer MatrixA 6 = 0 ist nicht eindeutig bestimmt: IstCeine
Zeilenstufenform vonA, so k ̈onnen wir eine der von Null verschiedenen Zeilen
mit einer Zahl ungleich Null multiplizieren und das Ergebnis ist immer noch eine
Zeilenstufenform.
```
Wir k ̈onnen die Zeilenstufenform eindeutig machen, indem wir verlangen, dass die
Pivotelementeci,ji= 1 sind (dazu teilen wir Zeileidurchci,ji), und ̈uber diesen Einsen
an den”Stufen“ Nullen erzeugen. Das ergibt die sogenanntenormierte Zeilenstufenform
(NZSF):

##### C=

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 0 1 ∗ 0 ∗ 0 ∗ ∗ 0 ∗

##### 0 0 0 1 ∗ 0 ∗ ∗ 0 ∗

##### 0 0 0 0 0 1 ∗ ∗ 0 ∗

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### . 0 0 ... 0 ∗

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### . 0 1 ∗

##### 0 0 0 0 0 0 0 0 0 0

##### 

##### 

##### 

##### 

##### 

##### 

##### 

Man kann zeigen: Die normierte Zeilenfstufenform einer Matrix ist eindeutig.


Beispiel 13.4.Wir berechnen die normierte Zeilenstufenform der Matrix



##### 2 4 1 1

##### 0 0 1 − 2

##### 0 0 0 3

##### 

##### 

in Zeilenstufenform. Dabei arbeiten wir uns”von unten noch oben“ durch:



##### 2 4 1 1

##### 0 0 1 − 2

##### 0 0 0 3

##### 

##### 

(^13) III
→

##### 

##### 

##### 2 4 1 1

##### 0 0 1 − 2

##### 0 0 0 1

##### 

##### 

```
II+2III
I−→III
```
##### 

##### 

##### 2 4 1 0

##### 0 0 1 0

##### 0 0 0 1

##### 

##### I−→II

##### 

##### 

##### 2 4 0 0

##### 0 0 1 0

##### 0 0 0 1

##### 

##### 

```
1
→^2 I
```
##### 

##### 

##### 1 2 0 0

##### 0 0 1 0

##### 0 0 0 1

##### 

##### 

und diese Matrix ist in normierter Zeilenstufenform.

### 13.3 Anwendung auf lineare Gleichungssysteme

Um das LGSAx=bzu l ̈osen, wenden wir den Gauß-Algorithmus auf dieerweiterte
Koeffizientenmatrix

```
[A,b] = [A|b] =
```
##### 

##### 

##### 

##### 

##### 

```
a 1 , 1 a 1 , 2 ... a 1 ,n b 1
a 2 , 1 a 2 , 2 ... a 2 ,n b 2
..
.
```
##### ..

##### .

##### ... ..

##### .

##### ..

##### .

```
am, 1 am, 2 ... am,n bm
```
##### 

##### 

##### 

##### 

##### 

an, bei derbrechts an die KoeffizientenmatrixAangeh ̈ang wird. Die elementaren Zeilen-
operationen auf der erweiterten Koeffizientenmatrix entsprechen den selben Operationen
mit den Gleichungen (Zeilen tauschen entspricht Gleichungen tauschen, Multiplikation
einer Zeile mitα 6 = 0 entspricht der Multiplikation der entsprechenden Gleichung mit
α 6 = 0, Addition eines Vielfachen von Zeileizu Zeilej entspricht der Addition eines
Vielfachen von Gleichungizu Gleichungj.) Da alle Operationen r ̈uckg ̈angig gemacht
werden k ̈onnen (alsoAquivalenzumformungen sind), ̈ bleibt die L ̈osungsmenge gleich!
Ist die Matrix in ZSF oder NZSF, so k ̈onnen wir die L ̈osung(en) des Gleichungssys-
tems durch Ruckw ̈ ̈artssubstitution bestimmen.

Beispiel 13.5. 1) Wir betrachten noch einmal das LGS

```
x 1 + 2x 2 −x 3 = 3
x 1 + 3x 2 +x 3 = 8
x 1 + 2x 2 +x 3 = 7
```
```
, also
```
##### 

##### 

##### 1 2 − 1

##### 1 3 1

##### 1 2 1

##### 

##### 

##### 

##### 

```
x 1
x 2
x 3
```
##### 

##### =

##### 

##### 

##### 3

##### 8

##### 7

##### 

##### ,

```
aus Beispiel 13.1 und bringen die erweiterte Koeffizientenmatrix in Zeilenstufen-
form:


```
##### 1 2 − 1 3

##### 1 3 1 8

##### 1 2 1 7

##### 

##### II→−I

##### 

##### 

##### 1 2 − 1 3

##### 0 1 2 5

##### 1 2 1 7

##### 

##### III→−I

##### 

##### 

##### 1 2 − 1 3

##### 0 1 2 5

##### 0 0 2 4

##### 

##### .


```
Durch Ruckw ̈ ̈artssubstitution erhalten wir die (eindeutige) L ̈osung
```
```
x=
```
##### 

##### 

##### 3

##### 1

##### 2

##### 

```
, d.h. L=
```
##### 

##### 

##### 

##### 

##### 

##### 3

##### 1

##### 2

##### 

##### 

##### 

##### 

##### 

##### .

```
Wir k ̈onnen die erweiterte Koeffizientenmatrix auch auf normierte Zeilenstufenform
bringen:


```
##### 1 2 − 1 3

##### 0 1 2 5

##### 0 0 2 4

##### 

##### 

(^12) III
→

##### 

##### 

##### 1 2 − 1 3

##### 0 1 2 5

##### 0 0 1 2

##### 

##### 

```
II− 2 III
I+→III
```
##### 

##### 

##### 1 2 0 5

##### 0 1 0 1

##### 0 0 1 2

##### 

##### 

##### I−→ 2 II

##### 

##### 

##### 1 0 0 3

##### 0 1 0 1

##### 0 0 1 2

##### 

##### .

```
Wir erhalten die gleiche L ̈osung wie eben. Wir beobachten: Fur die NZSF m ̈ ussen ̈
wir ein paar Zeilenumformungen mehr machen, dafur l ̈ ̈asst sich die L ̈osung des
LGS leichter ablesen.
```
2) Wir betrachten nun das LGS
[
1 1
2 2

##### ]

```
x=
```
##### [

##### 2

##### 0

##### ]

```
und bringen die erweiterte Koeffizientenmatrix in ZSF:
[
1 1 2
2 2 0
```
##### ]

##### II→− 2 I

##### [

##### 1 1 2

##### 0 0 − 4

##### ]

##### .

```
Die zweite Zeile bedeutet 0x 1 + 0x 2 =−4, was unm ̈oglich ist. Dieses LGS hat also
keine L ̈osung, d.h.L=∅.
```
3) F ̈ur das LGS [
1 1
2 2

##### ]

```
x=
```
##### [

##### 2

##### 4

##### ]

```
erhalten wir die ZSF (und NZSF)
```
```
[A,b] =
```
##### [

##### 1 1 2

##### 2 2 4

##### ]

##### II→− 2 I

##### [

##### 1 1 2

##### 0 0 0

##### ]

##### .

```
Die zweite Zeile bedeutet 0x 1 +0x 2 = 0, was f ̈ur allex 1 ,x 2 erf ̈ullt ist. Die erste Zeile
ergibtx 1 +x 2 = 2, oderx 1 = 2−x 2. Die beiden Variablen h ̈angen also voneinander
ab. Eine k ̈onnen wir frei w ̈ahlen, dann ist die andere eindeutig festgelegt. W ̈ahlen
wir zum Beispielx 2 =t∈R(diese Umbenennung dient nur der Verdeutlichung),
so istx 1 = 2−tfestgelegt. Das LGS hat also unendlich viele L ̈osungen,
```
```
L=
```
##### {[

```
2 −t
t
```
##### ]∣∣

##### ∣

```
∣t∈R
```
##### }

##### .


4) Wir wollen das LGS 

```

```
##### 2 −1 3 0

##### 2 −1 3 1

##### − 2 4 0 1

##### 

```
x=
```
##### 

##### 

##### 1

##### 0

##### 1

##### 

##### 

```
l ̈osen und bringen dazu die erweiterte Koeffizientenmatrix in Zeilenstufenform:
```
```
[
A b
```
##### ]

##### =

##### 

##### 

##### 2 −1 3 0 1

##### 2 −1 3 1 0

##### − 2 4 0 1 1

##### 

##### →

##### 

##### 

##### 2 −1 3 0 1

##### 0 0 0 1 − 1

##### 0 3 3 1 2

##### 

##### 

##### →

##### 

##### 

##### 2 −1 3 0 1

##### 0 3 3 1 2

##### 0 0 0 1 − 1

##### 

##### .

```
Nun k ̈onnen wir die L ̈osung ablesen: Die dritte Gleichung ergibtx 4 =−1 und somit
haben wir in der zweiten Gleichung 3x 2 + 3x 3 + 1x 4 = 2. Die Variablex 3 kann frei
gew ̈ahlt werden, zur Verdeutlichung schreiben wirx 3 =t∈R. Damit erhalten wir
x 2 = 1−t. Verwenden wir dies in der ersten Zeile, so ergibt sichx 1 = 1− 2 t, also
```
##### L=

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

```
1 − 2 t
1 −t
t
− 1
```
##### 

##### 

##### 

##### 

##### ∣∣

##### ∣∣

##### ∣∣

##### ∣∣

```
t∈R
```
##### 

##### 

##### 

##### 

##### .

```
In diesem Gleichungssystem sind die Variablen x 1 ,x 2 ,x 3
”
gekoppelt“, und wir
k ̈onnen eine der drei frei w ̈ahlen, dann sind die anderen beiden eindeutig festgelegt.
Welche man w ̈ahlt ist im Endeffekt gleich. Es ist aber durchaus sinnvoll, die nicht
zu Pivotelementen (Stufen) geh ̈orenden Variablen als Parameter (= frei) zu w ̈ahlen
(im Beispielx 3 ), dann sind die zu Stufen geh ̈orenden Variablen eindeutig bestimmt
(im Beispielx 1 ,x 2 ,x 4 ). Sie k ̈onnen von den frei gew ̈ahlten Variablen abh ̈angen (wie
x 1 ,x 2 ) oder auch nicht (wiex 4 ).
Auch hier k ̈onnen wir die erweiterte Koeffizientenmatrix auf normierte Zeilenstu-
fenform bringen und dann die L ̈osung ablesen. Wir finden


```
##### 2 −1 3 0 1

##### 0 3 3 1 2

##### 0 0 0 1 − 1

##### 

##### II−→III

##### 

##### 

##### 2 −1 3 0 1

##### 0 3 3 0 3

##### 0 0 0 1 − 1

##### 

##### 

(^13) II
→

##### 

##### 

##### 2 −1 3 0 1

##### 0 1 1 0 1

##### 0 0 0 1 − 1

##### 

##### 

##### I+→II

##### 

##### 

##### 2 0 4 0 2

##### 0 1 1 0 1

##### 0 0 0 1 − 1

##### 

##### 

(^12) I
→

##### 

##### 

##### 1 0 2 0 1

##### 0 1 1 0 1

##### 0 0 0 1 − 1

##### 

##### .

```
An der normierten Zeilenstufenform k ̈onnen wir wieder sehr gut die Pivotvariablen
x 1 ,x 2 ,x 4 (geh ̈oren zu einer Spalte mit einer”Stufe“) und die frei w ̈ahlbare Variable
x 3 erkennen und sehr einfach die L ̈osung ablesen:x 4 =−1, dannx 2 +x 3 = 1, also
x 2 = 1−x 3 , undx 1 + 2x 3 = 1, alsox 1 = 1− 2 x 3. Schreiben wir wiederx 3 =t∈R,
so erhalten wir die selbe L ̈osungsmenge wie eben.
```

Warnung vor dem folgenden
”
Trick“: Wir l ̈osen das LGS
[
1 1
1 1

##### ]

```
x=
```
##### [

##### 0

##### 0

##### ]

und bringen dazu die erweiterte Koeffizientenmatrix auf ZSF. Rechnen wir in einem
Schritt simultan 1. Zeile - 2. Zeile und 2. Zeile - 1. Zeile, so ist
[
1 1 0
1 1 0

##### ]I−II

##### II→−I

##### [

##### 0 0 0

##### 0 0 0

##### ]

Das zweite LGS wird von jedemx∈K^2 gel ̈ost, das erste LGS hingegen nur von Vektoren

der Formx=

##### [

```
a
−a
```
##### ]

```
mita∈K. Was ist passiert?Beim simultan nach oben und unten
```
Eliminieren ist Information vernichtet worden! Das sollte man vermeiden.

Wie viele L ̈osungen hat ein LGS? In den Beispielen haben wir gesehen, dass es
keine L ̈osungen des LGSAx=bgeben kann, oder genau eine L ̈osung, oder unendlich
viele L ̈osungen. Welcher Fall eintritt h ̈angt von der Zeilenstufenform vonAund [A,b]
ab. N ̈aheres diskutieren wir in Vorlesung 14, in der wir auch sehen werden, dass es keine
weiteren M ̈oglichkeiten gibt.

### 13.4 Struktur der L ̈osungsmenge

Wir untersuchen nun die Struktur der L ̈osungsmenge des LGSAx=bmitA∈Km,n
undb∈Km.

```
1) homogenes LGS, d.h.b= 0. Dann istL(A,0) ein Teilraum vonKn:
```
```
(a)A0 = 0, d.h. 0∈L(A,0),
(b)x,y ∈L(A,0), d.h.Ax= 0 undAy= 0, dann istA(x+y) =Ax+Ay=
0 + 0 = 0, alsox+y∈L(A,0)
(c) Istλ∈Kundx∈L(A,0), d.h.Ax= 0, dann istA(λx) =λAx=λ0 = 0,
alsoλx∈L(A,0).
Daher istL(A,0) ein Teilraum vonKnnach dem Teilraumkriterium (Satz 10.4).
```
```
2) inhomogenes LGS, d.h.b 6 = 0. Dann istL(A,b) kein Teilraum, da 0∈/L(A,0) ist:
A0 = 0 6 =b.
```
Ist das inhomogene lineare GleichungssystemAx=bmitb 6 = 0 l ̈osbar, so hat die
L ̈osungsmenge folgende spezielle Struktur.

Satz 13.6 (Struktur der L ̈osungsmenge). Das LGSAx =bmitA∈Km,nundb∈
Knhabe eine L ̈osungxP ∈Kn. Diese spezielle L ̈osung wird auch partikul ̈are L ̈osung
genannt. Dann gilt

```
L(A,b) ={x∈Kn|Ax=b}={xP+x∈Kn|Ax= 0}=:xP+L(A,0).
```

Beweis. Um zu zeigen, dass die beiden Mengen gleich sind, zeigen wir, dass sie gegenseitig
Teilmengen sind.

##### ”

```
⊆“ Seix∈L(A,b), alsoAx=b. Dann k ̈onnen wir schreibenx=xP+x−xP, und es
giltA(x−xP) =Ax−AxP=b−b= 0, d.h.x−xP∈L(A,0).
```
##### ”

```
⊇“ SeixP+x∈KnmitAx= 0. Dann giltA(xP+x) =AxP+Ax=b+ 0 =b, d.h.
xP+x∈L(A,b).
```
Beispiel 13.7.SeiAx=bmit

##### A=

##### 

##### 

##### 1 2 0 3

##### 0 0 1 4

##### 0 0 0 0

##### 

```
∈R^3 ,^4 , b=
```
##### 

##### 

##### 1

##### 3

##### 0

##### 

##### ∈R^3.

Die MatrixAist bereits in normierter Zeilenstufenform. Die Variablenx 2 ,x 4 sind frei
w ̈ahlbar, wir schreibenx 4 =t∈Rundx 2 = s∈R. Dann sindx 3 + 4x 4 = 3, also
x 3 = 3− 4 t, undx 1 + 2x 2 + 3x 4 = 1, alsox 1 = 1− 2 s− 3 t. Daher ist

```
L(A,b) =
```
##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

```
1 − 2 s− 3 t
s
3 − 4 t
t
```
##### 

##### 

##### 

##### ∣

##### ∣∣

##### ∣∣

##### ∣∣

##### ∣

```
s,t∈R
```
##### 

##### 

##### 

##### 

##### 

##### =

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 1

##### 0

##### 3

##### 0

##### 

##### 

```
+s
```
##### 

##### 

##### 

##### − 2

##### 1

##### 0

##### 0

##### 

##### 

```
+t
```
##### 

##### 

##### 

##### − 3

##### 0

##### − 4

##### 1

##### 

##### 

##### 

##### ∣

##### ∣∣

##### ∣∣

##### ∣∣

##### ∣

```
s,t∈R
```
##### 

##### 

##### 

##### 

##### 

##### =

##### 

##### 

##### 

##### 1

##### 0

##### 3

##### 0

##### 

##### 

##### 

##### ︸︷︷︸

```
=xP
```
##### +

##### 

##### 

##### 

##### 

##### 

```
s
```
##### 

##### 

##### 

##### − 2

##### 1

##### 0

##### 0

##### 

##### 

```
+t
```
##### 

##### 

##### 

##### − 3

##### 0

##### − 4

##### 1

##### 

##### 

##### 

##### ∣

##### ∣∣

##### ∣

##### ∣∣

##### ∣∣

```
s,t∈R
```
##### 

##### 

##### 

##### 

##### 

##### ︸ ︷︷ ︸

```
=span
```
```



```
```
[− 2
1
0
0
```
```
]
,
```
```


```
```
− 3
0
− 4
1
```
```


```
```


=L(A,0)
```
##### .

Eine partikul ̈are L ̈osung erhalten wir, indem wir spezielle Werte f ̈ursundtw ̈ahlen. Am
Einfachsten ist es mits=t= 0, wie in der Rechnung eben.



## Vorlesung 14

## 14 Weitere Anwendungen des Gauß-Algorithmus

# Gauß-Algorithmus

In dieser Vorlesung lernen wir weitere Anwendungen des Gauß-Algorithmus kennen.

### 14.1 Der Rang einer Matrix

In Vorlesung 13 haben wir gesehen, wie wir die L ̈osung eines linearen Gleichungssystems
Ax=bmithilfe des Gauß-Algorithmus berechnen k ̈onnen. In Beispiel 13.5 gab es drei
M ̈oglichkeiten:
1) das LGS hat keine L ̈osungen,
2) das LGS hat genau eine L ̈osung,
3) das LGS hat unendlich viele L ̈osungen.
Das liegt an der Zeilenstufenform vonAund

##### [

```
A b
```
##### ]

. Der entscheidende Begriff dabei ist
der Rang der Matrix.

Definition 14.1(Rang).Der Rang vonA∈Km,nist die Anzahl der Zeilen ungleich
Null in einer Zeilenstufenform vonAund wird mit Rang(A) bezeichnet.

Wie groß kann der Rang von A ∈ Km,n sein? Es ist sicher Rang(A) ≥ 0 und
Rang(A) = 0 nur wennA = 0 die Nullmatrix ist. DaAund damit die ZSF vonA
nurmZeilen hat, ist immer Rang(A)≤m. Außerdem ist Rang(A)≤n, denn es kann
h ̈ochstens so viele Stufen wie Spalten geben.

### 14.2 L ̈osbarkeitskriterium f ̈ur lineare Gleichungssysteme

Wir betrachten das LGS
Ax=b

mit A∈Km,nundb∈Km. Wir k ̈onnen das Gleichungssystem l ̈osen, indem wir die
erweiterte Koeffizientenmatrix

##### [

```
A b
```
##### ]

in ZSF bringen und dann die Gleichungen durch
Ruckw ̈ ̈artssubstitution l ̈osen. Wir betrachten dieses L ̈osungsverfahren jetzt genauer.


```
Wir betrachten die erweiterte Koeffizientenmatrix
```
##### [

```
A b
```
##### ]

und bringen den vorderen
Teil (dort wo zu BeginnAsteht) in ZSFC indem wir elementare Zeilenoperationen
verwenden. Diese wenden wir auch in der letzten Spalte (dort wobsteht) an. Das gibt
eine Matrix der Form

##### [

```
C d
```
##### ]

##### =

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

```
0 c 1 ,j 1 ∗ ∗ ∗ ∗ ∗ ∗ ∗ ∗ d 1
0 0 0 c 2 ,j 2 ∗ ∗ ∗ ∗ ∗ ∗ d 2
0 0 0 0 0 c 3 ,j 3 ∗ ∗ ∗ ∗ d 3
..
.
```
##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### . 0 0

##### ...

##### ∗ ∗

##### ..

##### .

```
0 0 0 0 0 0 0 0 cr,jr ∗ dr
0 0 0 0 0 0 0 ... 0 0 dr+1
..
.
```
##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ... ..

##### .

##### ..

##### .

##### ..

##### .

```
0 0 0 0 0 0 0 ... 0 0 dm
```
##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

mit den Pivotelementenc 1 ,j 1 ,c 2 ,j 2 ,...,cr,jr 6 = 0. Es sind

```
Rang(A) =r= Rang(C)
Rang(
```
##### [

```
A b
```
##### ]

```
) = Rang(
```
##### [

```
C d
```
##### ]

```
)≥r= Rang(A).
```
Nun gibt es folgende zwei F ̈alle:

- Fall 1: Mindestens eins derdr+1,...,dmist von Null verschieden. Das ist genau
    dann der Fall wenn Rang(

##### [

```
A b
```
##### ]

```
) =r+ 1>Rang(A) ist. In diesem Fall ist die
entsprechende Gleichung
```
```
0 x 1 + 0x 2 +...+ 0xn=dj 6 = 0,
```
```
was unm ̈oglich ist. In diesem Fall hat das LGS also keine L ̈osung.
```
- Fall 2: Alledr+1=...=dm= 0 sind Null, dann ist

##### [

```
C d
```
##### ]

```
die ZSF von
```
##### [

```
A b
```
##### ]

```
und daher Rang(
```
##### [

```
A b
```
##### ]

```
) =r= Rang(A). In diesem Fall hat das LGS L ̈osungen.
F ̈uri= 1,...,rsind dieci,ji 6 = 0, also lassen sich die zugeh ̈origen Variablenxji
eindeutig bestimmen (wir l ̈osen diei-te Gleichung nachxjiauf), das sind r=
Rang(A) viele, w ̈ahrend die anderenn−r=n−Rang(A) Variablen frei gew ̈ahlt
werden k ̈onnen.
```
```
Beispiel 14.2.Betrachte das LGS mit
```
```
[
C d
```
##### ]

##### =

##### [

##### 2 3 2

##### 0 0 0

##### ]

##### .

```
Hier istc 1 , 16 = 0 (alsoj 1 = 1). Die erste Zeile bedeutet 2x 1 + 3x 2 = 2 und wir
k ̈onnenx 1 eindeutig bestimmen:x 1 = 1−^32 x 2. Die Variablex 2 kann hingegen frei
gew ̈ahlt werden.
```

```
Ist dann Rang(
```
##### [

```
A b
```
##### ]

```
) = Rang(A) = n, so k ̈onnen allenVariablen eindeutig
bestimmt werden, und das LGS hat eine eindeutige L ̈osung.
Ist hingegen Rang(
```
##### [

```
A b
```
##### ]

```
) = Rang(A)< n, so k ̈onnen n−Rang(A)≥1 viele
Variablen frei gew ̈ahlt werden, und das LGS hat unendlich viele L ̈osungen.
```
Das halten wir als Satz fest.

Satz 14.3(L ̈osbarkeitskriterium f ̈ur lineare Gleichungssysteme).SeienA∈Km,nund
b∈Km. Das LGSAx=bhat

```
1) keine L ̈osung, genau dann wennRang(
```
##### [

```
A b
```
##### ]

```
)>Rang(A)ist,
```
```
2) genau eine L ̈osung, genau dann wennRang(
```
##### [

```
A b
```
##### ]

```
) = Rang(A) =n,
```
```
3) unendlich viele L ̈osungen, genau dann wennRang(
```
##### [

```
A b
```
##### ]

```
) = Rang(A)< n.
```
Der Rang von A und [A,b] gibt also Auskunft ̈uber die L ̈osbarkeit des linearen
GleichungssystemsAx=b. Im Falle von unendlich vielen L ̈osungen gibt der Rang zudem
an, wie viele Variablen frei gew ̈ahlt werden k ̈onnen (n ̈amlichn−Rang(A) viele).
Schema zum L ̈osbarkeitskriterium f ̈ur LGS (Satz 14.3):

```
Ax=bmitmGleichungen,nUnbekannten
```
```
Rang(A)<Rang([A,b])
```
```
LGS hat keine L ̈osung
```
```
Rang(A) = Rang([A,b])
```
```
LGS hat L ̈osungen
```
```
Rang(A) = n
```
```
LGS hat genau
eine L ̈osung
```
```
Rang(A) < n
```
```
LGS hat unendlich
viele L ̈osungen
```

### 14.3 Invertierbarkeit von Matrizen

In Definition 12.13 haben wir gesehen, dassA∈Kn,ninvertierbar ist, falls es eine Matrix
A−^1 ∈Kn,ngibt mitA−^1 A=InundAA−^1 =In, wobei eine der beiden Gleichungen
bereits gen ̈ugt.
Wie ̈uberpr ̈uft man, obAinvertierbar ist, und wie findet man die Inverse? Diese
Frage k ̈onnen wir ebenfalls mit dem Gauß-Algorithmus beantworten. Wir m ̈ochten also
die Inverse finden, wenn sie existiert, und schreiben dazuX=A−^1 (gesucht). Dann soll
gelten
AX=In.

Betrachten wir nun die Spalten vonXundIn, also

```
X=
```
##### [

```
x 1 ... xn
```
##### ]

```
und In=
```
##### [

```
e 1 ... en
```
##### ]

##### ,

so gilt
AX=A

##### [

```
x 1 ... xn
```
##### ]

##### =

##### [

```
Ax 1 ... Axn
```
##### ]

##### =

##### [

```
e 1 ... en
```
##### ]

##### ,

und ein Vergleich der Spalten ergibt

```
Ax 1 =e 1 , Ax 2 =e 2 , ..., Axn=en.
```
Das sindnlineare Gleichungssysteme um die Spalten vonX=A−^1 zu bestimmen. Da
wir immer die gleiche KoeffizientenmatrixAhaben, k ̈onnen wir die Gleichungssysteme
gleichzeitig l ̈osen, indem wir alle rechte Seiten an die Koeffizientenmatrix anh ̈angen:

```
[
A e 1 ... en
```
##### ]

##### =

##### [

```
A In
```
##### ]

##### .

Nun bringen wir die erweiterte Koeffizientenmatrix mit elementaren Zeilenoperationen in
NZSF. Das ergibt eine Matrix

##### [

##### C D

##### ]

mitCin normierter Zeilenstufenform. Nun wissen
wir: WennAinvertierbar ist, dann ist die Inverse eindeutig bestimmt, also mussen die ̈
linearen Gleichungssysteme eindeutig l ̈osbar sein. Dies ist genau dann der Fall, wenn
Rang(

##### [

```
A ej
```
##### ]

```
) = Rang(A) =nist. Daraus sehen wir:
```
```
1) Ist Rang(A)< n(d.h.C∈Kn,nhat eine oder mehrere Nullzeilen), so istAnicht
invertierbar. Wir k ̈onnen dann unsere Suche nach einer Inversen einstellen.
```
```
2) Ist hingegen Rang(A) =n, so istC=In. Dann gilt aberD=InX=X=A−^1 ,
d.h.Aist invertierbar und wir haben die Inverse bestimmt.
```
```
Wir haben jetzt mehrere Dinge gelernt.
```
Satz 14.4. F ̈urA∈Kn,ngilt:Aist invertierbar genau dann, wennRang(A) =n.

Damit k ̈onnen wir entscheiden,obeine Matrix invertierbar ist, indem wir eine ZSF
oder die NZSF vonAberechnen. Wollen wir auch die Inverse berechnen, dann gehen wir
wie folgt vor.


Berechnung der Inversen IstA∈Kn,n(quadratisch), so gehen wir wie folgt vor um
A−^1 zu berechnen (fallsAinvertierbar ist).

```
1) Bringe
```
##### [

```
A In
```
##### ]

```
mit elementaren Zeilenoperationen in NZSF
```
##### [

##### C D

##### ]

##### .

```
2) Wenn Rang(A) 6 =nist, istAnicht invertierbar und wir k ̈onnen aufh ̈oren.
```
```
3) Wenn Rang(A) =nist, so istC=InundA−^1 =D.
```
Beispiel 14.5.Wir pr ̈ufen, ob die Matrix

##### A=

##### [

##### 1 1

##### 1 1

##### ]

invertierbar ist und berechnen ggf. die Inverse. Daher bringen wir die Matrix

##### [

##### A I 2

##### ]

auf NZSF:
[
1 1 1 0
1 1 0 1

##### ]

##### →

##### [

##### 1 1 1 0

##### 0 0 −1 1

##### ]

##### →

##### [

##### 1 1 0 1

##### 0 0 −1 1

##### ]

##### →

##### [

##### 1 1 0 1

##### 0 0 1 − 1

##### ]

##### .

DaC=

##### [

##### 1 1

##### 0 0

##### ]

```
nicht die Einheitsmatrix ist, istAnicht invertierbar. Das hatten wir
```
bereits in Beispiel 12.16 durch ausprobieren herausgefunden.

Beispiel 14.6.Wiruberpr ̈ ufen, ob die Matrix ̈

##### A=

##### 

##### 

##### 1 3 1

##### 1 4 3

##### 1 2 0

##### 

##### ∈C^3 ,^3

invertierbar ist und berechnen ggf. die Inverse. Daher bringen wir die Matrix

##### [

##### A I 3

##### ]

auf NZSF:

```
[
A I 3
```
##### ]

##### =

##### 

##### 

##### 1 3 1 1 0 0

##### 1 4 3 0 1 0

##### 1 2 0 0 0 1

##### 

##### →

##### 

##### 

##### 1 3 1 1 0 0

##### 0 1 2 −1 1 0

##### 0 − 1 − 1 −1 0 1

##### 

##### 

##### →

##### 

##### 

##### 1 3 1 1 0 0

##### 0 1 2 −1 1 0

##### 0 0 1 −2 1 1

##### 

##### ,

woraus wir bereits Rang(A) = 3 ablesen, und somit dassAinvertierbar ist. Wir formen
weiter auf NZSF um:

```
[
A I 3
```
##### ]

##### →

##### 

##### 

##### 1 3 0 3 − 1 − 1

##### 0 1 0 3 − 1 − 2

##### 0 0 1 − 2 1 1

##### 

##### →

##### 

##### 

##### 1 0 0 − 6 2 5

##### 0 1 0 3 − 1 − 2

##### 0 0 1 − 2 1 1

##### 

##### .


Daher ist

```
A−^1 =
```
##### 

##### 

##### − 6 2 5

##### 3 − 1 − 2

##### − 2 1 1

##### 

##### .

Zur Probe k ̈onnen wirA−^1 AoderAA−^1 berechnen: kommt nicht die Einheitsmatrix
heraus, so haben wir uns verrechnet.

IstA∈Kn,ninvertierbar mit der InversenA−^1 , so ist das LGSAx=beindeutig
l ̈osbar mit
x=A−^1 b,

vergleiche den Beginn von Vorlesung 13. Das heißt: Wir k ̈onnen das LGS ganz einfach
mit der Inversen l ̈osen, wenn wir diese kennen. Sind wir nur an der L ̈osung vonAx=b
interessiert, ist es aber einfacher das LGS direkt zu l ̈osen.

### 14.4 Unterschiede zwischen Matrizen und Zahlen

In den Vorlesungen 12, 13 und 14 haben wir Matrizen und die L ̈osung linearer Glei-
chungssysteme studiert. Wir sammeln noch ein paar Eigenschaften von Matrizen, die
anders als bei reellen oder komplexen Zahlen sind.

```
1) Im Allgemeinen giltAB 6 =BA, selbst wenn beide Produkte definiert sind.
```
```
2) AusA 6 = 0 folgt nicht, dassAinvertierbar ist. Auch dann nicht, wennAquadratisch
ist.
```
```
Beispiel: Es istA=
```
##### [

##### 1 0

##### 0 0

##### ]

```
6 = 0, aber nicht invertierbar (denn Rang(A) = 1<2).
```
```
3) AusAB= 0 folgt im Allgemeinen nicht dassA= 0 oderB= 0 sein m ̈ussen.
```
```
Beispiel:A=
```
##### [

##### 1 0

##### 0 0

##### ]

```
undB=
```
##### [

##### 0 0

##### 0 1

##### ]

```
sind beide ungleich Null, aberAB= 0.
```
```
4) IstAB= 0 undAinvertierbar, so folgtB=A−^1 AB=A−^1 0 = 0.
IstAB= 0 undBinvertierbar, so folgt genauso, dassA= 0.
```

Vorlesung 15

## 15 Lineare Abbildungen

Wir lernen lineare Abbildungen und ihre Eigenschaften kennen.

### 15.1 Definition und erste Eigenschaften

Definition 15.1(Lineare Abbildung).SeienV,WzweiK-Vektorr ̈aume. Eine Abbildung
f:V →Wheißtlinear, wenn f ̈ur allev,w∈V undλ∈Kgilt
1)f(v+w) =f(v) +f(w),
2)f(λv) =λf(v).
Die Menge aller linearen Abbildungen vonV nachWwird mitL(V,W) oder Hom(V,W)
bezeichnet.

Lineare Abbildungen erhalten die Struktur eines Vektorraums (Plus und Mal). Eine
andere Bezeichnung f ̈ur eine lineare Abbildung istHomomorphismus.

Beispiel 15.2. 1) Die Abbildungf:R^2 →R^2 , [xx^12 ]7→[xx^21 ], ist die Spiegelung an der
ersten Winkelhalbierenden in der Ebene. Diese ist linear, denn fur alle ̈ v= [xx^12 ],
w= [yy^12 ]∈R^2 undλ∈Rgilt

```
f(v+w) =f
```
```
([x 1 +y 1
x 2 +y 2
```
##### ])

##### =

```
[x 2 +y 2
x 1 +y 1
```
##### ]

```
= [xx^21 ] + [yy^21 ] =f(v) +f(w)
f(λv) =f
```
##### ([

```
λx 1
λx 2
```
##### ])

##### =

##### [

```
λx 2
λx 1
```
##### ]

```
=λ[xx^21 ] =λf(v).
```
```
2) Die Abbildungf:C^3 →C^2 ,^2 ,
```
```
[x 1
xx 2
3
```
##### ]

```
7→[xx^11 xx^23 ], ist linear, denn: F ̈ur allev=
```
```
[x 1
xx 2
3
```
##### ]

```
undw=
```
```
[y 1
y 2
y 3
```
##### ]

```
∈C^3 undλ∈Cgilt:
```
```
f(v+w) =f
```
```
([x
x^1 +y^1
2 +y 2
x 3 +y 3
```
##### ])

##### =

```
[x 1 +y 1 x 2 +y 2
x 1 +y 1 x 3 +y 3
```
##### ]

```
= [xx^11 xx^23 ] + [yy^11 yy^23 ] =f(v) +f(w),
```
```
f(λv) =f
```
##### ([

```
λx 1
λx 2
λx 3
```
##### ])

##### =

##### [

```
λx 1 λx 2
λx 1 λx 3
```
##### ]

```
=λ[xx^11 xx^23 ] =λf(v).
```
```
3) IstV ein Vektorraum, so ist die Identit ̈at idV :V →V aufV (vgl. Beispiel 5.2)
linear, denn idV(v+w) =v+w= idV(v) + idV(w) und idV(λv) =λv=λidV(v).
```

```
4) Die Abbildungf:R→R,x7→ax, ist linear, aber die Abbildung mitf(x) = 2x+1
ist nicht linear. (Warum nicht?) Insbesondere sind Linearfaktoren von Polynomen
(z−z 0 ) trotz ihres Namens nur f ̈urz 0 = 0 eine lineare Abbildung.
5) Die Abbildungf:R→R,x7→x^2 , ist nicht linear, dennf(x+y) = (x+y)^2 =
x^2 + 2xy+y^26 =x^2 +y^2 (außer im Fall dasx= 0 odery= 0.)
```
Im Laufe dieser Vorlesung werden wir noch viele weitere lineare Abbildungen kennen
lernen.

Beispiel 15.3.SeiA∈Km,n. Dann istf:Kn→Km,x7→Ax, linear, denn nach den
Rechenregeln f ̈ur Matrizen gilt f ̈ur allex,y∈Knundλ∈K:

```
f(x+y) =A(x+y) =Ax+Ay=f(x) +f(y), f(λx) =A(λx) =λAx=λf(x).
```
Beispiel 15.3 zeigt, dass wir jede Matrix als lineare Abbildung ansehen k ̈onnen. Ins-
besondere gilt alles, was wir f ̈ur lineare Abbildungen lernen, auch fur Matrizen. In ̈ Kn
hat jede lineare Abbildung diese Gestalt.

Satz 15.4. Istf :Kn→Km linear, dann gibt es eine MatrixA∈Km,nmitf(x) =
Ax[ f ̈ur allex ∈ Kn. Dabei enth ̈alt A = [a 1 ,...,an]die Bilder der Vektoren e 1 =

(^10)
..
.
0

##### ]

```
,...,en=
```
##### [ 0

##### ..

##### .

```
0
1
```
##### ]

```
, d.h.a 1 =f(e 1 ),... ,an=f(en).
```
Beweis. Es sindf(e 1 ),...,f(en) Spaltenvektoren ausKm, d.h.Aist einem×n-Matrix.
Mit dieser ist

```
f(x) =f
```
##### 

##### 

```
∑n
```
```
j=1
```
```
xjej
```
##### 

##### =

```
∑n
```
```
j=1
```
```
xjf(ej) =
```
```
∑n
```
```
j=1
```
```
xjaj=Ax f ̈ur allex=
```
##### 

##### 

##### 

```
x 1
..
.
xn
```
##### 

##### 

```
∈Kn.
```
In Vorlesung 16 werden wir sehen, dass sich jede lineare Abbildung durch eine Matrix
darstellen l ̈asst, fallsV undWbeide endlichdimensional sind.
Versehen wir die Menge der linearen Abbildungen vonV nachWmit der punktweisen
Addition und Multiplikation, so wirdL(V,W) selbst auch wieder zu einem Vektorraum.
Das ist analog zum VektorraumKm,nderm×n-Matrizen.

Satz 15.5.SeienV,W zweiK-Vektorr ̈aume. F ̈urf,g∈L(V,W)undλ∈Kdefinieren
wir

```
f+g:V →W, v7→(f+g)(v):=f(v) +g(v),
λf:V →W, v7→(λf)(v):=λf(v).
```
Dann sindf+gundλfwieder linear undL(V,W)ist selbst wieder einK-Vektorraum.
IstV =W, so enth ̈altL(V,V)insbesondere die Identit ̈atidV.


Beweis. Wir rechnen nur nach, dassf+glinear ist. F ̈ur allev,w∈V undλ∈Kgilt
n ̈amlich

```
(f+g)(v+w) =f(v+w) +g(v+w) =f(v) +f(w) +g(v) +g(w)
=f(v) +g(v) +f(w) +g(w) = (f+g)(v) + (f+g)(w),
(f+g)(λv) =f(λv) +g(λv) =λf(v) +λg(v) =λ(f(v) +g(v)) =λ(f+g)(v).
```
Wo wurde die Definition von + verwendet, und wo die Linearit ̈at vonfundg? Rechnen
Sie zurUbung nach, dass auch ̈ λfwieder linear ist.
Die Null inL(V,W) ist die Nullabbildung, die jeden Vektor ausvauf 0∈Wabbildet.
F ̈urf ∈L(V,W) ist−f durch (−f)(v) =−f(v) gegeben. Alle anderen Rechenregeln
lassen sich leicht nachrechnen, wir verzichten hier darauf.

Wir sammeln einige Eigenschaften von linearen Abbildungen. Linearit ̈at bleibt bei
Komposition (Abschnitt 5.2) und beim Umkehren der Funktion (Abschnitt 5.3) erhalten.

Satz 15.6. SeienV,W,XK-Vektorr ̈aume.
1) Istf:V→W linear, so giltf(0) = 0.
2) Sindf:V →W undg:W→Xlinear, so ist auch die Kompositiong◦f:V →X
linear.
3) Istf:V→W linear und bijektiv, so ist auch die Umkehrabbildungf−^1 :W→V
linear.
Bei linearen Abbildungen wird die Umkehrabbildung ̈ofter auch dieInversegenannt
(wie bei quadratischen Matrizen). Bijektive lineare Abbildungen heißenIsomorphismus.

Beweis. 1) Es istf(0) =f(0 + 0) =f(0) +f(0). Addieren wir nun−f(0) erhalten wir 0 =f(0), wie

behauptet. Alternativ istf(0V) =f(0K· (^0) V) = 0K·f(0V) = 0W.
2) F ̈ur allev,w∈V undλ∈Kgilt
(g◦f)(v+w) =g(f(v+w)) =g(f(v) +f(w)) =g(f(v)) +g(f(w))
= (g◦f)(v) + (g◦f)(w),
(g◦f)(λv) =g(f(λv)) =g(λf(v)) =λg(f(v)).
3) Dafbijektiv ist, existiert die Umkehrabbildungf−^1 :W→V, und wir brauchen nur nachzu-
rechnen, dassf−^1 linear ist. Seienv,w∈Wundλ∈K. Daf−^1 die Umkehrabbildung zufist,
giltf(f−^1 (v)) =vundf(f−^1 (w)) =w(siehe Definition 5.9). Dann ist
v+w=f(f−^1 (v)) +f(f−^1 (w)) =f(f−^1 (v) +f−^1 (w)),
alsof−^1 (v+w) =f−^1 (v) +f−^1 (w) nach Anwenden vonf−^1. Genauso rechnet man
f−^1 (λv) =f−^1 (λf(f−^1 (v))) =f−^1 (f(λf−^1 (v))) =λf−^1 (v).
Daher istf−^1 linear.
IstV =W undf :V →V, so kannf◦fgebildet werden, was wieder eine lineare
Abbildung ist. Fur ̈ f◦fschreiben wir dannf^2 , und definieren ganz allgemein
fn:=f◦...◦f
︸ ︷︷ ︸
nMal
, n≥ 1.


Fur ̈ n= 0 setzen wirf^0 := idV. Istfinvertierbar, so definieren wir

```
f−n:= (f−^1 )n=f−^1 ◦...◦f−^1
︸ ︷︷ ︸
nMal
```
Beachten Sie:f^2 (v) =f(f(v)) 6 = (f(v))^2. Der letzte Ausdruck ist im Allgemeinen nicht
einmal definiert.
Schließlich sammeln wir Rechenregeln f ̈ur lineare Abbildungen. Diese sind genauso
wie die Rechenregeln fur die Matrizenmultiplikation (Satz 12.12), wobei die Komposition ̈
der Matrizenmultiplikation entspricht.

Satz 15.7. F ̈ur lineare Abbildungen f,g,hundλ ∈ Kgilt (falls die Verkn ̈upfungen
definiert sind)

```
1)f◦(g◦h) = (f◦g)◦h,
2)f◦(g+h) = (f◦g) + (f◦h),
3)(f+g)◦h= (f◦h) + (g◦h),
4)α(f◦g) = (αf)◦g=f◦(αg),
5)id◦f=f=f◦id.
```
### 15.2 Kern und Bild

Der Kern und das Bild sind wichtige Kenngr ̈oßen von linearen Abbildungen. Mit diesen
l ̈asst sich zum Beispiel einfach charakterisieren, ob eine lineare Abbildung injektiv oder
surjektiv ist. Das Bild kennen wir schon (Vorlesung 5), wir erinnern dennoch daran.

Definition 15.8(Kern und Bild).Seif:V →W linear.
1) DerKernvonfist das Urbild von 0:
Kern(f) ={v∈V |f(v) = 0}=f−^1 ({ 0 }).
2) DasBildvonfist die Menge
Bild(f) =f(V) ={f(v)|v∈V}.

```
V
```
```
Kern(f)⊆V
```
```
Kern(f)
```
##### W

```
Bild(f)⊆W
```
```
Bild(f)
```
```
f
```

Satz 15.9. Seif:V →Wlinear.
1) Kern(f)ist ein Teilraum vonV.
2) Bild(f)ist ein Teilraum vonW.

Beweis. Wir rechnen nur nach, dass Kern(f) ein Teilraum vonVist. Zun ̈achst istf(0) =
0, daf linear ist, also 0∈Kern(f). Sindv,w∈Kern(f), so folgtf(v+w) =f(v) +
f(w) = 0 + 0 = 0, also ist v+w ∈Kern(f). Sindv ∈Kern(f) undλ ∈K, so ist
f(λv) =λf(v) =λ·0 = 0, alsoλv∈Kern(f). Mit dem Teilraumkriterium (Satz 10.4)
ist Kern(f) ein Teilraum vonV.

Wir werden insbesondere an der Dimension von Kern und Bild interessiert sein. F ̈ur
Matrizen lassen sich diese leicht mit dem Gaußalgorithmus berechnen.

Basis des Kerns. SeiA∈Km,n. Dann ist Kern(A) ={x∈Kn|Ax= 0}=L(A,0)
die L ̈osungsmenge des homogenen linearen Gleichungssystems

```
Ax= 0.
```
Dieses l ̈osen wir mit dem Gaußalgorithmus und bringenAin (normierte) Zeilenstufen-
form. Gilt Rang(A) =r, so gibt esn−rfrei w ̈ahlbare Variablen. Eine Basis des Kerns
erhalten wir, indem wir jeweils eine der frei w ̈ahlbaren Variablen = 1 und die anderen = 0
setzen: Die L ̈osung vonAx= 0 mit erster frei w ̈ahlbarer Variablen = 1 und den anderen
= 0 gibt den ersten Basisvektor von Kern(A). Die L ̈osung mit zweiter frei w ̈ahlbarer
Variablen = 1 und den anderen = 0 gibt den zweiten Basisvektor von Kern(A), usw.
Das ergibt eine Basis von Kern(A) mitn−rBasisvektoren. Insbesondere ist

```
dim(Kern(A)) =n−r=n−Rang(A).
```
Basis des Bildes. SeiA∈Km,n. Wir bezeichnen die Spalten vonAmita 1 ,...,an∈
Km, d.h. es istA=

##### [

```
a 1 ... an
```
##### ]

. F ̈urx∈Knist

```
Ax=
```
##### [

```
a 1 ... an
```
##### ]

##### 

##### 

##### 

```
x 1
..
.
xn
```
##### 

##### 

```
=x 1 a 1 +...+xnan=
```
```
∑n
```
```
j=1
```
```
xjaj,
```
also die Linearkombination von den Spalten von Amit den Koeffizientenx 1 ,...,xn.
Daher ist
Bild(A) ={Ax|x∈Kn}= span{a 1 ,...,an}.

Die Spalten vonAbilden also ein Erzeugendensystem von Bild(A). Welche Spalten bilden
dann eine Basis? Die Idee ist: Spalten vonAsind linear unabh ̈angig genau dann, wenn
die entsprechenden Spalten in der Zeilenstufenform linear unabh ̈angig sind. Daher kann
man dierSpalten vonA, die zu den Stufenpositionen in der ZSF geh ̈oren, als Basis von
Bild(A) w ̈ahlen. Insbesondere ist

```
dim(Bild(A)) = Rang(A).
```

Beispiel 15.10.Wir berechnen eine Basis von Kern(A) und Bild(A) fur ̈ A=

##### [1 3− 1

```
2 7− 3
```
##### ]

##### ∈

R^2 ,^3. Dazu bringen wirAin normierte Zeilenstufenform:
[
1 3 − 1
2 7 − 3

##### ]

##### →

##### [

##### 1 3 − 1

##### 0 1 − 1

##### ]

##### →

##### [

##### 1 0 2

##### 0 1 − 1

##### ]

Da Rang(A) = 2 sind dim(Bild(A)) = 2 und dim(Kern(A)) = 1. Die erste und zweite
Spalte geh ̈oren zu den Pivotpositionen in der ZSF, also ist{[^12 ],[^37 ]}eine Basis von
Bild(A). Um eine Basis von Kern(A) zu bestimmen setzen wirx 3 = 1, und berechnen

x 1 =−2 undx 2 = 1. Daher ist

##### {[− 2

(^11)

##### ]}

```
eine Basis von Kern(A).
```
### 15.3 Dimensionsformel und Konsequenzen

Fur ̈ A∈Km,nerhalten wir

```
dim(Kern(A)) + dim(Bild(A)) =n−Rang(A) + Rang(A) =n= dim(Kn).
```
Das ist dieDimensionsformelf ̈ur Matrizen, die einen Zusammenhang zwischen der Di-
mension des Kerns und der Dimension des Bildes herstellt. Fur lineare Abbildungen gilt ̈
das entsprechend.

Satz 15.11(Dimensionsformel f ̈ur lineare Abbildungen). Seif:V →W linear undV
endlichdimensional. Dann gilt

```
dim(V) = dim(Kern(f)) + dim(Bild(f)).
```
Die folgende Skizze veranschaulicht die Dimensionsformel:V ist aufgeteilt in den
Kern vonf (diese Vektoren werden auf 0 ∈ W abgebildet), und den Rest, der auf
Bild(f) abgebildet wird.

```
Kern(f)
```
##### V

##### W

```
Bild(f)
```
Aus der Dimensionsformel fur lineare Abbildungen erhalten wir die folgende Cha- ̈
rakterisierung, wann lineare Abbildungen injektiv oder surjektiv sind.

Satz 15.12.Seif:V →Wlinear. Dann gilt:

```
1)f ist injektiv⇔Kern(f) ={ 0 } ⇔dim(Kern(f)) = 0.
```

```
2) fist surjektiv⇔Bild(f) =W ⇔dim(Bild(f)) = dim(W).
Dabei gilt die letzteAquivalenz nur falls ̈ dim(W)<∞.
```
```
3) Wenndim(V) = dim(W)<∞, so gilt: f ist bijektiv⇔f ist injektiv⇔f ist
surjektiv.
```
Beweis. 1) ”⇒“ Seif injektiv. Daflinear ist, giltf(0) = 0, also{ 0 } ⊆Kern(f). Ist dannv∈
Kern(f), so giltf(v) = 0 =f(0). Daf injektiv ist, folgtv= 0, also Kern(f)⊆ { 0 }.
Zusammen ist Kern(f) ={ 0 }und dann dim(Kern(f)) = 0.
”⇐“ Sei dim(Kern(mitf(v) =f(fw)) = 0, dann ist Kern(), so folgt 0 =f(v)−f) =f(w{) =^0 }. Wir zeigen, dassf(v−w), alsovf−injektiv ist. Sindw∈Kern(f) ={v,w 0 }, also∈V
v−w= 0 und dannv=w. Also istfinjektiv.
2) Die ersteAquivalenz ist die Definition von surjektiv. Wenn Bild( ̈ f) =Wist, dann haben beide
R ̈aume die gleiche Dimension. Sei andersherum dim(Bild(f)) = dim(W). Da Bild(f)⊆Wbeide
R ̈aume die gleiche Dimension haben, so sind sie gleich.
3) Zun ̈achst gilt:

```
finjektiv⇔1)dim(Kern(f)) = 0Dimensionsformel⇔ dim(V) = dim(Bild(f))
⇔2)fsurjektiv.
Ist nunfbijektiv, so istfinjektiv und surjektiv (per Definition). Ist andersherumfinjektiv oder
surjektiv, so istfinjektivundsurjektiv, wie wir eben gezeigt haben, und dann bijektiv.
```


## Vorlesung 16

## 16 Koordinaten und Matrixdarstellung

# Matrixdarstellung

Ziel dieser Vorlesung ist es, Vektoren und lineare Abbildungen durch Spaltenvektoren
und Matrizen darzustellen, da man mit Matrizen sehr gut rechnen kann. Dies wird zum
Beispiel in der”Numerik II fur Ingenieurwissenschaften“ bei der numerischen L ̈ ̈osung
von partiellen Differentialgleichungen verwendet.

### 16.1 Koordinaten

In Vorlesung 11 haben wir bereits gesehen, wie einem Vektor sein Koordinatenvek-
tor zugeordnet werden kann. IstV ein endlichdimensionalerK-Vektorraum mit Basis
B={b 1 ,...,bn}, so l ̈asst sich jeder Vektorv∈V als eindeutige Linearkombination der
Basisvektoren schreiben:

```
v=
```
```
∑n
```
```
j=1
```
```
λjbj=λ 1 b 1 +λ 2 b 2 +...+λnbn.
```
Dann ist der Koordinatenvektor vonv∈V bzgl. der BasisBder Spaltenvektor

```
~vB=
```
##### 

##### 

##### 

##### 

##### 

```
λ 1
λ 2
..
.
λn
```
##### 

##### 

##### 

##### 

##### 

```
∈Kn.
```
Der Ubergang von ̈ vzum Koordinatenvektor~vBist besonders wichtig und hat einen
eigenen Namen.


Definition 16.1 (Koordinatenabbildung). SeiV ein K-Vektorraum mit Basis B =
{b 1 ,...,bn}. DieKoordinatenabbildungvonV bzgl.Bist die Abbildung

```
KB:V →Kn, v=
```
```
∑n
```
```
j=1
```
```
λjbj7→~vB=
```
##### 

##### 

##### 

##### 

```
λ 1
λ 2
..
.
λn
```
##### 

##### 

##### 

##### 

##### .

Beispiel 16.2. SeiV = C[z]≤ 2 = {a 0 +a 1 z+a 2 z^2 |a 0 ,a 1 ,a 2 ∈C} mit der Basis
B={b 1 ,b 2 ,b 3 }={ 1 ,z,z^2 }. Fur ̈ p(z) =a 0 +a 1 z+a 2 z^2 ∈V istp(z) =a 0 ·1+a 1 ·z+a 2 ·z^2 ,
d.h. der Koordinatenvektor vonpbzgl.Bist

```
KB(p) =~pB=
```
##### 

##### 

```
a 0
a 1
a 2
```
##### 

##### .

Daher ist die Koordinatenabbildung

```
KB:V→C^3 , KB(a 0 +a 1 z+a 2 z^2 ) =
```
##### 

##### 

```
a 0
a 1
a 2
```
##### 

##### .

Satz 16.3.SeiV einK-Vektorraum mit BasisB={b 1 ,...,bn}. Die Koordinatenabbil-
dungKBist linear und bijektiv, also ein Isomorphismus. Die Inverse ist

```
KB−^1 :Kn→V,
```
##### 

##### 

##### 

```
λ 1
..
.
λn
```
##### 

##### 

```
7→v=
```
```
∑n
```
```
j=1
```
```
λjbj=λ 1 b 1 +...+λnbn.
```
Beweis. Wir rechnen die Linearit ̈at nach: F ̈urv=

```
∑n
j=1λjbj∈V undw=
```
∑n
j=1μjbj∈
V istv+w=

```
∑n
j=1(λj+μj)bj, also
```
```
KB(v+w) =
```
##### 

##### 

##### 

```
λ 1 +μ 1
..
.
λn+μn
```
##### 

##### 

##### =

##### 

##### 

##### 

```
λ 1
..
.
λn
```
##### 

##### 

##### +

##### 

##### 

##### 

```
μ 1
..
.
μn
```
##### 

##### 

```
=KB(v) +KB(w).
```
Weiter ist f ̈urλ∈Kauchλv=

```
∑n
j=1(λλj)bj, also
```
```
KB(λv) =
```
##### 

##### 

##### 

```
λλ 1
..
.
λλn
```
##### 

##### 

```
=λ
```
##### 

##### 

##### 

```
λ 1
..
.
λn
```
##### 

##### 

```
=λKB(v).
```
Daher istKBlinear. Mit der angegeben form vonKB−^1 rechnet man direkt nach, dass
KB−^1 (KB(v)) =vf ̈ur allev∈V undKB(KB−^1 (~x)) =~xfur alle ̈ ~x∈Kn. Daher istKB
bijektiv mit InversenKB−^1.


### 16.2 Matrixdarstellung

Unser n ̈achstes Ziel ist es, eine lineare Abbildungf:V →W durch eine Matrix darzu-
stellen. Seien dazu

```
1)V endlichdimensional mit BasisB={b 1 ,...,bn},
2)Wendlichdimensional mit BasisC={c 1 ,...,cm},
3)f:V →W linear.
```
F ̈urv=λ 1 b 1 +λ 2 b 2 +...+λnbn∈V gilt, daflinear ist,

```
f(v) =f(λ 1 b 1 ) +f(λ 2 b 2 ) +...+f(λnbn) =λ 1 f(b 1 ) +λ 2 f(b 2 ) +...+λnf(bn).
```
Dies sind Elemente des VektorraumsW. Betrachten wir die Koordinaten bzgl. der Basis
C, d.h. wenden wir die KoordinatenabbildungKCan, so folgt

```
KC(f(v)) =KC(λ 1 f(b 1 ) +λ 2 f(b 2 ) +...+λnf(bn))
=λ 1 KC(f(b 1 )) +λ 2 KC(f(b 2 )) +...+λnKC(f(bn))
```
##### =

##### [

```
KC(f(b 1 )) KC(f(b 2 )) ... KC(f(bn))
```
##### ]

##### ︸ ︷︷ ︸

```
∈Km,n
```
##### 

##### 

##### 

```
λ 1
..
.
λn
```
##### 

##### 

##### 

##### ︸︷︷︸

```
=KB(v)∈Kn
```
##### .

##### (16.1)

Definition 16.4(Darstellende Matrix).Die Matrix

```
fB,C:=
```
##### [

```
KC(f(b 1 )) KC(f(b 2 )) ... KC(f(bn))
```
##### ]

```
∈Km,n
```
heißt diedarstellende Matrixvonf bzgl.BundC, oder auchMatrixdarstellungvonf
bzgl.BundC.

Die erste Spalte der darstellenden Matrix enth ̈alt also den Koordinatenvektor von
f(b 1 ), die zweite Spalte den Koordinatenvektor vonf(b 2 ), usw. Hat man die Koor-
dinatenabbildungeKC nicht zur Hand, k ̈onnen wir die Eintr ̈age der Matrix wie folgt
berechnen: Wir stellenf(bj) in der BasisCdar,

```
f(b 1 ) =a 1 , 1 c 1 +a 2 , 1 c 2 +...+am, 1 cm,
f(b 2 ) =a 1 , 2 c 1 +a 2 , 2 c 2 +...+am, 2 cm,
..
.
f(bn) =a 1 ,nc 1 +a 2 ,nc 2 +...+am,ncm,
```
und erhalten

```
fB,C=
```
##### [

```
KC(f(b 1 )) KC(f(b 2 )) ... KC(f(bn))
```
##### ]

##### =

##### 

##### 

##### 

##### 

```
a 1 , 1 a 1 , 2 ... a 1 ,n
a 2 , 1 a 2 , 2 ... a 2 ,n
..
.
```
##### ..

##### .

##### ... ..

##### .

```
am, 1 am, 2 ... am,n
```
##### 

##### 

##### 

##### 

```
∈Km,n.
```

Beispiel 16.5. 1) Seif:R^2 →R^3 , [xx^12 ]7→

```
[ x 1
2 x 2 +x 1
x 1 −x 2
```
##### ]

. Wir betrachten die Standard-
basen

```
B={b 1 ,b 2 }=
```
##### {[

##### 1

##### 0

##### ]

##### ,

##### [

##### 0

##### 1

##### ]}

```
und C={c 1 ,c 2 ,c 3 }=
```
##### 

##### 

##### 

##### 

##### 

##### 1

##### 0

##### 0

##### 

##### ,

##### 

##### 

##### 0

##### 1

##### 0

##### 

##### ,

##### 

##### 

##### 0

##### 0

##### 1

##### 

##### 

##### 

##### 

##### 

```
vonR^2 undR^3. Dann sind
```
```
f(b 1 ) =f([^10 ]) =
```
##### [ 1

```
1
1
```
##### ]

```
= 1c 1 + 1c 2 + 1c 3 ,
```
```
f(b 2 ) =f([^01 ]) =
```
##### [ 0

```
2
− 1
```
##### ]

```
= 0c 1 + 2c 2 − 1 c 3 ,
```
```
und die darstellende Matrix vonfbzgl.BundCist
```
```
fB,C=
```
##### 

##### 

##### 1 0

##### 1 2

##### 1 − 1

##### 

##### ∈R^3 ,^2.

```
2) SeiV =C[z]≤ 2 ={a 0 +a 1 z+a 2 z^2 |a 0 ,a 1 ,a 2 ∈C}mit der BasisB={b 1 ,b 2 ,b 3 }=
{ 1 ,z,z^2 }, und seif:V →V mit
```
```
f(a 0 +a 1 z+a 2 z^2 ) = (a 0 +a 2 ) + (a 1 −a 0 )z+ (2a 1 +a 2 )z^2.
```
```
Hier istW=V und wir nehmenC=B. Dann sind
f(b 1 ) =f(1) =f(1·1 + 0z+ 0z^2 ) = 1· 1 − 1 z+ 0z^2 = 1b 1 + (−1)b 2 + 0b 3 ,
f(b 2 ) =f(z) =f(0·1 + 1z+ 0z^2 ) = 0·1 + 1z+ 2z^2 = 0b 1 + 1b 2 + 2b 3 ,
f(b 3 ) =f(z^2 ) =f(0·1 + 0z+ 1z^2 ) = 1·1 + 0z+ 1z^2 = 1b 1 + 0b 2 + 1b 3.
```
```
Daher ist die darstellende Matrix vonfbzgl.BundB
```
```
fB,B=
```
##### 

##### 

##### 1 0 1

##### −1 1 0

##### 0 2 1

##### 

##### .

Die Rechnung (16.1) zeigt, dass die darstellende Matrix auf Ebene der Koordina-
tenvektoren genau das gleiche tut, wie die lineare Abbildung in den”abstrakten“ Vek-
torr ̈aumenV undW. Das halten wir noch einmal als Satz fest.

Satz 16.6. Seif :V →W linear, wobeiV,W endlichdimensionale Vektorr ̈aume mit
BasenBundCsind. Dann gilt f ̈ur jeden Vektorv∈V

```
KC(f(v)) =fB,CKB(v),
```
alsoKC◦f=fB,CKB, d.h.
fB,C=KC◦f◦KB−^1. (16.2)


```
Diesen Sachverhalt kann man sich wie folgt veranschaulichen:
```
```
V W allg. Vektor und lineare Abbildung
```
```
Kn Km Spaltenvektor und Matrix
```
##### KB

```
f
```
```
fB,C
```
##### KC

### 16.3 Basiswechsel

Die Koordinatenvektoren h ̈angen von der gew ̈ahlten Basis ab. Wir lernen, wie die Koordi-
natenvektoren bzgl. verschiedener Basen zusammenh ̈angen, und wie sich ein Basiswechsel
auf die darstellenden Matrizen auswirkt.

Satz 16.7(Koordinatenvektor bei Basiswechsel).SeiV ein endlichdimensionaler Vek-
torraum mit BasisB 1 und BasisB 2. Dann gilt f ̈ur allev∈V:

```
~vB 2 = idB 1 ,B 2 ~vB 1.
```
D.h., bei Basiswechsel wird der Koordinatenvektor mit der Matrixdarstellung der Iden-
tit ̈at multipliziert.idB 1 ,B 2 heißt auchBasiswechselmatrixoderBasis ̈ubergangsmatrix.

Beweis. Nach Satz 16.6 giltKB 2 (v) =KB 2 (id(v)) = idB 1 ,B 2 KB 1 (v).

Beispiel 16.8. SeiV =R^2 mit der StandardbasisB 1 ={[^10 ],[^01 ]}und der BasisB 2 =
{[^10 ],[^11 ]}; vergleiche Beispiel 11.11. Dann sind

```
KB 1
```
##### ([

```
x 1
x 2
```
##### ])

##### =

##### [

```
x 1
x 2
```
##### ]

```
und KB 2
```
##### ([

```
x 1
x 2
```
##### ])

##### =

##### [

```
x 1 −x 2
x 2
```
##### ]

##### .

Die Matrix idB 1 ,B 2 berechnen wir wie folgt:

```
idB 1 ,B 2 =
```
##### [

```
KB 2 (id([^10 ])) KB 2 (id([^01 ]))
```
##### ]

##### =

##### [

##### KB 2 ([^10 ]) KB 2 ([^01 ])

##### ]

##### =

##### [

##### 1 − 1

##### 0 1

##### ]

##### .

Damit haben wir, wie es der Satz sagt:

```
idB 1 ,B 2 KB 1
```
##### ([

```
x 1
x 2
```
##### ])

##### =

##### [

##### 1 − 1

##### 0 1

##### ][

```
x 1
x 2
```
##### ]

##### =

##### [

```
x 1 −x 2
x 2
```
##### ]

##### =KB 2

##### ([

```
x 1
x 2
```
##### ])

##### .

Als letztes betrachten wir, wie sich die darstellende Matrix ̈andert, wenn wir die
Basen wechseln.

Satz 16.9(Darstellende Matrix bei Basiswechsel).Seif:V →Wlinear,B 1 ,B 2 Basen
vonV undC 1 ,C 2 Basen vonW. Dann gilt

```
fB 2 ,C 2 = idC 1 ,C 2 fB 1 ,C 1 idB 2 ,B 1 ,
```
d.h., wir erhalten die neue Matrixdarstellung aus der alten, indem wir mit den entspre-
chenden Basiswechselmatrizen multiplizieren.


Beweis. Die Formel rechnen wir mit (16.2) nach:

```
fB 2 ,C 2 =KC 2 ◦f◦K−B 21 =KC 2 ◦id◦f◦id◦KB− 21
=KC 2 ◦id◦KC− 11 ◦KC 1 ◦f◦KB− 11 ◦KB 1 ◦id◦KB− 21
= idC 1 ,C 2 fB 1 ,C 1 idB 2 ,B 1.
```
Beim dritten Gleichheitszeichen haben wirKB−^1 ◦KB= id verwendet, und beim ersten und letzten
Gleichheitszeichen mehrfach die Formel (16.2).

Die Formel k ̈onnen wir wie oben veranschaulichen. Dazu nehmen zwei Kopien des
obigen Diagramms (eine f ̈urB 1 undC 1 und eine fur ̈ B 2 undC 2 ) und heften diese zu-
sammen. Anschließend f ̈ugen wir noch die Basiswechsel ein (zwischenB 2 undB 1 und
zwischenC 1 undC 2 ) und erhalten folgendes Diagramm:

```
Kn Km
```
##### V W

```
Kn Km
```
```
idB 2 ,B 1
```
```
fB 2 ,C 2
```
```
f
```
##### KB 2

##### KB 1

##### KC 2

##### KC 1

```
fB 1 ,C 1
```
```
idC 1 ,C 2
```
Beispiel 16.10. SeiV = R^2 mit den Basen B 1 undB 2 aus Beispiel 16.8 und den
Basiswechselmatrizen

```
idB 1 ,B 2 =
```
##### [

##### 1 − 1

##### 0 1

##### ]

```
und idB 2 ,B 1 =
```
##### [

##### 1 1

##### 0 1

##### ]

##### .

Weiter seif∈L(R^2 ,R^2 ) mit

```
f
```
##### ([

```
x 1
x 2
```
##### ])

##### =

##### [

```
x 1 +x 2
2 x 2
```
##### ]

##### .

Wir berechnen zuerstfB 1 ,B 1 (also mitC 1 =B 1 ):

```
fB 1 ,B 1 =
```
##### [

```
KB 1 (f([^10 ])) KB 1 (f([^01 ]))
```
##### ]

##### =

##### [

##### KB 1

##### ([

##### 1

##### 0

##### ])

##### KB 1

##### ([

##### 1

##### 2

##### ])]

##### =

##### [

##### 1 1

##### 0 2

##### ]

##### .

Nun wollen wirfB 2 ,B 2 (also mitC 2 =B 2 ) berechnen. Das k ̈onnen wir direkt tun, oder
den Satz verwenden:

```
fB 2 ,B 2 = idB 1 ,B 2 fB 1 ,B 1 idB 2 ,B 1 =
```
##### [

##### 1 − 1

##### 0 1

##### ][

##### 1 1

##### 0 2

##### ][

##### 1 1

##### 0 1

##### ]

##### =

##### [

##### 1 0

##### 0 2

##### ]

##### .

Wir sehen, dass die zweite darstellende Matrix diagonal und damit noch einfacher ist als
die erste.


Besonders wichtig ist f ̈ur uns der Fall von gleichen Vektorr ̈aumen mit gleichen Basen:
V =WmitC 1 =B 1 undC 2 =B 2. Der zugeh ̈orige Basiswechsel hat die Gestalt

```
fB 2 ,B 2 = idB 1 ,B 2 fB 1 ,B 1 idB 2 ,B 1 = (idB 2 ,B 1 )−^1 fB 1 ,B 1 idB 2 ,B 1 ,
```
Schreiben wir abkurzend ̈ A 1 =fB 1 ,B 1 ,A 2 =fB 2 ,B 2 undS= idB 2 ,B 1 , so ist also

```
A 2 =S−^1 A 1 S,
```
genau wie im letzten Beispiel. In Vorlesung 34 werden wir systematisch quadratische
Matrizen auf diese Weise auf Diagonalgestalt bringen.



Vorlesung 17

## 17 Konvergenz von Zahlenfolgen

Wir f ̈uhren Folgen ein und den zentralen Begriff der Konvergenz.

### 17.1 Zahlenfolgen

Definition 17.1(Folge).EineFolgereeller Zahlen ist eine Abbildung

```
N→R, n7→an.
```
Schreibweisen: (an)n∈Nodera 0 ,a 1 ,a 2 ,.... Das Elementanheißtn-tes Folgenglied, und
nder zugeh ̈origeIndex.

Statt (an)n∈Nschreibt man auch kurzer ( ̈ an)noder nur (an). Wie man den Index
benennt ist dabei unerheblich: weitere ̈ubliche Bezeichnung sind j,k,`,m, also etwa
(an)n= (ak)k.
Allgemeiner kann man auch Folgen (an)n≥n 0 , alsoan 0 ,an 0 +1,an 0 +2,..., f ̈ur beliebiges
n 0 ∈Zbetrachten.
Genauso k ̈onnen wir Folgen komplexer Zahlen betrachten, wo dieandann komplexe
Zahlen sein k ̈onnen.

Beispiel 17.2.Beispiele f ̈ur Folgen sind:

```
1)an=cf ̈ur einc∈R, also die konstante Folgec,c,c,.... F ̈ur diese schreiben wir
auch (c)n.
```
```
2)an= 21 n, also 1,^12 ,^14 ,^18 ,...
```
```
3)an=^1 nf ̈urn≥1, also 1,^12 ,^13 ,...
```
```
4)an=(−1)
```
```
n
n f ̈urn≥1, also−^1 ,
```
```
1
2 ,−
```
```
1
3 ,
```
```
1
4 ,....
5)an= (−1)n, also 1,− 1 , 1 ,− 1 ,....
```

Diese Folgen sind alleexplizitdefiniert: man kannanf ̈ur jedesndirekt ausrechnen.
Folgen k ̈onnen auchrekursiv definiert werden, wobei ein Folgenglied von einem (oder
mehreren) Vorg ̈anger(n) abh ̈angt.

Beispiel 17.3.Beispiele f ̈urrekursivdefinierte Folgen sind:

```
1)a 1 = 1,an+1= (n+ 1)anf ̈urn≥1, also 1, 2 , 6 , 24 ,...
```
```
2)a 0 = 0,a 1 = 1 undan+2=an+1+anfur ̈ n≥0, also 0, 1 , 1 , 2 , 3 , 5 , 8 , 13 , 21 , 34 ,....
Dies ist die Fibonacci-Folge, die die Vermehrung von (unsterblichen) Kaninchen
modelliert.
```
```
Beschr ̈anktheit und Monotonie sind wie f ̈ur reelle Funktionen definiert.
```
Definition 17.4(Beschr ̈anktheit und Monotonie von Folgen).Die Folge (an)n∈Nheißt

```
1)nach unten beschr ̈ankt, wenn es eine Zahlm∈Rgibt mitm≤anf ̈ur allen∈N.
```
```
2)nach oben beschr ̈ankt, wenn es eine ZahlM∈Rgibt mitan≤Mf ̈ur allen∈N.
```
```
3)beschr ̈ankt, wenn es eine ZahlM∈Rgibt mit|an|≤Mf ̈ur allen∈N.
```
```
4)monoton wachsend, wennan≤an+1fur alle ̈ n∈N.
```
```
5)streng monoton wachsend, wennan< an+1f ̈ur allen∈N.
```
```
6)monoton fallend, wennan≥an+1f ̈ur allen∈N.
```
```
7)streng monoton fallend, wennan> an+1f ̈ur allen∈N.
```
Wir sagen kurz(streng) monoton, wenn die Folge (streng) monoton wachsend oder
(streng) monoton fallend ist.

Vergleichen Sie die Definition von Monotonie bei Folgen und allgemeinen reellen
Funktionen: Da Folgen auf den nat ̈urlichen Zahlen definiert sind, reicht es zwei aufein-
ander folgende Punkte (nundn+ 1) zu betrachten. Bei Funktionen, die in einem ganzen
Intervall definiert sind, geht das nicht: Zux∈Rgibt es nicht die”n ̈achste“ reelle Zahl.
Daher m ̈ussen dort immer Funktionswerte f ̈ur beliebige Punktex < yverglichen werden.

Beispiel 17.5. 1)an=nist (streng) monoton wachsend und nach unten beschr ̈ankt
(an≥0 f ̈ur allen∈N), aber nicht nach oben beschr ̈ankt.
2)an=^1 nf ̈urn≥1 ist (streng) monoton fallend und beschr ̈ankt: 0≤an≤1 f ̈ur alle
n≥1.
3)an= (−1)nist beschr ̈ankt (|an|= 1), aber nicht monoton.

Eine monoton wachsende Folge ist nach unten durcha 0 beschr ̈ankt, eine monoton
fallende Folge ist nach oben durcha 0 beschr ̈ankt.


### 17.2 Konvergenz

Eine zentrale Frage bei Folgen ist, was die Folge fur ̈ ”n→ ∞“, also f ̈ur immer gr ̈oßer
werdenden, tut.

Definition 17.6 (Konvergenz). Sei (an)n eine Folge reeller Zahlen. Die Folge heißt
konvergentgegena∈R, falls gilt: F ̈ur alleε >0 existiert einNε∈Nso, dass fur alle ̈
n≥Nεgilt
|an−a|< ε.

Die Zahlaheißt derGrenzwert(oderLimes) der Folge und wir schreiben limn→∞an=a,
oder
”
an→af ̈urn→∞“ oder kurz
”
an→a“.
Die Folge (an)nheißtkonvergent, falls (an)neinen Grenzwert hat. Die Folge (an)n
heißtdivergent, falls sie nicht konvergent ist.
Eine Folge, die gegen Null konvergiert, heißtNullfolge.

Die reelle Zahlenfolge (an)nkonvergiert also genau dann gegena∈R, falls f ̈ur jede
(noch so kleine) Toleranzε >0 alle bis auf endlich viele Folgenglieder (n ̈amlich h ̈ochstens
die mitn < Nε) im Intervall ]a−ε,a+ε[ liegen.
Betrachtet man die Folgenglieder als Punkte auf der Zahlengeraden, so bedeutet
Konvergenz gegena∈R, dass f ̈ur jedesε >0 alle Folgenglieder (abNε) in dem hellgrauen
Intervall liegen:

```
a−ε a a+ε
```
```
x 0 x 1 xNεxNε+1 x 2
```
Tr ̈agt man die Folge als Funktion auf, so bedeutet Konvergenz, dass fur jedes ̈ ε >0 alle
Folgenglieder (abNε) in dem”ε-Schlauch“ umaliegen:

```
n
```
```
an
```
```
a
```
```
a+ε
```
```
a−ε
```
```
Nε
```
Beispiel 17.7. 1) Die Folge

##### ( 1

```
n
```
##### )

```
n≥ 1 konvergiert gegen 0:
```
```
lim
n→∞
```
##### 1

```
n
```
##### = 0.


```
Dazu prufen wir die Definition nach: Sei ̈ ε >0 beliebig. Dann gibt esNε∈Nmit
Nε>^1 εund f ̈urn≥Nεist dann
∣
∣∣
∣
```
##### 1

```
n
```
##### − 0

##### ∣

##### ∣∣

##### ∣=

##### 1

```
n
```
##### ≤

##### 1

```
Nε
< ε.
```
```
2) Die konstante Folgec,c,c,...konvergiert gegenc.
3) Die Folge ((−1)n)nist divergent. Die Folgenglieder sind abwechselnd 1 und−1,
haben also Abstand 2. W ̈urde die Folge konvergieren, m ̈ussten aber alle Folgen-
glieder (bis auf endlich viele Ausnahmen)
”
nahe“ an dem Grenzwert liegen, zum
Beispiel n ̈aher alsε= 1...
Das pr ̈azisieren wir jetzt. Dazu nehmen wir an, dass die Folge gegen einen Grenz-
werta∈Rkonvergiert. Zuε= 1 gibt es dann einN∈Nmit|xn−a|< ε= 1 f ̈ur
allen≥N. Dann gilt insbesondere
|xN−xN+1|=|xN−a+a−xN+1|≤|xN−a|+|xN+1−a|<1 + 1 = 2,
andererseits ist aber|xN−xN+1|=|(−1)N−(−1)N+1|=|(−1)N+ (−1)N|= 2,
so dass also 2<2 gelten wurde, ein Widerspruch. Daher ist die Annahme, dass ̈
((−1)n)nkonvergiert, falsch, also ist ((−1)n)ndivergent.
```
Bemerkung 17.8.Das Verhalten der erstenmFolgengliedera 0 ,a 1 ,...,amhat keinen
Einfluss auf das Konvergenzverhalten. Genauer gesagt konvergiert die Folge (an)n≥n 0
genau dann, wenn die Folge (an)n≥Nkonvergiert. Kurz:

```
”
Ende gut, alles gut.“
```
Als Konsequenz kann man Konvergenz nicht durch Berechnung von endlich vielen Fol-
gengliedern nachweisen. Will man Aussagen ̈uber Konvergenz (von Verfahren) machen,
so muss man wirklich etwas beweisen.

Satz 17.9. 1) Der Grenzwert einer konvergenten Folge ist eindeutig.

```
2) Konvergente Folgen sind beschr ̈ankt.
3) Unbeschr ̈ankte Folgen sind divergent.
```
Beweis. 1) Angenommen, die Folge (an)nh ̈atte zwei verschiedene Grenzwertea 6 =a′.
Dann istε=|a−a′|/ 2 >0, und es existiert einNε∈N, so dass|an−a|< εf ̈ur alle
n≥Nεgilt, d.h. es gibt h ̈ochstens endlich viele Folgenglieder, die außerhalb von
]a−ε,a+ε[ liegen. Damit sind insbesondere h ̈ochstens endlich viele Folgenglieder
in ]a′−ε,a′+ε[, im Widerspruch dazu, dass die Folge auch gegena′konvergiert.
Damit ist die Annahme, dass es zwei verschiedene Grenzwerte geben kann, falsch.

```
a−ε a a+ε a′ a′+ε
a′−ε
alleanmit
n≥Nεhier
```
```
h ̈ochstens end-
lich vielean
```

```
2) Sei (an)n≥n 0 konvergent gegen a∈R. Zuε = 1 gibt es dann einN ∈ Nmit
|an−a|<1 fur alle ̈ n≥N. Dann ist
```
```
|an|=|an−a+a|≤|an−a|+|a|<1 +|a| fur ̈ n≥N.
```
```
Nehmen wir dannMals die Gr ̈oßte der Zahlen|an 0 |,|an 0 +1|,...,|aN− 1 |,1 +|a|, so
gilt
|an|≤M
f ̈ur allen≥n 0 , d.h. die Folge ist beschr ̈ankt.
```
```
3) Folgt aus2).
```
Bemerkung 17.10.Eine konvergente Folge ist beschr ̈ankt. Andersherum gilt das nicht:
Es gibt beschr ̈ankte Folgen die nicht konvergieren, z.B. ((−1)n)n∈Naus Beispiel 17.7.

Beispiel 17.11. Die Folgean=nist unbeschr ̈ankt, also divergent. Ebenso sindan=
(−1)nnundan=n^2 unbeschr ̈ankt und damit divergent.

### 17.3 Bestimmte Divergenz

Definition 17.12(Bestimmte Divergenz).Eine reelle Zahlenfolge (an)nheißtbestimmt
divergentgegen +∞, falls zu jedemM∈ReinN∈Nexistiert mit

```
an> M f ̈ur allen≥N.
```
Wir schreiben limn→∞an= +∞.
Entsprechend heißt (an)nbestimmt divergent gegen−∞, falls zu jedemM∈Rein
N∈Nexistiert mit
an< M f ̈ur allen≥N.

Wir schreiben limn→∞an=−∞.

Manche Autoren verwenden stattbestimmt divergent gegen+∞auch den Begriff
uneigentlich konvergent gegen+∞. Wir machen das nicht.”Konvergent“ bedeutet immer

”konvergent gegen eine reelle Zahl“.

Beispiel 17.13. 1) Die Folge (n)n∈Nist bestimmt divergent gegen +∞(fur ̈ M∈R
tut es jedesN∈NmitN > M). Ebenso sind (n^2 )n, (n^3 )n, usw. bestimmt divergent
gegen +∞.

```
2) Die Folge (ln(1/n))n≥ 1 ist bestimmt divergent gegen−∞. Ebenso sind (−n), (−n^2 ),
```
... bestimmt divergent gegen−∞.

```
3) Die Folge ((−1)n)nist divergent aber nicht bestimmt divergent.
```

```
4) Mit
```
```
xn=
```
##### {

```
n ngerade
0 nungerade
ist die Folge (xn) = (0, 0 , 2 , 0 , 4 , 0 , 6 ,...) divergent aber nicht bestimmt divergent
gegen +∞.
```
Bemerkung 17.14.Ebenso k ̈onnen wir Folgen komplexer Zahlen (komplexe Zahlenfolgen) betrachten,
die AbbildungenN→C,n7→an, sind. D.h. die Folgengliederansind komplexe Zahlen, zum Beispiel
ist (ik)k∈Ndie Folge
1 ,i,− 1 ,−i, 1 ,i,− 1 ,−i,....
Beschr ̈ankheit und Konvergenz sind ganz genau so wie f ̈ur reelle Zahlenfolgen definiert. Zum Beispiel
konvergiert die Folge

```
( 1
(3+4i)k
```
```
)
k∈Ngegen 0, weil
∣∣
∣∣^1
(3 + 4i)k−^0
```
```
∣∣
∣∣=^1
|3 + 4i|k=
```
```
1
5 k→^0.
```
Nur Monotonie k ̈onnen wir nicht mehr erkl ̈aren, da wir komplexe Zahlen nicht anordnen k ̈onnen.

In dieser Vorlesung haben wir Folgen kennen gelernt und den zentralen Begriff der
Konvergenz von Zahlenfolgen. Die Beispiele, die wir betrachtet haben, waren allesamt
sehr einfach. Fur komplizierter aussehende Folgen ist es hilfreich ein paar weitere Hilfs- ̈
mittel zur Hand zu haben. Diese lernen wir in der n ̈achsten Vorlesung kennen.


Vorlesung 18

## 18 Berechnung von Grenzwerten

In Vorlesung 17 haben wir die Konvergenz von Folgen kennen gelernt. In dieser Vorlesung
behandeln wir wichtige Hilfsmittel zur Berechnung von Grenzwerten von Folgen.

### 18.1 Grenzwerts ̈atze

Die Grenzwerts ̈atze erlauben die Bestimmung von Grenzwerten durch”R ̈uckfuhrung auf ̈
bekannte Bauteile“.

Satz 18.1(Grenzwerts ̈atze). Seien(an)nund(bn)nkonvergente Folgen mit

```
lim
n→∞
an=a und lim
n→∞
bn=b.
```
Dann gilt:

```
1) nlim→∞(an+bn) = limn→∞an+ limn→∞bn=a+b.
```
```
2) nlim→∞(an−bn) = limn→∞an−nlim→∞bn=a−b.
```
```
3) nlim→∞(anbn) = ( limn→∞an)( limn→∞bn) =ab, insbesondere istnlim→∞(can) =acf ̈urc∈R.
```
```
4) Ist b 6 = 0, so gibt es ein n 0 ∈ N mitbn 6 = 0f ̈ur alle n≥ n 0 , und dann ist
lim
n→∞
```
```
an
bn
```
##### =

```
limn→∞an
limn→∞bn
```
##### =

```
a
b
```
##### .

Beweis. Wir beweisen 1), um den Begriff der Konvergenz zuuben. Seien ̈ a= limn→∞an
und b= limn→∞bn. Sei ε > 0, dann ist auch 2 ε > 0 und es existiertNε ∈ Nmit
|an−a|<ε 2 und|bn−b|<ε 2. F ̈ur allen≥Nεist

```
|(an+bn)−(a+b)|=|(an−a) + (bn−b)|≤|an−a|+|bn−b|<
ε
2
```
##### +

```
ε
2
```
```
=ε.
```
Daher ist (an+bn)nkonvergent gegena+b.


Bemerkung 18.2.Achtung, wenn die Folgen (an)nund (bn)nnicht konvergieren, dann
gelten die Genzwerts ̈atze nicht.

Beispiel 18.3. 1) Was ist der Grenzwert der Folge

```
( 2 n+1
1+n
```
##### )

```
n∈N? Standardtrick: Nutze
limn→∞n^1 = 0 (siehe Beispiel 17.7) und die Grenzwerts ̈atze. K ̈urzen wirngilt f ̈ur
n≥1:
```
```
an=
```
```
2 n+ 1
1 +n
```
##### =

```
2 +n^1
1
n+ 1
```
##### .

```
Da die Folge (^1 n)nund die konstanten Folgen (2)nund (1)nkonvergieren, gilt mit
den Grenzwerts ̈atzen:
```
```
lim
n→∞
an= lim
n→∞
```
```
2 +^1 n
1
n+ 1
```
##### =

```
limn→∞2 + limn→∞^1 n
limn→∞^1 n+ limn→∞ 1
```
##### =

##### 2 + 0

##### 0 + 1

##### = 2.

```
2) Was ist der Grenzwert der Folgebn=a^2 n? Da wir schon wissen, dass (an)nkonver-
giert, finden wir mit den Grenzwerts ̈atzen
```
```
lim
n→∞
bn= lim
n→∞
(an·an) =
```
##### (

```
lim
n→∞
an
```
##### )

##### ·

##### (

```
lim
n→∞
an
```
##### )

##### = 2·2 = 4.

Beispiel 18.4.Sindk,`∈Nundpk,q` 6 = 0, so ist

```
lim
n→∞
```
```
pknk+pk− 1 nk−^1 +...+p 1 n+p 0
q`n`+q`− 1 n`−^1 +...+q 1 n+q 0
```
##### =

##### 

##### 

##### 

##### 

##### 

##### 

```
0 fur ̈ k < `
pk
q` fur ̈ k=`
+∞ fur ̈ k > `undpqk` > 0
−∞ fur ̈ k > `undpqk` < 0.
```
Das sieht man so: Fur ̈ k≤`k ̈urzt mann`und verwendet die Grenzwerts ̈atze, genau wie
in Beispiel 18.3. F ̈urk > `klammern wir im Z ̈ahlernkund im Nennern`aus:

```
nk−`
```
```
pk+pkn−^1 +...+npk^1 − 1 +np^0 k
q`+q`n−^1 +...+nq`−^11 +qn^0 `
```
##### .

##### ︸ ︷︷ ︸

```
=:an
```
Dann konvergiertangegena:= pqk` 6 = 0 (Grenzwerts ̈atze!). Ista >0, so istε= a 2 > 0
und es gibtN∈Nmitan∈]a−a 2 ,a+a 2 [ f ̈urn≥N, d.h.an≥a 2. Damit sehen wir, dass
nk−`an≥nk−`a 2 , was gr ̈oßer als jede SchrankeM∈Rwird, d.h. die Folge ist bestimmt
divergent gegen +∞. Fallsa <0 ist, sehen wir ̈ahnlich, dass die Folge bestimmt divergent
gegen−∞ist.


### 18.2 Grenzwerte und Ungleichungen

Grenzwertbildung erh ̈alt schwache Ungleichungen.

Satz 18.5.Sind(an)nund(bn)nkonvergente Folgen reeller Zahlen mitan≤bnf ̈ur alle
n∈N, so gilta≤b.

Vorsicht: Ausan < bnfolgt ebenfalls nura≤b.Die strikte Ungleichung geht im
Grenzwert verloren.Beispiel:an= 0<^1 n=bn, aber limn→∞an= 0 = limn→∞bn.
Besonders hilfreich ist das folgende Sandwich-Theorem (oder Drei-Folgen-Satz).

Satz 18.6(Sandwich-Theorem). Seien(an)n,(bn)n,(cn)ndrei reelle Folgen mit

```
an≤bn≤cn f ̈ur allen
```
und mit
lim
n→∞
an= lim
n→∞
cn=a.

Dann konvergiert auch(bn)ngegena:

```
lim
n→∞
bn=a.
```
Besonders oft verwendet man das Sandwich-Theorem f ̈ur Absch ̈atzungen 0≤|an|≤
bnmitbn→0. Dann folgt|an|→0 und mit−|an|≤an≤|an|dann auchan→0.

Beispiel 18.7. 1) Es ist

```
−
```
##### 1

```
n
```
##### ≤

```
(−1)n
n
```
##### ≤

##### 1

```
n
```
##### .

```
Dan^1 →0 ist auch−^1 n →0 (Grenzwertsatz) und daher (−1)
```
```
n
n →0 (Sandwich-
Theorem).
2) Wir zeigen, dassan=sin(nn)gegen 0 konvergiert:
```
```
0 ≤|an|=
```
##### ∣∣

```
∣∣sin(n)
n
```
##### ∣∣

##### ∣∣≤^1

```
n
```
##### → 0 ,

```
alsoan→0.
3) Wir zeigen limn→∞x
n
n! = 0 f ̈ur x∈ R. Dazu w ̈ahlen wir ein festesk∈ Nmit
k > 2 |x|. Dann gilt f ̈ur allen≥k:
∣
∣∣
∣
```
```
xn
n!
```
##### ∣

##### ∣∣

##### ∣=

```
|x|n
n!
```
##### =

```
|x|k
k!
```
```
|x|n−k
(k+ 1)·...·n
```
##### ≤

```
|x|k
k!
```
```
|x|n−k
(2|x|)n−k
```
##### =

```
|x|k
k!
```
##### 1

```
2 n−k
```
##### =

```
|x|k 2 k
k!
```
##### 1

```
2 n
```
##### → 0

```
f ̈urn→∞.
```
```
Als Folgerung aus dem Sandwich-Theorem haben wir den folgenden Satz.
```
Satz 18.8. Nullfolge mal beschr ̈ankte Folge ist wieder eine Nullfolge.


### 18.3 Monotonie und Konvergenz

Ein schwieriges Problem ist es, die Konvergenz einer Folge zu beweisen, wenn man den
Grenzwert nicht kennt (und keine Vermutung f ̈ur ihn hat). Relativ oft kann einem das
folgende hinreichende Kriterium helfen.

Satz 18.9(Monotoniekriterium).Jede beschr ̈ankte und monotone Folge reeller Zahlen
ist konvergent.

Beachten Sie: Andersherum ist jede konvergente Folge beschr ̈ankt, aber nicht not-
wendigerweise monoton: zum Beispiel konvergiert

##### (

```
(−1)n
n
```
##### )

```
n≥ 1
```
```
, ist aber nicht monoton.
Monotone Folgen, die nicht beschr ̈ankt sind, sind bestimmt divergent.
```
Beispiel 18.10(Wurzelfolge).Konvergiert die durch

```
a 0 >0 und an+1=
```
##### 1

##### 2

##### (

```
an+
```
##### 2

```
an
```
##### )

```
, n≥ 0 ,
```
rekursiv definierte Folge? Um eine Idee zu bekommen betrachten wir den Startwert
a 0 = 1 und berechnen die ersten Folgenglieder (auf 15 Nachkommastellen gerundet):

```
a 0 = 1
a 1 = 1. 5
a 2 = 1. 416666666666667
a 3 = 1. 414215686274510
a 4 = 1. 414213562374690
a 5 = 1. 414213562373095.
```
Die Folgescheintzu konvergieren und abn= 1 monoton fallend zu sein. Wir versuchen
daher Konvergenz mit dem Monotoniekriterium zu beweisen.

- Die Folge ist nach unten beschr ̈ankt: F ̈urn≥0 ist

```
an+1=
```
```
an+a^2 n
2
```
##### ≥

##### √

```
an
```
##### 2

```
an
```
##### =

##### √

##### 2 ,

```
weil das arithmetische Mittel gr ̈oßer gleich dem geometrischen Mittel ist (Bei-
spiel 2.3).
```
- Die Folge ist monoton fallend (abn= 1): F ̈urn≥1 istan≥

##### √

```
2, alsoa^2 n≥2, also
```
```
an+1
an
```
##### =

##### 1

##### 2

##### (

##### 1 +

##### 2

```
a^2 n
```
##### )

##### ≤

##### 1

##### 2

##### (

##### 1 +

##### 2

##### 2

##### )

##### = 1,

alsoan+1≤an.
Damit ist die Folge konvergent gegen einen Grenzwerta∈R. Daan≥

##### √

```
2 ista≥
```
##### √

##### 2.


Trick zur Bestimmung des Grenzwertes bei rekursiven Folgen: Daan→agilt
auchan+1→a. Nun ist

```
an+1=
```
##### 1

##### 2

##### (

```
an+
```
##### 2

```
an
```
##### )

##### .

Bilden wir auf beiden Seiten den Grenzwert (Grenzwerts ̈atze) erhalten wir:

```
a= limn→∞an+1= limn→∞
```
##### 1

##### 2

##### (

```
an+
```
##### 2

```
an
```
##### )

##### =

##### 1

##### 2

##### (

```
a+
```
##### 2

```
a
```
##### )

##### =

```
a
2
```
##### +

##### 1

```
a
```
##### ,

also a 2 =^1 a und danna^2 = 2. Damit kommen f ̈ur den Grenzwerta=

##### √

```
2 odera=−
```
##### √

##### 2

in Frage. Daa≥

##### √

```
2 (s.o.), ist der Grenzwerta=
```
##### √

2 = 1. 414213562373095 ...(auf 15
Nachkommastellen gerundet). Wir sehen: bereits das funfte Folgeglied ist auf 15 Stellen ̈
genau, diese Folge konvergiert sehr schnell!
Solche Folgen werden verwendet, um Quadratwurzeln von reellen Zahlen im Taschen-
rechner und Computer zu berechnen.

### 18.4 Wichtige Grenzwerte

Beispiel 18.11(Geometrische Folge). F ̈urq∈Rist

```
nlim→∞qn=
```
##### 

##### 

##### 

##### 

##### 

```
0 , − 1 < q < 1 ,
1 , q= 1,
+∞, 1 < q,
divergent, q≤− 1.
```
Anschaulich ist das vermutlich klar, wir wollen das aber mit den gelernten Methoden
nachweisen.

- F ̈urq >1: Setzey:=q− 1 >0. Dann ist

```
qn= (1 +q−1)n= (1 +y)n= 1 +ny+
```
```
∑n
```
```
k=2
```
##### (

```
n
k
```
##### )

```
yk 1 n−^1 ≥ny,
```
```
und day >0 istqnbestimmt divergent gegen +∞.
```
- F ̈urq= 1 istqn= 1→1.
- F ̈ur− 1 < q <1: Es ist|qn− 0 |=|q|nund wir zeigen mit dem Monotoniekriterium,
    dass (|q|n) gegen Null konvergiert. Die Folge ist beschr ̈ankt, da|q|n≤1 f ̈ur alle
    n∈N. Die Folge ist monoton fallend, da|q|n+1=|q|·|q|n≤ |q|ngilt. Damit ist
    |q|n→af ̈ur eina∈R. Aus|q|n+1=|q|·|q|nfolgt danna=|q|a, also (1−|q|)a= 0,
    alsoa= 0. Somit gilt|qn|=|q|n→0.
- F ̈urq≤−1: F ̈urq=−1 ist ((−1)n)ndivergent (Beispiel 17.7). Fur ̈ q <−1 istqn
    nicht beschr ̈ankt, denn|qn|=|q|n→+∞. Da die Vorzeichen vonqnwechseln, ist
    (qn)naber nicht bestimmt divergent, sondern unbestimmt divergent.


Beispiel 18.12(Geometrische Reihe).F ̈urq∈Rist mit der geometrischen Summe

```
sn:=
```
```
∑n
```
```
k=0
```
```
qk=
```
```
{ 1 −qn+1
1 −q , q^6 = 1,
n+ 1, q= 1.
```
Mit Beispiel 18.11 sehen wir: Die Folge (sn)n≥ 0 konvergiert fur ̈ |q|<1 gegen 1 −^1 q, f ̈ur
q≥1 ist (sn)n≥ 0 bestimmt divergent gegen +∞, und f ̈urq≤ −1 ist (sn)n≥ 0 divergent.
Fur den Grenzwert der Summe schreibt man suggestiv ̈

##### ∑∞

```
k=0q
k = limn→∞∑n
k=0q
k.
```
Zusammengefasst haben wir

##### ∑∞

```
k=0
```
```
qk= lim
n→∞
```
```
∑n
```
```
k=0
```
```
qk=
```
##### 

##### 

##### 

```
1
1 −q, |q|<^1 ,
+∞, q≥ 1 ,
divergent, q≤− 1.
```
Grenzwerte von Summen nennt man auchReihen, wir werden diese in Vorlesung 40
n ̈aher untersuchen.

```
Schlußendlich f ̈uhren wir ein paar n ̈utzliche Grenzwerte ohne Beweis auf:
```
- lim
    n→∞

```
xn
n!
```
```
= 0 f ̈ur allex∈R.
```
- lim
    n→∞

```
√nx= 1 f ̈urx >0.
```
- lim
    n→∞

```
√nn= 1.
```
- lim
    n→∞

##### (

##### 1 +

```
x
n
```
```
)n
=exf ̈ur allex∈R.
```
- nlim→∞

```
ln(n)
nα
= 0 f ̈urα >0.
```
- lim
    n→∞

```
nα
eβn
= lim
n→∞
nαe−βn= 0 f ̈urα∈R,β >0.
```
- nlim→∞nαqn= 0 f ̈urα∈R,|q|<1.
- lim
    n→∞

```
n
√nn!=e.
```

Vorlesung 19

## 19 Stetigkeit

Ein sich bewegendes physikalisches Objekt kann nicht an einer Stelle verschwinden und
an einer anderen Stelle wieder erscheinen, um seine Bewegung fortzusetzen. Die Kurve,
die das Objekt beschreibt, hat keine
”
Spr ̈unge“. Das mathematische Konzept hinter
diesem Ph ̈anomen ist die Stetigkeit, die wir in dieser Vorlesung behandeln. Unser Zugang
zur Stetigkeit einer Funktion verwendet Grenzwerte von Funktionen, die wir zun ̈achst
erkl ̈aren.

### 19.1 Grenzwerte von Funktionen

Wir f ̈uhren Grenzwerte von Funktionen ein. Dabei verwenden wir den Grenzwertbegriff
von Folgen.
Vorweg eine Beobachtung: Istf:D→Reine Funktion, und ist (xn)neine Folge im
Definitionsbereich vonf, so k ̈onnen wirf(xn) berechnen, und erhalten eine neue Folge
(f(xn))nvon reellen Zahlen, die wir wieder auf Konvergenz untersuchen k ̈onnen.

Definition 19.1 (Konvergenz). Seif :D→Reine Funktion (D⊆R) und seia∈
R∪{−∞,+∞}. Wir sagen,fhat f ̈urxgegenaden Grenzwertc∈R∪{−∞,+∞}, in
Zeichen
lim
x→a
f(x) =c,

falls gilt:

```
1) F ̈urjedeFolge (xn)nmit
(a) xn∈D,
(b)xn 6 =a,
(c) limn→∞xn=a,
ist limn→∞f(xn) =c.
```
```
2) Es gibt mindestens eine Folge (xn)nmit (a)–(c).
```
Bemerkung 19.2. 1)akann im Definitionsbereich vonfsein, muss aber nicht.


```
2) In der Definition setzen wir voraus, dass es Folgen (xn)ninD\{a}gibt, die gegen
akonvergieren: Der Punktamuss vonD\{a}”erreichbar“ sein. Zum Beispiel ist
a= 2 f ̈urD= [0,1]∪{ 2 }nicht ausD\{ 2 }= [0,1] erreichbar.
```
Beispiel 19.3. 1) Seif:R→R,x7→x^2. Bestimme limx→af(x) fur ̈ a∈R. Fur jede ̈
Folge (xn)nmit limn→∞xn=aist (Grenzwertsatz)

```
nlim→∞f(xn) = limn→∞x^2 n=a^2.
Da dies f ̈ur jede Folge (xn)ngilt und immer den gleichen Grenzwert liefert, ist
```
```
xlim→af(x) =a^2.
```
```
2) F ̈urf :R\{ 0 } →R,f(x) =xsin(^1 x), ist limx→ 0 f(x) = 0, denn f ̈ur jede Folge
(xn)nmitxn∈R\{ 0 }und limn→∞xn= 0 gilt:
```
```
|f(xn)|=
```
##### ∣

##### ∣∣

```
∣xnsin
```
##### (

##### 1

```
xn
```
##### )∣∣

##### ∣

```
∣=|xn|
```
##### ∣

##### ∣∣

```
∣sin
```
##### (

##### 1

```
xn
```
##### )∣∣

##### ∣

```
∣≤|xn|→^0 ,
```
```
also limn→∞f(xn) = 0. Da dies f ̈ur jede Folge gilt und jeweils der gleiche Grenzwert
herauskommt, ist limx→ 0 f(x) = 0.
```
##### − 0. 2 − 0. 1 0 0. 1 0. 2

##### − 0. 2

##### − 0. 1

##### 0

```
0. 1 xsin(x^1 )
```
```
3) Die Heaviside-FunkionH:R→R,H(x) =
```
##### {

```
0 x < 0
1 x≥ 0
```
```
, besitzt in 0 keinen Grenz-
```
```
wert, denn
```
```
nlim→∞H
```
##### (

##### −

##### 1

```
n
```
##### )

```
= limn→∞0 = 0 aber nlim→∞H
```
##### (

##### 1

```
n
```
##### )

```
= limn→∞1 = 1.
```
```
x
```
```
y
1
H(x)
```

```
Es gelten die selben Rechenregeln wie f ̈ur Grenzwerte von Folgen.
```
Satz 19.4(Grenzwerts ̈atze fur Funktionen) ̈ .Sindlimx→af(x) =cundlimx→ag(x) =d
mitc,d∈R, so gilt:

```
1) limx→a(f(x) +g(x)) = limx→af(x) + limx→ag(x) =c+d,
```
```
2) limx→a(f(x)g(x)) =
```
##### (

```
limx→af(x)
```
##### )(

```
limx→ag(x)
```
##### )

```
=cd,
```
```
3) limx→a(αf(x)) =αlimx→af(x) =αcf ̈ur jedesα∈R,
```
```
4) limx→afg((xx))=limlimxx→→aafg((xx))=cd, fallsd 6 = 0.
```
Die Rechenregeln gelten auch f ̈ura=±∞. Hingegen m ̈ussenc,dendlich sein.

```
Die Rechenregeln 1) und 3) besagen, dass Grenzwertbildung linear ist.
Das Sandwich-Prinzip gilt auch f ̈ur Grenzwerte von Funktionen.
```
Beispiel 19.5.Wir berechnen limx→ 0 sin(xx) mit dem Sandwich-Prinzip. Zur Absch ̈at-
zung betrachten wir f ̈ur 0< x <π 2 die Fl ̈acheninhalte in der Skizze:

##### 0 1

```
x
cos(x)
```
```
sin(x)
```
```
tan(x)
```
- Fl ̈ache des kleinen Dreiecks:^12 sin(x) cos(x)
- Fl ̈ache des Kreissektors: 2 xπ 12 π=x 2
- Fl ̈ache des großen Dreiecks:^12 tan(x).
Damit ist sin(x) cos(x)≤x≤tan(x), also

```
cos(x)≤
sin(x)
x
```
##### ≤

##### 1

```
cos(x)
```
f ̈ur 0< x <π 2 und, wegen der Symmetrie von cos und sin, auch f ̈ur−π 2 < x <0. Wegen

limx→ 0 cos(x) = 1 folgt limx→ 0 sin(xx)= 1 mit dem Sandwich-Theorem.

### 19.2 Einseitige Grenzwerte von Funktionen

Manchmal sind auch einseitige Grenzwerte n ̈utzlich. Fur den ̈ linksseitigen Grenzwert
vonf inafordert manxn< astattxn 6 =ain Definition 19.1 (diexnn ̈ahern sich von
links ana) und schreibt

```
lim
x↗a
f(x) =c oder lim
x→a−
f(x) =c.
```

Fur den ̈ rechtsseitigen Grenzwert vonf inafordert manxn> a stattxn 6 = aund
schreibt
lim
x↘a
f(x) =c oder xlim→a+f(x) =c.

Es gelten die gleichen Rechenregeln wie in Satz 19.4. Man kann zeigen:

```
lim
x→a
f(x) =c ⇔ lim
x↗a
```
```
f(x) =c= lim
x↘a
```
```
f(x),
```
d.h. die Funktion hat den Grenzwertcfur ̈ x gegenagenau dann, wenn links- und
rechtsseitiger Grenzwert existieren und beide gleichcsind.

Beispiel 19.6. 1) Seif:R\{ 0 } →R,f(x) = |xx|=

##### {

```
1 , x > 0 ,
− 1 , x < 0.
```
. Fur jede Folge ̈

```
(xn)nmitxn<0 und limn→∞xn= 0 ist
```
```
nlim→∞f(xn) = limn→∞
```
```
xn
−xn
```
##### =− 1 ,

```
also ist limx↗ 0 f(x) =−1. Genauso findet man limx↘ 0 f(x) = 1. Insbesondere
existiert der Grenzwert limx→ 0 f(x) nicht.
2) limx↘ 0 ln(x) =−∞.
```
### 19.3 Stetigkeit

Definition 19.7(Stetigkeit). Istf:D→Runda∈D, dann heißtfstetig inawenn

```
lim
x→a
f(x) =f(a). (19.1)
```
Die Funktion heißtstetig aufD, wenn sie in allena∈Dstetig ist.

Gleichung (19.1) beinhaltet zwei Bedingungen: Der Grenzwert muss existierenund
gleich dem Funktionswert sein.

```
x
```
```
y
```
```
a
```
```
x
```
```
y
```
```
a
```
```
f(a)
x
```
```
y
```
```
a
```
Die Funktion links ist stetig ina, die Funktion in der Mitte ist unstetig, da der Grenzwert
limx→af(x) 6 =f(a) ist, die Funktion rechts ist unstetig, da der Grenzwert limx→af(x)
nicht existiert.


Bemerkung 19.8.Stetigkeit erlaubt die
”
Vertauschung von Funktion und Grenzwert“:

```
lim
x→a
f(x) =f(a) =f
```
##### (

```
lim
x→a
x
```
##### )

##### .

Alternativ kann Stetigkeit ̈uber das sogenannte-δ-Kriterium definiert werden. Beide Definitionen
ergeben den gleichen Begriff.

Definition 19.9. Istf:D→Runda∈D, dann heißtfstetig inawenn es zu jedemε >0 einδ > 0
gibt, so dass f ̈ur allex∈Dmit|x−a|< δgilt|f(x)−f(a)|< ε.
Die Funktion heißtstetig aufD, wenn sie in allena∈Dstetig ist.
Die Idee bei derε-δ-Definition ist: In einer kleinen Umgebung vonasind die Funktionswerte nahe
dem Funktionswertf(a). Insbesondere hat die Funktion keinen Sprung.
Aus den Rechenregeln fur Grenzwerte erhalten wir die folgenden Rechenregeln f ̈ ̈ur
stetige Funktionen.

Satz 19.10(Rechenregeln f ̈ur stetige Funktionen).
1) Sindf,g:D→Rstetig, so sindf+g,f−g,αf f ̈urα∈RundfginDstetig.
2) Sindf,g:D→Rstetig, so istf/ginD\{x|g(x) = 0}stetig, also ̈uberall dort
wof/ggebildet werden kann.
3) Sindf :D→R,g:E→Rstetig undg(E)⊆D, so ist auch die Komposition
f◦g:E→R,x7→f(g(x)), inEstetig.

Beweis von 3). Seia∈E. Dagstetig ist, gilt limx→ag(x) =g(a). Dafstetig ing(a) ist, ist dann auch
f(g(a)) =f(limx→ag(x)) = limx→af(g(x)), also istf◦gstetig ina.

Teil 1) des Satzes besagt insbesondere, dass stetige Funktionen einen Vektorraum
bilden.

Beispiel 19.11. 1)f:R→R,x7→x, ist stetig, denn f ̈ur ein beliebigesa∈Rund
eine beliebige Folge (xn)nmit limn→∞xn=aundxn 6 =agilt:

```
nlim→∞f(xn) = limn→∞xn=a=f(a).
```
```
2)Polynome sind stetig:p :R→R,p(x) =a 0 +a 1 x+...+anxn, ist stetig als
Summe von Produkten stetiger Funktionen: Daf(x) = xstetig ist, sind auch
stetigx^2 =f(x)f(x),x^3 =f(x)·x^2 ,... , danna 0 ,a 1 x,... ,anxnund schließlich
die Summe dieser stetigen Funktionen.
```
```
3)Rationale Funktionen sind stetig:f:R→R,f(x) =pq((xx)), mit Polynomenp,qist
stetig inD=R\{x|q(x) = 0}als Quotient stetiger Funktionen.
f(x) =^1 xist stetig inR\{ 0 }, undf(x) =x
```
(^3) − 5 x+4
x^2 − 4 ist stetig inR\{−^2 ,^2 }.
4) Der Absolutbetragf:R→R,f(x) =|x|, ist stetig in 0: Da der Betrag links und
rechts von Null verschieden definiert ist, betrachten wir die links- und rechtsseitigen
Grenzwerte. Es sind
lim
x↘ 0
|x|= lim
x↘ 0
x= 0, lim
x↗ 0
|x|= lim
x↗ 0
(−x) = 0.


```
Da die einseitigen Grenzwerte existieren und gleich sind, ist limx→ 0 |x|= 0 =| 0 |,
also ist der Absolutbetrag in 0 stetig. Der Betrag ist auch in allen anderen Punkten
a 6 = 0 stetig.
```
```
5) Die Heaviside-FunktionH:R→Rist unstetig in 0, denn limx→ 0 H(x) existiert
nicht (Beispiel 19.3). Hingegen istH:R\{ 0 }→Rstetig.
```
```
6) Die Funktion
```
```
f:R→R, f(x) =
```
##### {

```
sin
```
##### ( 1

```
x
```
##### )

```
, x 6 = 0,
0 , x= 0,
ist unstetig in 0. Hier existiert der Grenzwert gegen 0 nicht. Dies sieht man zum
Beispiel mit der Folgexn= π 2 +^1 nπ 6 = 0 f ̈ur die limn→∞xn= 0 ist, aber
```
```
f(xn) = sin
```
```
(π
2
+nπ
```
##### )

```
= (−1)n
```
```
nicht konvergiert. Das zeigt auch, dass Unstetigkeitsstellen nicht unbedingt Sprung-
stellen sein mussen. Die Funktion oszilliert (schwingt) unendlich oft und immer ̈
schneller zwischen−1 und 1 hin und her:
```
##### − 1 − 0. 5 0 0. 5 1

##### − 1

##### − 0. 5

##### 0

##### 0. 5

##### 1

```
sin(^1 x)
```
```
Im Plot erreichen die Oszillationen nahe Null nicht ganz±1, da der Rechner nur
endlich viele Punkt zum Zeichnen des Graphens verwenden kann.
```
```
Die folgenden elementaren Funktionen sind stetig (Nachweis sp ̈ater):
1) Wurzelfunktionen sind stetig:f: [0,∞[→[0,∞[,x7→ k
```
##### √

```
x, ist stetig.
```
```
2) Die elementaren Funktionen exp, ln, sin, cos, tan sind stetig.
```
Stetige Fortsetzbarkeit. Seif :I\{a} →Rstetig, wobeiIein Intervall ist und
a∈I. Wenn der Grenzwert limx→af(x) =c∈Rexistiert, so k ̈onnen wir die Funktion

```
g:I→R, g(x) =
```
##### {

```
f(x), x 6 =a,
c, x=a
```

definieren, die dann stetig auf ganzIist. Diese Funktion setztfvonI\{a}nachIfort
und wird einestetige Fortsetzungvonf genannt. H ̈aufig nennt mangauch wiederf.

Beispiel 19.12. 1) Die rationale Funktionf:R\{ 1 } →R,f(x) = x

(^2) − 1
x− 1 , ist stetig
aufR\{ 1 }. Wegen
lim
x→ 1
f(x) = lim
x→ 1
x^2 − 1
x− 1
= lim
x→ 1
(x−1)(x+ 1)
x− 1
= lim
x→ 1
(x+ 1) = 2
kannfzu einer stetigen Funktion auf ganzRfortgesetzt werden:
f:R→R, f(x) =

##### {

```
x^2 − 1
x− 1 , x^6 = 1,
2 , x= 1.
```
```
In diesem Beispiel l ̈asst sichfauch Vereinfachen:f(x) =x
```
(^2) − 1
x− 1 =x+ 1, so dass
man sofort sieht, dassfstetig auf ganzRist (bzw. fortgesetzt werden kann). Das
ist aber nicht immer m ̈oglich.
x
y
f(x) =x
(^2) − 1
x− 1

##### 1

##### 2

```
x
```
```
y
```
```
stetige Fortsetzung
```
##### 1

##### 2

```
2) F ̈urf:R\{ 0 }→R,f(x) =xsin(^1 x), ist limx→ 0 f(x) = 0 (Beispiel 19.3), also ist
```
```
f:R→R, f(x) =
```
##### {

```
xsin
```
##### ( 1

```
x
```
##### )

```
, x 6 = 0,
0 , x= 0,
```
```
stetig.
3)f(x) = 1/xundf(x) =|xx|sind inR\{ 0 }stetig, lassen sich aber nicht stetig in 0
fortsetzen, da limx→ 0 f(x) nicht existiert.f(x) = 1/x^2 l ̈asst sich nicht stetig in 0
fortsetzen, da limx→ 0 f(x) =∞zwar existiert aber nicht endlich ist.
```


Vorlesung 20

## 20 S ̈atzeuber stetige Funktionen ̈

Wir behandeln die wichtigsten S ̈atze ̈uber stetige Funktionen. Dies sind der Zwischen-
wertsatz und der Satz ̈uber die Existenz vom Minimum und Maximum.

### 20.1 Bestimmung von Nullstellen

Das L ̈osen von Gleichungen fuhrt oft auf das Problem, die Nullstellen von Funktionen zu ̈
bestimmen. Bei quadratischen Polynomen ist das einfach, bei komplizierteren Funktionen
hingegen helfen oft nur numerische Verfahren. Dazu ist es n ̈utzlich zu wissen, ob es
uberhaupt Nullstellen gibt. ̈
Wennf(a)<0 undf(b)>0 undf stetig ist, muss anschaulich der Graph vonf
diex-Achse mindestens einmal zwischenaundbschneiden. Damit hatfeine Nullstelle.
Das besagt der folgende Satz. Der Beweis beruht auf einem einfachen Verfahren, um die
Nullstelle beliebig genau anzun ̈ahern.

```
x
```
```
y
```
```
a b
x
```
```
y
```
```
a b
```
Satz 20.1(Zwischenwertsatz – Version f ̈ur Nullstellen).Seif:I→Rstetig auf einem
IntervallI⊆R. Sind danna,b∈Imita < bundf(a)< 0 undf(b)> 0 (oder umgekehrt
f(a)> 0 undf(b)< 0 ), dann hatfmindestens eine Nullstelle in]a,b[.

Beweis mit dem Intervallhalbierungsverfahren. Wir nehmenf(a)<0 undf(b)>0 an
und konstruieren zwei Folgen (ak)kund (bk)kmitf(ak)<0 undf(bk)>0, die gegen
eine Nullstelle konvergieren.
Wir setzena 0 =aundb 0 =b. Dann istf(a 0 )<0 undf(b 0 )>0. Fur ̈ k= 0, 1 , 2 , 3 ,...

- Berechne den Mittelpunktxk=ak+ 2 bkvon [ak,bk] undf(xk).
- Fallunterscheidung:


- Wennf(xk) = 0 haben wir eine Nullstelle gefunden und sind fertig.
- Wennf(xk)<0, setzeak+1:=xkundbk+1:=bk.
- Wennf(xk)>0, setzeak+1:=akundbk+1:=xk.
Das liefert die beiden Folgen

```
a=a 0 ≤a 1 ≤a 2 ≤a 3 ≤...≤b,
b=b 0 ≥b 1 ≥b 2 ≥b 3 ≤...≤a.
```
Die Folge (ak)kist monoton wachsend und beschr ̈ankt, also konvergent gegen einx∗, die
Folge (bk)kist monoton fallend und beschr ̈ankt, also konvergent gegen einy∗. Wegen der
Halbierung der Intervalle ist

```
bk−ak=
```
##### 1

##### 2

```
(bk− 1 −ak− 1 ) =...=
```
##### 1

```
2 k
```
```
(b 0 −a 0 ) =
b−a
2 k
```
##### .

Fur ̈ k→∞folgt darausx∗=y∗. Nach Konstruktion der Folgen ist weiter fur alle ̈ k∈N

```
f(ak)≤ 0 ≤f(bk),
```
also, da Grenzwerte Ungleichungen erhalten (Satz 18.5) undfstetig ist,

```
f(x∗) = lim
k→∞
f(ak)≤ 0 ≤ lim
k→∞
f(bk) =f(x∗),
```
alsof(x∗) = 0. Damit hatfeine Nullstellex∗∈[a,b].

Das Intervallhalbierungsverfahren heißt auchBisektionsverfahren. Da die Nullstelle
x∗in jedem Intervall [ak,bk] liegt, gilt

```
|xk−x∗|≤
bk−ak
2
```
##### =

```
b−a
2 k+1
```
##### .

```
ak xk bk
```
```
x∗
```
Daher k ̈onnen wir die Nullstellex∗beliebig genau durchxkapproximieren (ann ̈ahern):
Zu vorgegebener Genauigkeitεw ̈ahlen wirkso groß, dass (b−a)/ 2 k+1< ε.

Beispiel 20.2. Wir betrachten die Funktionf : [0,2] →R,f(x) =x^2 −2. Es sind
f(0) =− 2 <0 undf(2) = 2>0, so dassfnach dem Zwischenwertsatz eine Nullstelle
in [0,2] hat (n ̈amlich

##### √

2). Um die Nullstelle approximativ zu berechnen, wenden wir
das Bisektionsverfahren an, und erhalten mita= 0 undb= 2 Folgen (ak)kund (bk)k,
die gegen die Nullstelle

##### √

2 ≈ 1 .414213562373095 konvergieren; vergleiche Tabelle 20.1.
Korrekte Stellen sind unterstrichen.
Der n ̈achste Mittelpunkt istx 20 = 1. 41421 4134216309. Immerhin die ersten funf ̈
Nachkommastellen stimmen mit

##### √

```
2 uberein. Der Fehler ist ̈ |x 20 −
```
##### √

2 |≈ 5. 7 · 10 −^7. Die
Folge (xk)kaus dem Bisektionsverfahren konvergiert viel langsamer als die Wurzelfolge
aus Beispiel 18.10.


```
k ak bk
0 0 2. 000000000000000
1 1.000000000000000 2. 000000000000000
2 1.000000000000000 1. 500000000000000
3 1.250000000000000 1. 500000000000000
4 1.375000000000000 1. 500000000000000
5 1.375000000000000 1. 437500000000000
6 1. 4 06250000000000 1. 437500000000000
7 1. 4 06250000000000 1. 421875000000000
8 1. 414 062500000000 1. 421875000000000
9 1. 414 062500000000 1. 417968750000000
10 1. 414 062500000000 1. 416015625000000
11 1. 414 062500000000 1. 415039062500000
12 1. 414 062500000000 1. 414550781250000
13 1. 414 062500000000 1. 414306640625000
14 1. 414 184570312500 1. 414306640625000
15 1. 414 184570312500 1. 414245605468750
16 1. 414 184570312500 1. 414215087890625
17 1. 414 199829101563 1. 414215087890625
18 1. 4142 07458496094 1. 414215087890625
19 1. 41421 1273193359 1. 414215087890625
20 1. 41421 3180541992 1. 414215087890625
```
```
Tabelle 20.1: Die Folgen (ak)kund (bk)kaus Beispiel 20.2.
```
Beispiel 20.3.Die Funktionf:R\{ 0 }→R,f(x) =^1 x, ist stetig undf(−1) =− 1 < 0
undf(1) = 1>0, sie hat aber keine Nullstelle. Warum nicht? Der Definitionsbereich ist
kein Intervall!

Der letzte Satz kann auf andere Werte als Nullstellen verallgemeinert werden. Sei
f : [a,b]→Rstetig mitf(a)< f(b) undcein Wert mitf(a)< c < f(b). Dann ist
f(x)−cstetig auf [a,b] und es giltf(a)−c < 0 < f(b)−c, also hatf(x)−ceine
Nullstelleξ∈]a,b[, d.h. es istf(ξ)−c= 0, alsof(ξ) =c. F ̈urf(a)> f(b) geht das
genauso. Eine stetige Funktion nimmt also alle Werte zwischenf(a) undf(b) an.


Satz 20.4(Zwischenwertsatz). Seif :I→Rstetig auf einem IntervallI⊆R. Sind
danna,b∈Iundcein Wert zwischenf(a)undf(b), so gibt es mindestens einξ∈[a,b]
mitf(ξ) =c.

```
x
```
```
y
```
```
a b
```
```
f(a)
```
```
f(b)
```
```
c
```
```
ξ
```
### 20.2 Existenz von Extremwerten

Ein anderes h ̈aufig auftretendes Problem neben der Bestimmung von Nullstellen ist die
Ermittlung des Maximums oder Minimums einer reellwertigen Funktion, also der gr ̈oßten
oder kleinsten Werte der Funktion. Dann ist es naturlich gut zu wissen, ob Minimum ̈
und Maximum ̈uberhaupt existieren, sonst sucht man vielleicht etwas, dass man nicht
finden kann, da es gar nicht existiert.

Definition 20.5(Maximum und Minimum). Seif:R⊇D→R. Dann heißtx 0 ∈D
eine

```
1)Maximalstelle (oderStelle eines Maximums), wennf(x 0 )≥f(x) f ̈ur allex∈D
ist. Der Wertf(x 0 ) ist der gr ̈oßte Funktionswert, denfaufDannimmt, und heißt
dasMaximumvonf.
Bezeichnung: maxx∈Df(x) oder nur maxf.
```
```
2)Minimalstelle(oderStelle eines Minimums), wennf(x 0 )≤f(x) f ̈ur allex∈Dist.
Der Wertf(x 0 ) ist der kleinste Funktionswert, denf aufDannimmt, und heißt
dasMinimumvonf.
Bezeichnung: minx∈Df(x) oder nur minf.
```
EinExtremumbezeichnet ein Maximum oder Minimum und eineExtremalstelleist eine
zugeh ̈orige Maximal- oder Minimalstelle.

Bemerkung 20.6.Plural: Maxima, Minima, Extrema.

Beispiel 20.7. 1) Seif: [− 1 ,2]→R,f(x) =x^2 −1. Das Maximum vonfist 3 (bei
der Maximalstelle 2) und das Minimum vonfist−1 (bei der Minimalstelle 0).


```
2)f: [− 2 ,2]→R,f(x) =x^2 −1. Das Maximum vonfist 3 (bei den Maximalstellen
−2 und 2) und das Minimum vonfist−1 (bei der Minimalstelle 0). Insbesondere
kann es mehrere Maximalstellen (oder Minimalstellen) geben, aber immer nur ein
Maximum und ein Minimum.
```
```
x
```
```
y
```
##### − 2 − 1 1 2

##### 1

##### 2

##### − 1

##### 0

##### 3

```
Maximum ist 3
```
```
Minimum ist− 1
```
```
x
```
```
y
```
##### − 2 − 1 1 2

##### 1

##### 2

##### − 1

##### 0

##### 3

```
Maximum ist 3
```
```
Minimum ist− 1
```
```
3) cos :R→Rhat das Maximum 1 und das Minimum−1 (die Maximalstellen sind
alle 2kπmitk∈Z, die Minimalstellen sind alle (2k+ 1)πmitk∈Z):
```
```
x
```
```
y
```
```
π π 2 π 3 π 4 π
2
3 π
2
```
```
5 π
2
```
```
7 π
2
− 1
```
##### 0

##### 1

```
Nicht jede Funktion besitzt ein Minimum und Maximum.
```
Beispiel 20.8. 1) Die Funktionf: [0,1]→R,f(x) =

##### {

```
x f ̈ur 0≤x < 1
0 f ̈urx= 1
```
```
, hat das
```
```
Minimum 0 (beix 0 = 0 undx 0 = 1), hat aber kein Maximum:
```
```
x
1
```
##### 1

##### 0

```
Die Funktion nimmt jeden Wert in [0,1[ an, so dass das Maximum, wenn es denn
existiert,≥1 sein muss. Diese Werte werden von der Funktion aber nicht ange-
nommen.
```
2) Auchf: ]0,∞[→R,f(x) =^1 x, hat kein Maximum: Da limx↘ (^0) x^1 = +∞ist, gibt
es keinx 0 ∈]0,∞[ so dassf(x 0 )≥f(x) fur alle ̈ x∈]0,∞[ gilt. Weiter hatfkein


```
Minimum: Es istf(x)≥0 f ̈ur allex∈]0,∞[ und limx→∞f(x) = 0, aber es gibt
keinx 0 mitf(x 0 ) =x^10 = 0.
Hingegen hatf: ]0,1]→R,f(x) =^1 xdas Minimum 1 (mit Minimalstellex 0 = 1),
aber kein Maximum.
```
```
3) Der Arcus Tanges arctan : R → ]−π 2 ,π 2 [ ist die Umkehrfunktion des Tangens
tan : ]−π 2 ,π 2 [→R(siehe Vorlesung 7):
```
```
x
```
```
y
y=π 2
```
```
y=−π 2
```
```
arctan
```
```
Kandidat f ̈ur ein Maximum von arctan istπ 2 (wegen limx→+∞arctan(x) =π 2 kann
das Maximum nicht kleiner als π 2 sein), aber der Wertπ 2 wird vom Tangens nicht
angenommen: es gibt keinx 0 ∈ Rmit arctan(x 0 ) = π 2. Daher hat arctan kein
Maximum. Genauso sieht man, dass arctan kein Minimum hat.
```
In allen Beispielen erreicht die Funktion den”Kandidaten f ̈ur das Maximum“ nicht:
Es gibt keinen Punkt im Definitionsbereich, an dem die Funktion den Wert erreicht.
Dafur n ̈ ̈ahert sich die Funktion diesem Wert beliebig nahe an.
Man definiert dasSupremumals diekleinste obere Schrankeder Funktion, d.h. man
sucht den kleinsten WertMmitf(x)≤Mf ̈ur allexim Definitionsbereich vonf. Analog
definiert man dasInfimumals diegr ̈oßte untere Schrankeder Funktion.

Definition 20.9(Infimum und Supremum). Seif:R⊇D→R.

```
1)y∗∈R∪{+∞}ist dasSupremumvonf, geschriebeny∗= supx∈Df(x) = supf,
wenn gilt:
```
```
(a)f(x)≤y∗f ̈ur allex∈D, d.h.y∗ist eine obere Schranke, und
(b) es gibt eine Folge (xn)ninDmit limn→∞f(xn) =y∗.
```
```
2)y∗∈R∪{−∞}ist dasInfimumvonf, geschriebeny∗= infx∈Df(x) = inff, wenn
gilt:
```
```
(a)f(x)≥y∗fur alle ̈ x∈D, d.h.y∗ist eine untere Schranke, und
(b) es gibt eine Folge (xn)ninDmit limn→∞f(xn) =y∗.
```
Dabei gilt fur das Supremum und das Infimum: Die Folge ( ̈ xn)nbraucht nicht zu kon-
vergieren.


Bemerkung 20.10. 1) Jede Funktionf:D→Rhat ein Supremum und ein Infi-
mum.

```
2) Nimmtfsein Supremum an, d.h. gibt es einx 0 ∈Dmitf(x 0 ) = supx∈Df(x), so
ist das Supremum vonfauch ein Maximum.
```
```
3) Nimmtfsein Infimum an, d.h. gibt es einx 0 ∈Dmitf(x 0 ) = infx∈Df(x), so ist
das Infimum vonf auch ein Minimum.
```
Beispiel 20.11. 1) Es sind

```
sup
x∈]0,∞[
```
##### 1

```
x
```
```
= +∞ und inf
x∈]0,∞[
```
##### 1

```
x
```
##### = 0,

```
und
sup
x∈]0,1]
```
##### 1

```
x
```
```
= +∞ und inf
x∈]0,1]
```
##### 1

```
x
```
```
= 1 = min
x∈]0,1]
```
##### 1

```
x
```
##### .

```
2) F ̈ur den Arcus Tanges ist supx∈Rarctan(x) =π 2 und infx∈Rarctan(x) =−π 2.
```
Eine besondere Eigenschaft von stetigen Funktionen ist, dass sie auf einem Intervall
[a,b] immer ein Minimum und Maximum besitzen.

Satz 20.12(Existenz vom Minimum und Maximum). Seif : [a,b]→Rstetig. Dann
gibt es Stellenxmin,xmax∈[a,b]mit

```
f(xmin)≤f(x)≤f(xmax) f ̈ur allex∈[a,b],
```
d.h.f nimmt auf[a,b]Minimum und Maximum an.

Insbesondere sind stetige Funktionen auf kompakten Intervallen beschr ̈ankt. K ̈urzer
kann man den Satz wie folgt formulieren.

Satz 20.13(Existenz vom Minimum und Maximum).Stetige Funktionen auf kompakten
Intervallen besitzen ein Minimum und ein Maximum.



Vorlesung 21

## 21 Differenzierbarkeit

### 21.1 Definition

Zur Motivation seif:I→Reine stetige Funktion auf dem IntervallIundx 0 ,x 1 ∈I.
Dann l ̈asst sich eine Gerade durch die Punkte (x 0 ,f(x 0 )) und (x 1 ,f(x 1 )) legen, eine
sogenannteSekante. Die Steigung der Sekante ist f(xx^11 )−−fx( 0 x^0 ). Die Sekante wird daher
durch

```
s(x) =f(x 0 ) +
f(x 1 )−f(x 0 )
x 1 −x 0
```
```
(x−x 0 )
```
beschrieben. F ̈urx 1 →x 0 geht die Sekante in dieTangente

```
t(x) =f(x 0 ) +m(x−x 0 )
```
uber (falls diese existiert). Die Steigung der Tangente ist ̈

```
m= lim
x 1 →x 0
```
```
f(x 1 )−f(x 0 )
x 1 −x 0
```
##### .

```
x
```
```
y
```
```
x 0 x 1
```
```
f(x 0 )
```
```
f(x 1 )
```
```
f(x 1 )−f(x 0 )
```
```
x 1 −x 0
```
```
Funktionf
```
```
Sekantes
```
```
Tangentet
```

Definition 21.1(Differenzierbarkeit).Seif:R⊇D→Reine Funktion.

```
1)f heißtdifferenzierbar inx 0 ∈D, falls
```
```
f′(x 0 ):= lim
x→x 0
```
```
f(x)−f(x 0 )
x−x 0
```
```
existiert.f′(x 0 ) heißtAbleitungvonfinx 0.
```
```
2)fheißtdifferenzierbar aufD, fallsfin allenx 0 ∈Ddifferenzierbar ist. Dann heißt
die Abbildung
f′:D→R, x7→f′(x),
dieAbleitungvonf.
```
DenUbergang von ̈ f zuf′nennt manableitenoderdifferenzieren.

Beachten Sie: Damit der Grenzwert fur ̈ f′(x 0 ) ̈uberhaupt definiert ist, muss es Folgen
inD\{x 0 }geben, die gegenx 0 konvergieren (vgl. Definition 19.1). In”isolierten Punkten“
kann man nicht ableiten. F ̈urD= ]− 3 ,2[∪{ 3 }ist z.B. 3 ein Punkt, in dem man nicht
ableiten kann.

Bemerkung 21.2. 1)Umformulierung der Definition:Mith:=x−x 0 ist

```
f′(x 0 ) = lim
h→ 0
```
```
f(x 0 +h)−f(x 0 )
h
```
##### .

```
2)Weitere Schreibweisen: Schreibt manxstatt x 0 und ∆xstatt hund ∆f(x) =
f(x+ ∆x)−f(x), so hat man die weiteren Schreibweisen
```
```
f′(x) = lim
∆x→ 0
```
```
f(x+ ∆x)−f(x)
∆x
= lim
∆x→ 0
```
```
∆f(x)
∆x
```
##### =

```
df(x)
dx
```
##### =

```
df
dx
(x).
```
```
Das ∆ wird oft als Differenz zweier Werte gedacht.
```
```
3) In der Physik wird oftf ̇(t) stattf′(t) geschrieben, wenntdie Zeit bezeichnet.
```
Beispiel 21.3. 1)f :R→R,f(x) =x^2 , ist differenzierbar mitf′(x) = 2x: F ̈ur
x 6 =x 0 ist

```
f(x)−f(x 0 )
x−x 0
```
##### =

```
x^2 −x^20
x−x 0
```
##### =

```
(x−x 0 )(x+x 0 )
x−x 0
=x+x 0 ,
```
```
also
f′(x 0 ) = lim
x→x 0
```
```
f(x)−f(x 0 )
x−x 0
```
```
= lim
x→x 0
(x+x 0 ) = 2x 0.
```
```
Dax 0 beliebig war, istf′:R→R,f′(x) = 2x.
```

```
2) Seif:R\{ 0 }→R,f(x) =^1 x. Dann istfdifferenzierbar mitf′(x) =−x^12 , denn
```
```
f′(x) = lim
h→ 0
```
```
1
x+h−
```
```
1
x
h
```
```
= lim
h→ 0
```
```
x−(x+h)
hx(x+h)
```
```
= lim
h→ 0
```
##### −

##### 1

```
x(x+h)
```
##### =−

##### 1

```
x^2
```
```
3) Seif:R→R,x7→|x|, dann istfinx 0 = 0 nicht differenzierbar, denn
```
```
f(0 +h)−f(0)
h
```
##### =

```
|h|
h
```
##### =

##### {

```
1 h > 0 ,
− 1 h < 0 ,
```
```
so dass der Grenzwert f ̈urh→0 nicht existiert. In allen anderen Punkten istf
aber differenzierbar:f′(x) = 1 fur ̈ x >0 undf′(x) =−1 fur ̈ x <0.
```
Als erste Beobachtung halten wir fest, dass differenzierbare Funktionen stetig sind.
Andersherum: Ist eine Funktion nicht stetig, brauchen wir gar nicht erst untersuchen,
ob sie differenzierbar ist.

Satz 21.4. Istf inx 0 differenzierbar, so istf inx 0 stetig.

Beweis. Dies folgt aus

```
f(x)−f(x 0 ) =
f(x)−f(x 0 )
x−x 0
```
```
(x−x 0 )→f′(x 0 )·0 = 0
```
f ̈urx→x 0 , also limx→x 0 f(x) =f(x 0 ).

### 21.2 Interpretation der Ableitung

Geometrische Interpretation: Die Tangente. Geometrisch ist der Differenzenquo-
tientf(xx)−−fx 0 (x^0 )die Steigung der Sekante durch die Punkte (x,f(x)) und (x 0 ,f(x 0 )), und

der Differentialquotientf′(x 0 ) = limx→x 0 f(xx)−−fx( 0 x^0 )ist die Steigung der Tangente an den
Graph vonfim Punkt (x 0 ,f(x 0 )). Die Tangente wird dabei durch die Gleichung

```
t(x) =f(x 0 ) +f′(x 0 )(x−x 0 )
```
beschrieben. Insbesondere sind differenzierbare Funktionen solche, deren Graph eine
Tangente besitzt, und die in diesem Sinne glatt sind. Funktionen mit Knickstellen (wie
der Absolutbetrag in 0) sind hingegen nicht differenzierbar.

Analytische Interpretation: Lineare Approximation. Anschaulich ist f(x) ≈
t(x) =f(x 0 ) +f′(x 0 )(x−x 0 ) f ̈urxin der N ̈ahe vonx 0 ist. Was bedeutet”≈“ genau?
F ̈ur denRestoderFehlergilt

```
R(x) =f(x)−t(x) =f(x)−f(x 0 )−f′(x 0 )(x−x 0 )
```

und damitR(x)→0 f ̈urx→x 0. Weiter ist

```
R(x)
x−x 0
```
##### =

```
f(x)−f(x 0 )−f′(x 0 )(x−x 0 )
x−x 0
```
##### =

```
f(x)−f(x 0 )
x−x 0
−f′(x 0 )
```
und fur ̈ x→x 0 ist dann

```
xlim→x
0
```
```
R(x)
x−x 0
```
##### = 0.

Der Fehler bei der Approximation wird also schneller klein alsx−x 0 , also schneller als
linear. Man kann die Differentiation daher auffassen als Approximation von Funktionen
durch lineare Funktionen^1 mit
”
schnell verschwindendem“ Fehler:

```
f(x) =f(x 0 ) +f′(x 0 )(x−x 0 ) +R(x), mit lim
x→x 0
```
```
R(x)
x−x 0
```
##### = 0.

Man kann sich auch leicht ̈uberlegen: Istm(x−x 0 ) +beine lineare Approximation von
finx 0 , so dass der FehlerR(x) undR(x)/(x−x 0 ) f ̈urx→x 0 gegen Null gehen, so ist
m=f′(x 0 ) undb=f(x 0 ):Die Tangente ist die beste lineare Approximation anfinx 0.
Mehr zur Approximation einer Funktion durch Polynome sehen wir in Abschnitt 24.1.

Physikalische Interpretation: Die Geschwindigkeit. Wir betrachten ein Teil-
chen, dass sich entlang einer Geraden bewegt. Es bezeichnetdie Zeit unds= s(t)
die Position zum Zeitpunktt. Diemittlere Geschwindigkeitin [t 0 ,t 1 ] ist

```
∆v=
∆s
∆t
```
##### =

```
s(t 1 )−s(t 0 )
t 1 −t 0
```
Um zu wissen, wie schnell das Teilchen zum Zeitpunktt 0 ist, betrachtet man immer
kleinere Zeitintervalle [t 0 ,t 1 ], und erh ̈alt dieMomentangeschwindigkeitzum Zeitpunkt
t 0 durch Grenz ̈ubergangt 1 →t 0 :

```
v(t 0 ) = lim
t 1 →t 0
```
```
s(t 1 )−s(t 0 )
t 1 −t 0
```
```
=s′(t 0 ).
```
### 21.3 Rechenregeln

Als erstes sammeln wir Rechenregeln f ̈ur die Differentiation.

Satz 21.5(Ableitungsregeln).Seienf,g:D→Rdifferenzierbar inx∈D. Dann gilt:

```
1) Linearit ̈at 1:(f(x) +g(x))′=f′(x) +g′(x)
```
```
2) Linearit ̈at 2:(cf(x))′=cf(x)′f ̈ur allec∈R.
```
(^1) Streng genommen istf(x) =ax+bnur f ̈urb= 0 eine lineare Funktion wie in Definition 15.1,
f ̈urb 6 = 0 eineaffin lineare Funktion. Aus historischen Gr ̈unden wird aber auch f ̈urb 6 = 0 von linearen
Funktionen gesprochen.


```
3) Produktregel:(f(x)g(x))′=f′(x)g(x) +f(x)g′(x).
```
```
4) Quotientenregel:
```
##### (

```
f(x)
g(x)
```
##### )′

```
=f
```
```
′(x)g(x)−f(x)g′(x)
g(x)^2 , fallsg(x)^6 = 0.
```
```
Insbesondere
```
##### (

```
1
g(x)
```
##### )′

```
=−g
```
```
′(x)
g(x)^2.
5) Kettenregel: Istg:D→Rinxdifferenzierbar undf :E→Ring(x)differen-
zierbar, so gilt(f(g(x)))′=f′(g(x))g′(x).
```
Wegen1)und2)bilden differenzierbare Funktionen einen Vektorraum, der ein Teil-
raum des Vektorraums der stetigen Funktionen ist (Satz 21.4). Ableiten ist eine lineare
Abbildung.

Beweis. Die Rechenregeln folgen leicht aus der Definition, wir geben zurUbung nur ein paar wichtige ̈
Schritte an. Fur die Summe ist ̈
f(x+h) +g(x+h)−(f(x) +g(x))
h =

```
f(x+h)−f(x)
h +
```
```
g(x+h)−g(x)
h →f
```
```
′(x) +g′(x) f ̈urh→ 0.
```
Fur die Produktregel rechnet man ̈

```
f(x+h)g(x+h)−f(x)g(x)
h =
```
```
f(x+h)g(x+h)−f(x)g(x+h)
h +
```
```
f(x)g(x+h)−f(x)g(x)
h
→f′(x)g(x) +f(x)g′(x) f ̈urh→ 0.
```
Fur die Kettenregel rechnet man ̈

```
f(g(x+h))−f(g(x))
h =
```
```
f(g(x+h))−f(g(x))
g(x+h)−g(x)
```
```
g(x+h)−g(x)
h →f
```
```
′(g(x))g′(x) f ̈urh→ 0.
```
Fur den ersten Quotienten haben wir dabei lim ̈ h→ 0 g(x+h) =g(x) verwendet (Stetigkeit vong).
Fur (1 ̈ /g)′nutzen wir die Kettenregel: DieAußere Funktion ist 1 ̈ /xmit Ableitung− 1 /x^2 , die
Innere Funktion istg. Anschließend folgt die Quotientenregel aus (f(x)/g(x))′= (f(x)g(^1 x))′mit der
Produktregel und der Regel fur (1 ̈ /g)′.

Bemerkung 21.6.Merken Sie sich die Kettenregel alsf′(g(x))g′(x) (erstf′, danng′),
so herum stimmt es ebenfalls fur Funktionen von mehreren Variablen in der ̈
”
Analysis
II f ̈ur Ingenieurwissenschaften“. Dann kennen Sie die Regel schon.

Beispiel 21.7. 1) Es istf :R→R,f(x) =ax+b, differenzierbar mitf′(x) =a,
denn
f(x+h)−f(x)
h

##### =

```
(a(x+h) +b)−(ax+b)
h
```
##### =

```
ah
h
=a→a.
```
```
2) F ̈urn∈Nistf:R→R,f(x) =xn, differenzierbar mitf′(x) =nxn−^1.
Das kann man per Induktion und Produktregel nachrechnen (n= 0 undn= 1 haben wir eben
nachgerechnet) oder direkt, ganz ̈ahnlich wie f ̈urx^2 in Beispiel 21.3: Mit der binomischen Formel
ist
(x+h)n−xn
h =
```
```
xn+nxn−^1 h+∑nk=2
(n
k
```
```
)
xn−khk−xn
h =nx
```
n− (^1) +
∑n
k=2
(
n
k
)
xn−khk−^1
→nxn−^1 f ̈urh→ 0.


3)Polynome sind differenzierbar: p :R →R,p(x) = a 0 +a 1 x+...+anxn, ist
differenzierbar als Linearkombination der differenzierbaren Funktionenxn.

4)Rationale Funktionen sind differenzierbar:f:R→R,f(x) =pq((xx)), mit Polynomen
p,qist differenzierbar inD=R\{x|q(x) = 0}nach der Quotientenregel.

5)Die Exponentialfunktion ist differenzierbar:exp′(x) = exp(x). Das rechnen wir in
Vorlesung 26 nach.

6)Sinus und Cosinus sind differenzierbar:

```
sin′(x) = cos(x),
cos′(x) =−sin(x).
```
```
Fur die Ableitungen von Sinus und Cosinus ben ̈ ̈otigen wir die Grenzwerte
```
```
xlim→ 0 sin(xx)= 1 und xlim→ 0 cos(xx)−^1 = 0.
Den ersten kennen wir aus Beispiel 19.5, den zweiten erhalten wir als
cos(x)− 1
x =
```
```
(cos(x)−1)(cos(x) + 1)
x(cos(x) + 1) =
```
```
cos(x)^2 − 1
x(cos(x) + 1)=
```
```
−sin(x)^2
x(cos(x) + 1)
=sin(xx)cos(−sin(x) + 1x) → 1 ·− 20 = 0.
```
```
Fur die Ableitung des Sinus rechnen wir (mit den Additionstheoremen) ̈
sin(x+h)−sin(x)
h =
```
```
sin(x) cos(h) + sin(h) cos(x)−sin(x)
h
= sin(x)cos(hh)−^1
︸ ︷︷ ︸
→ 0
```
```
+sin(hh)
︸︷︷︸
→ 1
```
```
cos(x)→cos(x),
```
```
also sin′(x) = cos(x), und genauso f ̈ur den Cosinus
cos(x+h)−cos(x)
h =
```
```
cos(x) cos(h)−sin(x) sin(h)−cos(x)
h
= cos(x)cos(hh)−^1 −sin(x)sin(hh)→−sin(x),
```
```
also cos′(x) =−sin(x).
```
7) Wir leitenf(x) = sin(x^2 ) mit der Kettenregel ab: sin(t) ist die ̈außere Funktion,
x^2 die innere Funktion, also ist

```
f′(x) = sin′(x^2 )·(x^2 )′= cos(x^2 )· 2 x.
```
```
F ̈urf(x) = (sin(x))^2 istt^2 die ̈außere Funktion, sin(x) die innere Funktion, also
```
```
f′(x) = 2 sin(x) sin′(x) = 2 sin(x) cos(x).
```

## Vorlesung 22

## 22 Erste Anwendungen der Differenzierbarkeit

# Differenzierbarkeit

### 22.1 Ableitung der Umkehrfunktion

Als erste Anwendung leiten wir aus den Rechenregeln eine Formel zur Ableitung der
Umkehrfunktion her.
Istfumkehrbar und differenzierbar mitf′(x) 6 = 0, so ist auch die Umkehrabbildung
f−^1 differenzierbar.

Satz 22.1(Ableitung der Umkehrfunktion). SeienIundJ Intervalle. Seif:I→J
differenzierbar und umkehrbar mitf′(x) 6 = 0f ̈urx∈I. Dann ist auchf−^1 :J →I
differenzierbar mit

```
(f−^1 )′(x) =
```
##### 1

```
f′(f−^1 (x))
```
##### .

Beweis. Die Differenzierbarkeit vonf−^1 ist nicht ganz einfach zu zeigen und wir ver-
zichten auf diesen Teil. Wir rechnen nur die Formel f ̈ur (f−^1 )′nach: Es gilt

```
x=f(f−^1 (x)).
```
Differentiation und Anwendung der Kettenregel ergeben

```
1 =f′(f−^1 (x))(f−^1 )′(x),
```
also
(f−^1 )′(x) =

##### 1

```
f′(f−^1 (x))
```
##### ,

wobei nat ̈urlichf′(f−^1 (x)) 6 = 0 sein muss.

Beispiel 22.2(Naturlicher Logarithmus) ̈ .Der nat ̈urliche Logarithmus ln : ]0,∞[→R
ist die Umkehrfunktion von exp :R→]0,∞[, f ̈ur die exp′(x) = exp(x) gilt. Mit der
Formel fur die Ableitung der Umkehrfunktion ist ̈

```
ln′(x) =
```
##### 1

```
exp′(ln(x))
```
##### =

##### 1

```
exp(ln(x))
```
##### =

##### 1

```
x
```
##### .


Beispiel 22.3(Wurzeln).Die Funktionf: ]0,∞[→]0,∞[,x7→xn, ist differenzierbar
mit Ableitungf′(x) =nxn−^1. Wir werden sp ̈ater nachweisen, dass sie auch umkehrbar
ist. F ̈ur ihre Inverse, dien-te Wurzel

```
f−^1 : ]0,∞[→]0,∞[, x7→ n
```
##### √

```
x,
```
gilt

```
(f−^1 )′(x) = (n
```
##### √

```
x)′=
```
##### 1

```
f′(n
```
##### √

```
x)
```
##### =

##### 1

```
nn
```
##### √

```
xn−^1
```
##### .

Speziell f ̈ur die Quadratwurzel (n= 2) ist

```
(
```
##### √

```
x)′=
```
##### 1

##### 2

##### √

```
x
```
Beachten Sie, dass das nur auf ]0,∞[ geht, inx= 0 ist dien-te Wurzel nicht differen-
zierbar, wohl aber stetig.

### 22.2 Nullstellen

Wie bei Stetigkeit wenden wir uns der Aufgabe zu, Nullstellen zu bestimmen. Wir ler-
nen das sogenannteNewton-Verfahrenzur numerischen Approximation von Nullstellen
kennen. Wie beim Bisektionsverfahren (Abschnitt 20.1) wird eine Folge konstruiert, die
gegen die Nullstelle konvergiert.

Newton-Verfahren. W ̈ahle einen Startwertx (^0) ”nahe“ der Nullstellex∗vonf. Ist
xn gegeben, konstruieren wir xn+1 wie folgt: Wir approximierenf inxndurch ihre
Tangente,
f(x)≈f(xn) +f′(xn)(x−xn),
und bestimmenxn+1als Nullstelle der Tangente, also aus
f(xn) +f′(xn)(xn+1−xn) = 0.
x
y Funktion
Tangente inxn
xn+1 xn


Aufl ̈osen ergibt

```
xn+1=xn−
```
```
f(xn)
f′(xn)
, n= 1, 2 , 3 ,...
```
Dabei setzen wir voraus, dassf′(xn) 6 = 0, und auch dassf′(x∗) 6 = 0. Unter gewissen
Voraussetzungen konvergiert die Folge (xn)ngegen eine Nullstelle vonf. N ̈aheres k ̈onnen
Sie in der”Numerik I fur Ingenieurwissenschaften“ lernen. ̈

Beispiel 22.4. Seif(x) =x^2 −amit gegebenema >0. Die Nullstellen sind nat ̈urlich
±

##### √

```
a, und es istf′(x) = 2x. Damit ergibt sich die Folge von N ̈aherungen an die Nullstelle:
```
```
xn+1=xn−
x^2 n−a
2 xn
```
##### =

```
x^2 n+a
2 xn
```
##### =

##### 1

##### 2

##### (

```
xn+
a
xn
```
##### )

##### .

Dies ist genau die Wurzelfolge aus Beispiel 18.10.
Zur Illustration betrachten wir a = 3. Was ist

##### √

3 =? W ̈ahle x 0 = 2 als erste
N ̈aherung. Dann ist

```
x 1 =
```
##### 1

##### 2

##### (

##### 2 +

##### 3

##### 2

##### )

##### = 1 +

##### 3

##### 4

##### = 1, 75 ,

schon die erste Nachkommastelle ist korrekt. Die weiteren Folgenglieder sind:

```
x 0 = 2. 000000000000000
x 1 = 1. 750000000000000
x 2 = 1. 732142857142857
x 3 = 1. 732050810014727
x 4 = 1. 732050807568877 ,
```
wobei korrekte Stellen unterstrichen sind. Schon nach vier Schritten sind 15 Nachkom-
mastellen korrekt berechnet.

### 22.3 H ̈ohere Ableitungen

Istf :R⊇D→Rauf ganzDdifferenzierbar, so istf′:D→R,x7→f′(x), wieder
eine Funktion. Istf′differenzierbar, so schreiben wirf′′= (f′)′f ̈ur diezweite Ableitung
vonf. Diese kann wieder stetig oder differenzierbar sein, usw. Wir definieren diek-te
Ableitung vonfper Induktion:

```
f(0):=f, f(k):= (f(k−1))′ f ̈urk≥ 1.
```
Die 0-te Ableitung ist die Funktion selbst,f(0)=f, und es sindf(1)=f′,f(2)= (f′)′=
f′′, usw. Beachten Sie: Die Ordnung der Ableitung wird dabei in runden Klammern ge-
schrieben, um Missverst ̈andnisse zu vermeiden. Eine weitere oft verwendete Schreibweise
ist
dkf
dxk

```
:=f(k).
```
Im Fallk= 1 schreiben wir nurdfdx, vergleiche Bemerkung 21.2.


Bemerkung 22.5.Interpretation der zweiten Ableitung:

- Geometrische Interpretation:f′′(x) beschreibt dieKr ̈ummungdes Funktionsgra-
    phen vonfinx. Punkte, in denenf′′das Vorzeichen ̈andert, heißenWendepunkte.
- Physikalische Interpretation: Die zweite Ableitung ist die Ableitung der Geschwin-
    digkeit, also die Beschleunigung.
Nicht jede differenzierbare Funktion ist auch zweimal differenzierbar.

Beispiel 22.6.Die Funktion

```
f:R→R, f(x) =
```
##### {

```
x^2 , x≥ 0 ,
−x^2 , x < 0 ,
```
ist differenzierbar mit

```
f′(x) =
```
##### {

```
2 x x≥ 0
− 2 x x < 0
```
```
, also f′(x) = 2|x|.
```
Daher istf′nicht in 0 differenzierbar.

##### − 1 0 1

##### − 2

##### 0

##### 2

```
f(x) =
```
##### {

```
x^2 , x≥ 0
−x^2 , x < 0
```
##### − 1 0 1

##### 0

##### 1

##### 2

##### 3

```
f′(x) = 2|x|
```
### 22.4 Regel von Bernoulli/de l’Hospital

Die Regel von Bernoulli/de l’Hospital hilft bei der Berechnung von Grenzwerten von
Quotientenf/g.

Satz 22.7(Regel von Bernouilli/de l’Hospital). Seienf,g: ]a,b[→Rdifferenzierbar,
wobei−∞≤a < b≤∞. Weiter gelte

```
lim
x→b
f(x) = lim
x→b
g(x) = 0 oder lim
x→b
g(x)∈{−∞,∞}.
```
Fallslimx→bf

```
′(x)
g′(x)∈R∪{−∞,∞}existiert (undg
```
```
′(x) 6 = 0f ̈ur allexnaheb), dann ist
```
```
lim
x→b
```
```
f(x)
g(x)
= lim
x→b
```
```
f′(x)
g′(x)
```
##### .

Gleiches gilt f ̈ur Grenzwertex→a.


Bemerkung 22.8. 1) Es kommt vor, dass man den Grenzwert limx→bf

```
′(x)
g′(x)auch erst
mit der Regel von l’Hospital berechnet, die Regel also mehrfach anwendet.
2) Bei Grenzwerten” 0 ·∞“ kann man oft umformen und anschließend die Regel von
l’Hospital anwenden:
f(x)g(x) =
f(x)
1
g(x)
```
##### .

Beispiel 22.9. 1) Was ist limx→ 0 sin(xx)? Hier gilt

```
lim
x→ 0
sin(x) = 0 und lim
x→ 0
x= 0,
```
```
so dass die erste Bedingung gegeben ist. Weiter ist
```
```
lim
x→ 0
```
```
sin′(x)
(x)′
```
```
= lim
x→ 0
```
```
cos(x)
1
```
##### = 1,

```
also auch
lim
x→ 0
```
```
sin(x)
x
```
##### = 1.

```
2) Wir bestimmen den Grenzwert limx→∞sin(xx). Hier ist limx→∞x=∞, aber
```
```
xlim→∞
```
```
sin′(x)
(x)′
= limx→∞
cos(x)
1
existiert nicht. Das heißt aber nicht, dass der urspr ̈ungliche Grenzwert nicht exis-
tiert, sondern nur, dass die Regel von l’Hospital nicht angewendet werden kann.
```
Da sin(x) beschr ̈ankt ist und limx→ (^01) x= 0, ist auch limx→∞sin(xx)= 0.
3) Wir bestimmen den Grenzwert limx→ 01 −cos(x 2 x). Z ̈ahler und Nenner konvergieren
gegen 0, und
lim
x→ 0
(1−cos(x))′
(x^2 )′
= lim
x→ 0
sin(x)
2 x

##### .

```
Wieder konvergieren Z ̈ahler und Nenner gegen 0, und
```
```
lim
x→ 0
```
```
sin′(x)
(2x)′
```
```
= lim
x→ 0
```
```
cos(x)
2
```
##### =

##### 1

##### 2

##### ,

```
also ist limx→ 01 −cos(x 2 x)=^12 nach zweimaliger Anwendung der Regel von l’Hospital.
4) Wir bestimmen den Grenzwert limx→ 0 xln(x). Hier istx→0 und ln(x)→ −∞,
so dass wir nicht direkt die Regel von l’Hospital anwenden k ̈onnen. Jedoch ist
```
```
xln(x) =
ln(x)
1 /x
und nun sind die Grenzwerte von Z ̈aher und Nenner±∞. Zudem ist
```
```
lim
x→ 0
```
```
ln′(x)
(1/x)′
= lim
x→ 0
```
```
1
x
−x^12
```
```
= lim
x→ 0
−x= 0,
```
```
also auch limx→ 0 xln(x) = 0.
```

5) F ̈ura∈Rist

```
nlim→∞
```
##### (

##### 1 +

```
a
n
```
```
)n
=ea.
```
```
Auch hier kann man die Regel von l’Hospital gewinnbringend einsetzen, allerdings
mit einem Trick. Fur ̈ n >|a|ist
(
1 +
a
n
```
```
)n
= exp
```
##### (

```
ln
```
##### ((

##### 1 +

```
a
n
```
```
)n))
= exp
```
##### (

```
nln
```
##### (

##### 1 +

```
a
n
```
##### ))

##### ,

```
also, da exp stetig ist,
```
```
lim
n→∞
```
##### (

##### 1 +

```
a
n
```
```
)n
= lim
n→∞
exp
```
##### (

```
nln
```
##### (

##### 1 +

```
a
n
```
##### ))

```
= exp
```
##### (

```
lim
n→∞
nln
```
##### (

##### 1 +

```
a
n
```
##### ))

##### .

```
Nun kommt der Trick: Wir ersetzenn∈Ndurchx ∈Rund betrachten einen
Grenzwert von Funktionen, fur den wir die Regel von l’Hospital wie im letzten ̈
Beispiel anwenden k ̈onnen:
```
```
lim
x→∞
xln
```
##### (

##### 1 +

```
a
x
```
##### )

```
= lim
x→∞
```
```
ln
```
##### (

```
1 +xa
```
##### )

```
1 /x
```
```
= lim
x→∞
```
```
1
1+ax
```
```
−a
x^2
−x^12
```
```
= lim
x→∞
```
```
a
1 +xa
```
```
=a.
```
```
Daher ist auch limn→∞nln
```
##### (

```
1 +an
```
##### )

```
=aund damit
```
```
lim
n→∞
```
##### (

##### 1 +

```
a
n
```
```
)n
= exp
```
##### (

```
lim
n→∞
nln
```
##### (

##### 1 +

```
a
n
```
##### ))

```
= exp(a).
```

Vorlesung 23

## 23 Mittelwertsatz und Anwendungen

Wir wenden uns nun der Aufgabe zu, Extremwerte und Extremstellen von differenzier-
baren Funktionen zu bestimmen. F ̈ur stetige Funktionen auf einem kompakten Intervall
konnten wir immerhin deren Existenz garantieren. Nun lernen wir mit Hilfe der Diffe-
rentialrechnung Methoden kennen, um Extremstellen zu charakterisieren.
Dabei hilft der Mittelwertsatz, einer der zentralen S ̈atze ̈uber differenzierbare Funk-
tionen.

### 23.1 Extremwerte

Die Extremwerte einer Funktion bezeichnen die gr ̈oßten und kleinsten Werte, die die
Funktion annimmt. Das Maximum ist der gr ̈oßte angenommene Wert, das Minimum
ist der kleinste angenommene Wert; vergleiche Definition 20.5. Unter einem Extremum
versteht man ein Minimum oder Maximum. (Plural: Maxima, Minima und Extrema.)
In Abschnitt 20.2 haben wir gesehen, dass eine stetige Funktionf : [a,b]→Rimmer
Minimum und Maximum besitzt (Satz 20.12).
Mit der Differentialrechnung lassen sich Extremalstellen mit Hilfe der Ableitung be-
stimmen. Da die Ableitungf′(x 0 ) nur von den Werten vonfnahex 0 abh ̈angt, ̈uberrascht
es vielleicht nicht, dass man nicht nur die Extrema der Funktion findet, sondern auch
solche Punkte, die in einer kleinen Umgebung wie Extrema sind, z.B. die markierten
Punkte:

```
x
```

Dies sind so genanntelokaleExtremwerte (siehe Definition 23.1 unten). Zur Unterschei-
dung nennt man das Minimum und Maximum dann globales Minimum undglobales
Maximum, was wir gleich mit wiederholen.

Definition 23.1(Lokale und globale Extrema). Seif:D→RmitD⊆R. Man nennt
einx 0 ∈D

```
1)Stelle eines globalen Maximums, fallsf(x 0 )≥f(x) fur alle ̈ x∈D.
Man nennt dannf(x 0 )das globale Maximum.
```
```
2)Stelle eines globalen Minimums, fallsf(x 0 )≤f(x) fur alle ̈ x∈D.
Man nennt dannf(x 0 )das globale Minimum.
```
```
3)Stelle eines lokalen Maximums, falls esε >0 gibt, so dass f ̈ur allex∈Dmit
|x−x 0 |< εgiltf(x 0 )≥f(x).
Man nennt dannf(x 0 )ein lokales Maximum.
```
```
4)Stelle eines lokalen Minimums, falls esε >0 gibt, so dass fur alle ̈ x∈Dmit
|x−x 0 |< εgiltf(x 0 )≤f(x).
Man nennt dannf(x 0 )ein lokales Minimum.
```
Gilt sogar>statt≥bzw.<statt≤, so spricht man vonstrengenoderstriktenlokalen
oder globalen Extrema.

```
lokales Maximum
```
```
globales Maximum
```
```
lokales Minimum
```
```
globales Minimum
```
Das globale Maximum ist der gr ̈oßte Wert, den die Funktion auf ihrem Definitionsbereich
annimmt. Dieser Wert ist eindeutig, kann aber an mehreren Stellen angenommen werden.
Fur ein lokales Maximum reicht es, dass die Funktion in einer kleinen Umgebung kleiner ̈
als dieser Wert ist. Das globale Maximum ist auch ein lokales Maximum. Entsprechendes
gilt f ̈ur globale und lokale Minima.


Nimmt die Funktionf: [a,b]→Rin einem inneren Punktx 0 ∈]a,b[ ein Maximum
an, dann haben die Sekanten links davon eine Steigung ≥ 0 und rechts davon eine
Steigung≤0.

```
x
```
```
y
```
```
Funktion
```
```
x 0
```
```
f(x 0 )
```
Istfdann inx 0 differenzierbar, so ist die Ableitungf′(x 0 ) der Grenzwert der Stei-
gungen der Sekanten, also einerseits≥0, andererseits≤0, und deshalb istf′(x 0 ) = 0.
Wir notieren dieses Resultat.

Satz 23.2(Notwendiges Extremwertkriterium).Sei f:D→Rim inneren Punktx 0
differenzierbar. Wenn x 0 eine lokale Extremstelle ist, so ist die Ableitung dort Null:
f′(x 0 ) = 0.

```
In Randpunkten muss das nicht sein.
```
Bemerkung 23.3.Das ist nur einenotwendigeBedingung, d.h.

- in einem lokalen Extremum istf′(x 0 ) = 0,
- aberf′(x 0 ) = 0 kann gelten, ohne dass inx 0 ein lokales Extremum vorliegt. Ein
    Beispiel istf: [− 1 ,1]→R,f(x) =x^3. Dann hatf′(x) = 3x^2 eine Nullstelle in 0,
    dort hatfaber kein lokales Extremum.

```
x
```
```
y
f(x) =x^3
```
Kandidaten f ̈ur lokale Extremstellen. Istf: [a,b]→Rdifferenzierbar, so sind die
einzigen Kandidaten f ̈ur lokale Extremstellen:

- die Randpunkteaundb,
- die Nullstellen vonf′in ]a,b[. (Die Nullstellen vonf′heißen auchkritische Punkte
    oderstation ̈are Punktevonf.)
Typischerweise sind das nur endlich viele Punkte, und man kann ausprobieren, wofam
gr ̈oßten oder kleinsten ist. Dort sind dann wirklich das globale Maximum und Minimum,
dafstetig auf einem abgeschlossenen Intervall ist (siehe Satz 20.12).

```
Istfim Punktx 0 nichtdifferenzierbar, so kann dort ebenfalls ein lokales Extremum vorliegen.
```

Beispiel 23.4. 1) f:R→R,f(x) =|x|, hat inx 0 = 0 ein lokales Minimumf(0) = 0 (das sogar
das globale Minimum ist). Allerdings existiert die Ableitung vonf(x) =|x|inx 0 = 0 nicht, so
dass wir die lokale Extremstelle nicht ̈uber die Bedingungf′(x) = 0 finden k ̈onnen.
2) Die Funktion
f:R→R, f(x) =

```
{
x f ̈urx < 0
x/2 f ̈urx≥ 0 ,
ist inx 0 = 0 nicht differenzierbar. Daher istx 0 ein Kandidat f ̈ur eine Extremstelle, aberfhat
dort kein lokales Extremum:
```
```
x
```
```
y
```
### 23.2 Mittelwertsatz

Der folgende Satz ist von fundamentaler Wichtigkeit in der Differentialrechnung.

Satz 23.5 (Mittelwertsatz). Sei f :I→Rdifferenzierbar auf dem IntervallI⊆ R.
Sinda,b∈Ibeliebig mita < b, so gibt es mindestens eine Stelleξ∈]a,b[mit

```
f(b)−f(a)
b−a
=f′(ξ).
```
Anschaulich bedeutet der Mittelwertsatz, dass es irgendwo eine Tangente gibt, die
die gleiche Steigung wie die Sekante durch (a,f(a)) und (b,f(b)) besitzt.

Beweis. Wir betrachten die Differenz zwischenfund der Sekante:g: [a,b]→R,

```
g(x) =f(x)−
```
```
(
f(a) +f(b)b−−fa(a)(x−a)
```
```
)
.
```
Dann sindg(a) = 0 undg(b) = 0. Dagstetig auf [a,b] ist, hatgmindestens einen Extremwertξin ]a,b[.
Dagaber in ]a,b[ differenzierbar ist (als Differenz differenzierbarer Funktionen), ist danng′(ξ) = 0. Da
g′(x) =f′(x)−f(bb)−−fa(a)giltf′(ξ) =f(bb)−−fa(a), wie behauptet.

```
x
```
```
y
f
```
```
Tangente
```
```
Tangente
```
```
Sekante
```
```
a b
```
```
f(a)
```
```
f(b)
```
```
ξ 1 ξ 2
```

Bezeichnets(t) den zum Zeitpunkttzur ̈uckgelegten Weg, so besagt der Mittelwert-
satz, dass es einen Zeitpunktτgibt, zu dem die Momentangeschwindigkeit gleich der
Durchschnittsgeschwindigkeit ist:

```
v(τ) =s′(τ) =
```
s(b)−s(a)
b−a
Im Mittelwertsatz ist die Stelleξzun ̈achst nicht bekannt, man weiß nur, dass es sie
gibt. Viele Anwendungen des Mittelwertsatzes betreffen aber Situationen, wo man die
Ableitung ̈uberall gut kennt und damit R ̈uckschlusse auf die Funktion selbst zieht. ̈ Der
Mittelwertsatz ist die Br ̈ucke von Ableitungsinformationen zu Informationen ̈uber die
Funktion selbst.

### 23.3 Anwendungen des Mittelwertsatzes

Eine der wichtigsten Anwendungen des Mittelwertsatzes ist das Monotoniekriterium.

Satz 23.6(Monotoniekriterium). Seif: [a,b]→Rstetig und auf]a,b[differenzierbar.
Dann gilt:

```
1) f′(x)> 0 f ̈ur allex∈]a,b[⇒fist streng monoton wachsend auf[a,b].
```
```
2) f′(x)< 0 f ̈ur allex∈]a,b[⇒fist streng monoton fallend auf[a,b].
```
```
3) f′(x)≥ 0 f ̈ur allex∈]a,b[⇔fist monoton wachsend auf[a,b].
```
```
4) f′(x)≤ 0 f ̈ur allex∈]a,b[⇔fist monoton fallend auf[a,b].
```
Beweis. 1)Seienx 1 ,x 2 ∈ [a,b] mitx 1 < x 2. Nach dem Mittelwertsatz existiertξ ∈
]x 1 ,x 2 [ mit
f(x 2 )−f(x 1 ) =f′(ξ) (x 2 −x 1 )
︸ ︷︷ ︸
> 0

##### > 0 ,

alsof(x 2 )> f(x 1 ), so dassfstreng monoton wachsend ist.2)und”⇒“ in3)und4)
gehen genauso. Die Ruckrichtung sieht man direkt mit der Definition: Ist ̈ f monoton
wachsend undx∈]a,b[, so giltf(xx^22 )−−fx(x)≥0 f ̈ur allex 2 ∈[a,b] mitx 26 =x, also

```
f′(x) = limx
2 →x
```
```
f(x 2 )−f(x)
x 2 −x
```
##### ≥ 0.

Genauso f ̈ur monoton fallendesf.

Satz 23.7 (Konstanzkriterium).Seif: [a,b]→Rstetig und auf]a,b[differenzierbar.
Dann gilt:f′(x) = 0f ̈ur allex∈]a,b[⇔f ist konstant auf[a,b].

Beweis.
”
⇐“ ist klar. Zu
”
⇒“: Wennf′(x) = 0 fur alle ̈ x∈]a,b[ ist, dann ist auch
f′(x)≥0 undf′(x)≤0, also istf zugleich monoton wachsend and monoton fallend,
und damit konstant.


Bemerkung 23.8. 1) Istfin einem Randpunktaoderb(oder in beiden Randpunk-
ten) nicht definiert, so gilt der Satz immer noch, wenn man den betroffen Punkt
aus [a,b] entfernt. Ist z.B.fnicht inadefiniert, so ersetzt man [a,b] durch ]a,b].

```
2) In1)und2)gilt nur die Richtung
”
⇒“. Die andere Richtung ist im Allgemeinen
falsch. Beispiel: Die Funktionf:R→R,f(x) =x^3 , ist streng monoton wachsend,
aberf′(0) = 0.
```
Beispiel 23.9. Wir zeigen, dass der Tangens tan =cossin streng monoton wachsend auf
]−π 2 ,π 2 [ ist. Das kann man mit der Definition versuchen, dann muss man zeigen, dass

```
x < y ⇒
sin(x)
cos(x)
```
##### <

```
sin(y)
cos(y)
```
gilt. Einfacher ist es mit dem Monotoniekriterium. Mit der Quotientenregel findet man

```
tan′(x) =
sin′(x) cos(x)−sin(x) cos′(x)
cos(x)^2
```
##### =

```
cos(x)^2 + sin(x)^2
cos(x)^2
```
##### =

##### 1

```
cos(x)^2
```
##### > 0 ,

also ist tan streng monoton wachsend.

```
x
```
```
y
```
```
tan
```
```
−π 2 π 2
```
```
Aus dem Monotoniekriterium erhalten wir einen ersten Test f ̈ur Extremwerte.
```
Satz 23.10(Extremwert-Test). Seif differenzierbar im offenen Intervall]a,b[und sei
x 0 ein kritischer Punkt, alsof′(x 0 ) = 0. Dann gilt:

```
1)f hat inx 0 ein lokales Maximum, falls einε > 0 existiert mit:f′(x)> 0 fur alle ̈
x∈]x 0 −ε,x 0 [undf′(x)< 0 f ̈ur allex∈]x 0 ,x 0 +ε[.
```
```
2)f hat inx 0 ein lokales Minimum, falls einε > 0 existiert mit:f′(x)< 0 f ̈ur alle
x∈]x 0 −ε,x 0 [undf′(x)> 0 f ̈ur allex∈]x 0 ,x 0 +ε[.
```
Beweis. 1)besagt, dassf links vonx 0 w ̈achst, und rechts vonx 0 wieder f ̈allt, also hat
finx 0 ein Maximum. F ̈ur lokale Minima geht es genauso.


```
lokales Maximum
f′> 0 f′< 0
```
```
x 0 −ε x 0 x 0 +ε
```
Bemerkung 23.11. 1) Istf′(x)>0 f ̈ur allexlinksundrechts vonx 0 (bzw.f′(x)<
0), so istfstreng monoton wachsend (bzw. fallend) und hat damitkeinExtremum.
2) Der Test gibt nur fur innere Punkte Auskunft. Randpunkte m ̈ ̈ussen immer getrennt
betrachtet werden.
3) Istfinanoch stetig, so gilt:

- Istf′(x)>0 in ]a,a+ε[, so istfdort monoton wachsend undfhat inaein
    lokales Minimum.
- Istf′(x)<0 in ]a,a+ε[, so istf dort monoton fallend undf hat inaein
    lokales Maximum.
Analog wennfinbnoch stetig ist:
- Istf′(x)>0 in ]b−ε,b[, so istfdort monoton wachsend undf hat inbein
lokales Maximum.
- Istf′(x)<0 in ]b−ε,b[, so istf dort monoton fallend undf hat inbein
lokales Minimum.
In Satz 24.6 werden wir einen weiteren Test f ̈ur Extremwerte kennen lernen, der auf
h ̈oheren Ableitungen der Funktion beruht.

Beispiel 23.12. Bestimme alle lokalen und globalen Extremwerte vonf: [− 1 ,2]→R,
f(x) = 2x^3 − 3 x^2 + 1.
Wir suchen zuerst nach kritischen Punkten in ]− 1 ,2[. Es ist

```
f′(x) = 6x^2 − 6 x= 6x(x−1),
```
somit hatfdie beiden kritischen Punkte 0 und 1. Aus der Faktorisierung vonf′k ̈onnen
wir das Vorzeichen vonf′ablesen:

```
x − 1 0 1 2
f′(x) + 0 − 0 +
```
Neben den kritischen Punkten sind auch die Randpunkte Kandidaten f ̈ur lokale Extrema.
Mit dem Vorzeichen der ersten Ableitung erhalten wir:fhat

- in 0 ein lokales Maximum mitf(0) = 1,
- in 1 ein lokales Minimum mitf(1) = 0.
- in−1 ein lokales Minimum mitf(−1) =−4,
- in 2 ein lokales Maximum mitf(2) = 5.
Somit hatfdas globale Maximum 5 an der Stellex= 2 und das globale Minimum− 4
an der Stellex=−1.



Vorlesung 24

## 24 Taylor-Approximation

### 24.1 Die Taylor-Approximation

Wir haben bisher gesehen, dass wir eine differenzierbare Funktion durch ihre Tangente
ann ̈ahern k ̈onnen,
f(x)≈f(x 0 ) +f′(x 0 )(x−x 0 ).

Das liefert die Approximation vonfdurch das Polynom

```
p(x):=f(x 0 ) +f′(x 0 )(x−x 0 )
```
vom Grad≤1 mitp(x 0 ) = 0 undp′(x 0 ) =f′(x 0 ). Der FehlerR(x) =f(x)−p(x) wird

dabei schneller klein als linear, limx→x (^0) xR−(xx) 0 = 0.
Hoffnung: Durch hinzunahme weiterer Ableitungen erhalten wir eine bessere Appro-
ximation. Istfdannn-mal differenzierbar, so wollen wirfdurch ein Polynom
p(x) =a 0 +a 1 (x−x 0 ) +a 2 (x−x 0 )^2 +...+an(x−x 0 )n
vom Grad≤napproximieren mit
p(k)(x 0 ) =f(k)(x 0 ), k= 0, 1 ,...,n,
d.h. die ersten Ableitungen vonpundfsollen inx 0 ̈ubereinstimmen.
Dann gilt notwendigerweise:
p(x) =a 0 +a 1 (x−x 0 ) +a 2 (x−x 0 )^2 +a 3 (x−x 0 )^3 +...+an(x−x 0 )n
p′(x) =a 1 + 2a 2 (x−x 0 )^1 + 3a 3 (x−x 0 )^2 +...+nan(x−x 0 )n−^1 ,
p′′(x) = 2!a 2 + 3· 2 a 3 (x−x 0 ) +...+n(n−1)an(x−x 0 )n−^2 ,
p′′′(x) = 3!a 3 +...+n(n−1)(n−2)an(x−x 0 )n−^3 ,
..
.
p(n)(x) =n!an,


also
f(k)(x 0 ) =p(k)(x 0 ) =k!ak

so dass die Koeffizientenak=f

```
(k)(x 0 )
k! des Polynoms eindeutig bestimmt sind.
```
Definition 24.1(Taylorpolynom).Seif:D→Rn-mal differenzierbar, dann heißt

```
Tn(x) =
```
```
∑n
```
```
k=0
```
```
f(k)(x 0 )
k!
(x−x 0 )k
```
dasn-te TaylorpolynomvonfimEntwicklungspunktx 0.

Im Allgemeinen istf 6 = Tnund wir schreibenf(x) = Tn(x) +Rn(x) mit einem
Restglied (oder Fehler)Rn(x). Dabei misstRn(x) =f(x)−Tn(x) also den Abstand
zwischen der Funktionfund dem TaylorpolynomTnim Punktx. Wir gut n ̈ahert das
Taylorpolynom die Funktion an? Anders gefragt: Wie klein ist der FehlerRn? Antwort
gibt der Satz von Taylor.

Satz 24.2(Taylorformel). Seif:I→Rn-mal differenzierbar im IntervallI, und sei
x 0 ∈I. Dann gilt

```
f(x) =Tn(x) +Rn(x) =
```
```
∑n
```
```
k=0
```
```
f(k)(x 0 )
k!
```
```
(x−x 0 )k+Rn(x),
```
mit

```
lim
x→x 0
```
```
Rn(x)
(x−x 0 )n
```
##### = 0. (24.1)

Istf sogar(n+ 1)-mal differenzierbar, so kann man das Restglied auch schreiben als

```
Rn(x) =
f(n+1)(ξ)
(n+ 1)!
```
```
(x−x 0 )n+1 (24.2)
```
mit einemξzwischenxundx 0. Das Restglied in dieser Darstellung nennt man auch
dasLagrange-Restglied. Es sieht so aus wie der n ̈achste Summand im Taylorpolynom,
nur dass das Argument der Ableitung die Zwischenstelleξstattx 0 ist.

Der Satz zeigt, dass der FehlerRn(x) f ̈urx→x 0 sehr schnell gegen 0 geht, d.h. dass
sich das Taylorpolynom wirklich sehr gut an die Funktion anschmiegt.

Bemerkung 24.3. 1) F ̈ur ein Polynomf vom Grad ngiltf(x) =Tn(x), d.h.f
stimmt mit seinem Taylorpolynom ̈uberein. Das kommt daher, dassf(n+1)(x) = 0
f ̈ur allexist und somit das Restglied (24.2) Null ist.

```
2) Istf eine Funktion mitf(n+1)(x) = 0 f ̈ur allex ∈ I, so ist Rn(x) = 0 und
f(x) =Tn(x) ist ein Polynom vom Grad h ̈ochstensn.
```
```
3) Die Stelleξim Lagrange-Restglied liegt zwischenx 0 undx. Dax > x 0 oderx < x 0
sein kann, kann man das in Intervallschreibweise alsξ∈]x,x 0 [∪]x 0 ,x[ schreiben.
```

Beispiel 24.4. 1) Wir berechnen das Taylorpolynomn-ter Ordnung vonexinx 0 =

0. Es istf′(x) = (ex)′=ex, alsof(k)(x) =exund dannf(k)(0) =e^0 = 1 fur alle ̈
k. Damit ist das Taylorolynomn-ter Ordnung

```
Tn(x) =
```
```
∑n
```
```
k=0
```
```
f(k)(0)
k!
```
```
(x−0)k=
```
```
∑n
```
```
k=0
```
##### 1

```
k!
```
```
xk= 1 +x+
```
##### 1

##### 2

```
x^2 +
```
##### 1

##### 6

```
x^3 +...+
```
##### 1

```
n!
```
```
xn.
```
```
Das zugeh ̈orige Restglied ist
```
```
Rn(x) =
eξ
(n+ 1)!
```
```
xn+1,
```
```
wobeiξvonxabh ̈angt und zwischen 0 undxliegt.
```
##### − 2 − 1 0 1 2

##### 0

##### 2

##### 4

##### 6

##### 8

```
exp
```
##### T 0

##### T 1

##### T 2

##### T 3

```
exp
T 0
T 1
T 2
T 3
```
##### − 4 − 2 0 2 4

##### − 2

##### 0

##### 2

```
sin
```
##### T 1

##### T 3

##### T 5

##### T 7

```
sin
T 1
T 3
T 5
T 7
```
```
2) Wir berechnen dasn-te Taylorpolynom von sin(x) inx 0 = 0. Es sind
f′(x) = cos(x),f′′(x) =−sin(x),f′′′(x) =−cos(x),f(4)(x) = sin(x) =f(x),
d.h.
sin(2k)(x) = (−1)ksin(x) und sin(2k+1)(x) = (−1)kcos(x), k= 0, 1 , 2 ,...
Im Entwicklungspunktx 0 = 0 ist dann
sin(2k)(0) = 0 und sin(2k+1)(0) = (−1)k,
also
T 2 n+1(x) =T 2 n+2(x) =
```
```
∑n
```
```
k=0
```
```
(−1)k
(2k+ 1)!
x^2 k+1.
```
```
Speziell sind
T 1 (x) =x,
```
```
T 3 (x) =x−
```
```
x^3
3!
```
##### ,

```
T 5 (x) =x−
```
```
x^3
3!
```
##### +

```
x^5
5!
```
##### ,

```
T 7 (x) =x−
x^3
3!
```
##### +

```
x^5
5!
```
##### −

```
x^7
7!
```
##### .


### 24.2 Extremwerte

Mit dem Satz von Taylor k ̈onnen wir ein hinreichendes Kriterium f ̈ur (lokale) Extrem-
stellen angeben, das auf den h ̈oheren Ableitungen der Funktion beruht.
Wir wissen schon:

- Ist f : [a,b] → Rstetig, so besitztf ein globales Minimum und ein globales
    Maximum (Satz 20.12).
- Istf auch differenzierbar, so sind die einzigen Kandidaten f ̈ur Extremstellen die
    Randpunkteaundbund die Punktexmitf′(x) = 0 (Satz 23.2).
- Wenn die Ableitung ihr Vorzeichen an einem kritischen Punkt wechselt, so liegt
    dort ein Extremum vor (Satz 23.10).

Sie kennen sicher folgendes hinreichende Kriterium f ̈ur ein lokales Extremum mit der
zweiten Ableitung vonf.

Satz 24.5. Seifauf[a,b]differenzierbar undx 0 ∈]a,b[mitf′(x 0 ) = 0. Dann gilt:
1) Wennf′′(x 0 )> 0 , dann hatfinx 0 ein lokales Minimum.
2) Wennf′′(x 0 )< 0 , dann hatfinx 0 ein lokales Maximum.

Ist auchf′′(x 0 ) = 0 (wie beif(x) =x^3 oderf(x) =x^4 inx 0 = 0), so geben die
h ̈oheren Ableitungen Auskunft.

Satz 24.6(Lokale Extremwerte). Seif:I→Reinen-mal differenzierbare Funktion
und seix 0 ein innerer Punkt vonI. Es gelte

```
f′(x 0 ) =f′′(x 0 ) =...=f(n−1)(x 0 ) = 0 und f(n)(x 0 ) 6 = 0.
```
Dann gilt:

```
1) Istnungerade, so hatf inx 0 kein lokales Extremum.
```
```
2) Istngerade, so hatf ein lokales Extremum:
```
- Istf(n)(x 0 )< 0 , so hatfein lokales Maximum.
- Istf(n)(x 0 )> 0 , so hatfein lokales Minimum.

Bemerkung 24.7. 1) F ̈urn= 1 sagt der Satz noch einmal, dass innere Punkte mit
f′(x 0 ) 6 = 0 nicht als Extremstellen in Frage kommen.
2) F ̈urn= 2 ist das Satz 24.5.

Beweis. Zum Beweis nutzen wir die Taylorformel:

```
f(x) =
```
```
∑n
```
```
k=0
```
```
f(k)(x 0 )
k!
(x−x 0 )k+Rn(x)
```
```
=f(x 0 ) +
f(n)(x 0 )
n!
```
```
(x−x 0 )n+Rn(x),
```

da nach Voraussetzungf′(x 0 ) =...=f(n−1)(x 0 ) = 0 sind. Also ist

```
f(x) =f(x 0 ) + (x−x 0 )n
```
##### (

```
f(n)(x 0 )
︸ n︷︷! ︸
6 =0
```
##### +

```
Rn(x)
(x−x 0 )n
︸ ︷︷ ︸
→0 f ̈urx→x 0
```
##### )

und der Term in Klammern hat nah genug beix 0 das selbe Vorzeichen wief

(n)(x 0 )
n!.
Bei ungerademnwechselt (x−x 0 )ndas Vorzeichen inx 0 , so dassf(x) einmal kleiner
und einmal gr ̈oßer alsf(x 0 ) ist, d.h. es liegt kein Extremum inx 0 vor.
Bei gerademnist (x−x 0 )n≥0 fur alle ̈ x. Ist dannf(n)(x 0 )>0, so ist

```
f(x) =f(x 0 ) + (x−x 0 )n
︸ ︷︷ ︸
≥ 0
```
##### (

```
f(n)(x 0 )
n!
```
##### +

```
Rn(x)
(x−x 0 )n
```
##### )

##### ︸ ︷︷ ︸

```
> 0
```
```
≥f(x 0 )
```
f ̈urxnah genug beix 0 , d.h.f hat inx 0 ein Minimum. Ist hingegenf(n)(x 0 )<0, so
findet man genauso, dassfinx 0 ein Maximum hat.

Der Beweis zeigt, wie man die Taylorformel benutzen kann, um Eigenschaften von
(einfachen) Polynomen auf (komplizierte) Funktionen zu ̈ubertragen.

Beispiel 24.8. 1) Bestimme die lokalen und globalen Extrema vonf: [−^32 ,3]→R,
f(x) =x^3 − 3 x.
Suche zun ̈achst nach lokalen Extremwerten im Innern:

```
f′(x) = 3x^2 −3 = 3(x^2 −1) = 3(x−1)(x+ 1)
```
```
hat die Nullstellen +1 und−1. Die zweite Ableitung ist
```
```
f′′(x) = 6x,
```
```
damit ist
```
- f′′(−1) =− 6 <0, d.h.fhat in−1 ein lokales Maximumf(−1) = 2,
- f′′(+1) = 6>0, d.h.fhat in 1 ein lokales Minimum:f(1) =−2.
Untersuchung der Intervallenden: Dafauch noch in−^32 und 3 differenzierbar ist,
k ̈onnen wir hier auch die erste Ableitung betrachten:

```
f′(− 3 /2) = 3
```
##### (

##### 9

##### 4

##### − 1

##### )

##### =

##### 15

##### 4

```
> 0 , f′(3) = 3(9−1) = 24> 0 ,
```
```
undf ist monoton wachsend nahe der Randpunkte. Damit ist hatf in−^32 ein
lokales Minimumf(−^32 ) =^98 und in 3 ein lokales Maximumf(3) = 18.
Daher hatfein globales Maximum inx= 3 und ein globales Minimum inx= 1.
```

```
Skizze der Funktion (y-Achse mit 1/4 skaliert):
```
```
globales Maximum
```
```
globales Minimum
```
```
lokales Maximum
lokales Minimum
```
2) Hatf(x) =x^2 + cos(x) ein lokales Extremum in 0? Wir testen die ersten Ablei-
tungen:

```
f′(x) = 2x−sin(x) f′(0) = 0,
f′′(x) = 2−cos(x) f′′(0) = 1> 0 ,
```
```
also hatfinx= 0 ein lokales Minimum.
```
3) Hatf(x) = sin(x)−xein lokales Extremum in 0? Wir testen die ersten Ableitungen:

```
f′(x) = cos(x)− 1 , f′(0) = 0,
f′′(x) =−sin(x), f′′(0) = 0,
f′′′(x) =−cos(x), f′′′(0) =− 16 = 0,
```
```
also hatfin 0 kein lokales Extremum.
```

## Vorlesung 25

## 25 Anwendungen der Taylor-Approximation

# Taylor-Approximation

### 25.1 N ̈aherungsweise Berechnung von Funktionswerten

Eine wichtige Anwendung der Taylorformel steckt heute in jedem Computer: Die Be-
rechnung von Funktionswerten. Das Taylorpolynom gibt die M ̈oglichkeit, eine mehrfach
differenzierbare Funktion in der Umgebung eines Entwicklungspunktes durch ein Poly-
nom zu approximieren und den Approximationsfehler (Rn=f−Tn) qualitativ mit (24.1)
oder quantitativ mit (24.2) abzusch ̈atzen.

Beispiel 25.1. Berechne n ̈aherungsweise

##### √

4 .4. Dazu betrachten wir die 2-te Taylo-
rapproximation von f(x) =

##### √

xim Entwicklungspunktx 0 = 4 (dort k ̈onnen wir die
Quadratwurzel berechnen). Es ist

```
f(x) =x
```
(^12)
=

##### √

```
x, f(4) = 2,
```
```
f′(x) =
```
##### 1

##### 2

```
x−
```
```
1
```
(^2) =^1
2

##### √

```
x
```
```
, f′(4) =
```
##### 1

##### 4

##### ,

```
f′′(x) =−
```
##### 1

##### 4

```
x−
```
(^32)
=−

##### 1

##### 4

##### √

```
x^3
```
```
, f′′(4) =−
```
##### 1

##### 4 · 23

##### =−

##### 1

##### 32

##### ,

```
f′′′(x) =
```
##### 3

##### 8

```
x−
```
(^52)
=

##### 3

##### 8

##### √

```
x^5
```
##### .

Somit ist das 2-te Taylorpolynom vonfim Entwicklungspunktx 0 = 4:

```
T 2 (x) =f(4) +f′(4)(x−4) +
```
```
f′′(4)
2
(x−4)^2 = 2 +
```
##### 1

##### 4

```
(x−4)−
```
##### 1

##### 64

```
(x−4)^2 ,
```
und damit

```
√
4. 4 ≈T 2 (4.4) = 2 +
```
##### 1

##### 4

##### 0. 4 −

##### 1

##### 64

##### (0.4)^2 = 2. 1 −

##### 1

##### 64

##### ·

##### 42

##### 100

##### = 2. 1 −

##### 1

##### 400

##### = 2. 1 − 0. 0025

##### = 2. 0975.


Wir sch ̈atzen den Fehler ab:

```
R 2 (x) =
```
##### 1

##### 3!

##### 3

##### 8

##### √

```
ξ^5
```
```
(x−4)^3 =
```
```
(x−4)^3
24
```
##### √

```
ξ^5
```
mitξzwischenx 0 = 4 undx. F ̈urx= 4.4 finden wir

##### R 2 (4.4) =

##### 0. 43

##### 24

##### √

```
ξ^5
```
##### ≤

##### 0. 43

##### 24

##### √

##### 45

##### =

##### 0. 13 · 43

##### 29

##### =

##### 0. 001

##### 8

##### = 0. 000125.

Das bedeutet
√
4. 4 ≈ 2. 0975 ± 0. 000125 ,

also

```
2 .097375 = 2. 0975 − 0. 000125 ≤
```
##### √

##### 4. 4 ≤ 2 .0975 + 0.000125 = 2. 097625.

Taschenrechner oder Computer geben 2.0976177 (auf 7 Nachkommastellen gerundet).
Die Approximation mit dem 2-ten Taylorpolynom gibt also schon drei richtige Nach-
kommastellen. Mit gr ̈oßeremnbekommt man noch genauere Ergebnisse.

Beispiel 25.2. Wir wollen nachweisen, dass fur die Eulersche Zahl ̈ e= e^1 < 3 gilt.
Dazu zeigen wire−^1 >^13 mit einer Taylorapproximation der Exponentialfunktion. Nach
Beispiel 24.4 ist

```
ex=T 3 (x) +R 3 (x) = 1 +x+
x^2
2!
```
##### +

```
x^3
3!
```
##### +

```
eξ
4!
x^4
```
mit einemξzwischenx 0 = 0 undx. Fur ̈ x=−1 ist

```
e−^1 =
```
##### 1

##### 2!

##### −

##### 1

##### 3!

##### +

```
eξ
4!
```
##### =

##### 1

##### 3

##### +

```
eξ
4!
```
##### ,

wobeiξzwischen 0 und−1 liegt. Wegeneξ>0 fur jedes ̈ ξ∈R, ist alsoe−^1 >^13 und
somite <3.

Beispiel 25.3.Wir wollen die Funktionswerte der Exponentialfunktion auf [− 1 ,1] be-
rechnen. Dazu approximieren wir die Exponentialfunktion durch ihr Taylorpolynom und
untersuchen, fur welches ̈ nder Fehler|Rn(x)|klein genug ist.
Dasn-te Taylorpolynom mit Entwicklungspunktx 0 = 0 istTn(x) =

```
∑n
k=0
```
```
1
k!x
```
```
kmit
```
zugeh ̈origem Lagrange-RestgliedRn(x) = e
ξ
(n+1)!x

```
n+1, vergleiche Beispiel 24.4. Da wir
```
das Intervall [− 1 ,1] betrachten, ist|x|≤1. Weiter istξzwischen 0 undx, also ebenfalls
in [− 1 ,1]. Daher gilteξ≤e^1 (Monotonie) und wegene≤3 dann|Rn(x)|≤(n+1)!^3.

Fur ̈ n= 8 ist bereits|R 8 (x)|≤ 10 −^5 = 0.00001, d.h. erst in der f ̈unften Nachkommas-
telle haben wir einen Fehler von±1, wenn wirTn(x) anstattexverwenden. Fur gr ̈ ̈oßere
nwird die Genauigkeit noch besser.


### 25.2 Fehlerabsch ̈atzung

Wie wirken sich Messfehler bei Experimenten aus?

Beispiel 25.4. Seif(x) = 3x. Wir messenx mit einem Fehler ∆x und wollen die
Auswirkung auf den beobachteten Wertf quantifizieren. Wir beobachtenf(x+ ∆x),
d.h. wir sind interessiert am Fehler

```
∆f:=f(x+ ∆x)−f(x) = 3(x+ ∆x)− 3 x= 3∆x,
```
d.h. der Fehler verdreifacht sich.

Im Beispiel warflinear und die Quantifizierung des Fehlers einfach. Bei nichtlinea-
ren Funktionenf hilft die Taylorapproximation erster Ordnung vonf inxweiter (mit
Lagrange-Restglied):

```
f(x+ ∆x) =f(x) +f′(x)((x+ ∆x)−x) +
```
##### 1

##### 2

```
f′′(ξ)((x+ ∆x)−x)^2
```
```
=f(x) +f′(x)∆x+
```
##### 1

##### 2

```
f′′(ξ)(∆x)^2.
```
F ̈ur den Fehler infschreiben wir ∆f:=f(x+ ∆x)−f(x) und erhalten

```
∆f=f′(x)∆x+
```
##### 1

##### 2

```
f′′(ξ)(∆x)^2.
```
F ̈ur kleine ∆xist (∆x)^2 viel kleiner als ∆x, also haben wir diequalitative Fehlerappro-
ximation
|∆f|≈|f′(x)||∆x|.

Eine verl ̈assliche Schranke f ̈ur den Fehler liefert

```
|∆f|=|f′(x)∆x+
```
##### 1

##### 2

```
f′′(ξ)(∆x)^2 |≤|f′(x)||∆x|+
```
##### 1

##### 2

```
max
ξ
|f′′(ξ)|(∆x)^2 ,
```
wenn man die zweite Ableitung absch ̈atzen kann. Beim Maximum durchl ̈auftξalle Werte
zwischenxundx+ ∆x.

Beispiel 25.5.Berechnung der Turmh ̈ohehaus der Entfernung`:

```
Position
```
```
Turmspitze
```
```
h
```
##### `

```
α(Erhebungswinkel)
```

Wir nehmen an, dass wir die Distanz`exakt kennen, und wollen die H ̈ohehdes Turms
in Abh ̈angigkeit vom Winkelαbestimmen. Es ist

```
tan(α) =
h
`
, also h=`tan(α).
```
Damit ist

```
h′(α) =`tan′(α) =
```
##### `

```
(cos(α))^2
```
##### .

Ist zum Beispiel`= 70 m undα=π 4 , so isth= 70 tan(π 4 ) = 70 m mit dem Fehler

```
|∆h|≈|h′(α)||∆α|=
```
##### 70

```
cos(π 4 )^2
```
```
|∆α|= 140·|∆α|.
```
Ist dann zum Beispiel der Messfehler des Winkels ∆α≈± 1 ◦= 180 π, so folgt

```
|∆h|≈
```
##### 140

##### 180

```
πm≈ 2 .44 m.
```
Eine pr ̈azise Schranke fur den Fehler erhalten wir mit ̈

```
|∆h|≤|h′(α)||∆α|+
```
##### 1

##### 2

```
max
α−π/ 180 ≤ξ≤α+π/ 180
```
```
|h′′(ξ)|·(∆α)^2
```
Es ist

```
h′′(α) = 2`
tan(α)
(cos(α))^2
```
##### ,

also

```
max
α−π/ 180 ≤ξ≤α+π/ 180
|h′′(ξ)|≤ 2 `
```
```
tan(α+π/180)
(cos(α+π/180))^2
```
##### ,

denn tan ist monoton wachsend und cos^2 ist monoton fallend naheπ 4. Damit ist

```
|∆h|≤
```
##### 140

##### 180

```
π+ 70
```
```
tan(π 4 + 180 π )
(cos(π 4 + 180 π ))^2
```
```
(π
180
```
##### ) 2

##### ≈ 2. 49.

Die qualitative Fehlersch ̈atzung war also schon ganz gut.

Beispiel 25.6(Wheatstone-Br ̈ucke). Bei der Wheatstone-Br ̈ucke zur Vergleichsmessung von Wider-
st ̈anden greift man auf einem Widerstandsdraht der L ̈ange`eine Streckexso ab, dass ein Potentiometer
Nullstellung anzeigt. Dann ist das Verh ̈altnis von unbekanntem WiderstandRzu dem bekannten Ver-
gleichswiderstandR 0
R
R 0 =

x
`−x.
Also ist
R=R(x) =`R−^0 xx.

Ein Messfehler ∆xam Abgleich produziert also einen Fehler

```
∆R≈R′(x)∆x= R^0 `
(`−x)^2
∆x.
```
Ist dann zum Beispiel`= 10 cm und erfolgt der Abgleich gegen den Vergleichswiderstand von 300 Ω bei
x= 2 cm mit einem Messfehler ∆x=± 0 .1 cm, so ist der gesuchte WiderstandR=^3008 ·^2 Ω = 75 Ω mit
einer Genauigkeit ∆R≈^30082 ·^10 · 0 .1 Ω = 4.7 Ω. Eine genaue Fehlerabsch ̈atzung erh ̈alt man mit

```
|∆R|≤|R′(x)|·|∆x|+^1
2
ξ∈[x−max∆x,x+∆x]|R′′(ξ)|(∆x)^2 ≤^4 .8 Ω.
```

### 25.3 Diskretisierung von Ableitungen

Viele physikalische Gesetze sind durch Differentialgleichungen gegeben, also Gleichungen, die eine ge-
suchte Funktion und ihre Ableitungen enthalten, z.B.y′=ay,y′′=−y,.... Viele dieser Differential-
gleichungen lassen sich nicht explizit l ̈osen, so dass man auf numerische L ̈osungsverfahren angewiesen
ist. Ein wichtiges Hilfsmittel dabei ist die Diskretisierung des Problems. Eine M ̈oglichkeit bietet die
Finite-Differenzen-Methode. Die gesuchte Funktiony(x) wird dabei durch eine (diskrete) Zahlenfolgeyk
ersetzt, die die Funktionswerte an den Stellen

```
xk=x 0 +kh, k= 0, 1 , 2 ,...
```
approximieren soll. Dabei isthdie sogenannteDiskretisierungskonstante, und man hofft, f ̈ur sehr kleines
heine gute Approximationyk≈y(xk) zu bekommen.
Dann braucht man auch eine Diskretisierung der Ableitung(en), die man aus der Taylorformel ge-
winnen kann. Aus der Taylorformel folgt, wennh”klein“ ist,

```
f(x+h)≈f(x) +f′(x)h+^12 f′′(x)h^2 ,
```
```
f(x−h)≈f(x)−f′(x)h+^12 f′′(x)h^2 ,
```
Addition der beiden Gleichungen liefert

```
f′′(x)≈f(x+h)−^2 f(x) +f(x−h)
h^2
```
. (25.1)

Die erste Ableitung kann direkt mit der Definition approximiert werden:

```
f′(x)≈f(x+hh)−f(x). (25.2)
```
Beispiel 25.7.Wir betrachten die Differentialgleichung 2. Ordnung

```
y′′+ 6y′+ 9y= 0, y(0) = 0, y′(0) = 0. 5.
```
Wir diskretisieren diese mit den Approximationen (25.1) und (25.2) auf dem Gitterx 0 = 0 undxk=kh
und erhalten die Differenzengleichung
yk+1− 2 yk+yk− 1
h^2 + 6

yk+1−yk
h + 9yk= 0, y^0 = 0, y^1 = 0.^5 h.
Aufl ̈osen nachyk+1liefert die Rekursion

```
yk+1=(2 + 6h−^9 h
```
(^2) )yk−yk− 1
1 + 6h , y^0 = 0, y^1 = 0.^5 h.
In diesem Fall kann man die Differentialgleichung auch exakt l ̈osen und die exakte L ̈osungy(x) =^12 xe−^3 x
mit der nach Diskretisierung berechneten L ̈osung vergleichen. Fur ̈ h= 0.01 undh= 0.001 findet man
zum Beispiel:
(^000). 5 1 1. 5 2
2
4
6
· 10 −^2
h= 0. 01
y(x)
yk
(^000). 5 1 1. 5 2
2
4
6
· 10 −^2
h= 0. 001
y(x)
yk


Mehr zu Differentialgleichungen lernen Sie in den Vorlesungen”Differentialgleichungen f ̈ur Ingenieur-
wissenschaften“ und”Integraltransformationen und partielle Differentialgleichungen“. Mehr zur numeri-
schen L ̈osung von Differentialgleichungen (in einer und mehreren Variablen) lernen Sie in der”Numerik
II f ̈ur Ingenieurwissenschaften“.

### 25.4 Taylorreihen

Wir beginnen mit dem Beispiel der Exponentialfunktion. Dasn-te Taylorpolynom von
exmit Entwicklunspunktx 0 = 0 ist

```
Tn(x) =
```
```
∑n
```
```
k=0
```
##### 1

```
k!
xk
```
mit zugeh ̈origem Restglied

```
Rn(x) =
```
```
eξ
(n+ 1)!
xn+1,
```
wobeiξzwischen 0 undxliegt. Da die Exponentialfunktion monoton wachsend ist, ist
eξ≤e|x|, also

```
|Rn(x)|=
|eξ|
(n+ 1)!
|x|n+1≤e|x|
|x|n+1
(n+ 1)!
→0 f ̈ur n→∞
```
(Beispiel 18.7), d.h. limn→∞Rn(x) = 0. WegenTn(x) =ex−Rn(x) konvergiert die Folge
der Taylorpolynome gegenex:

```
lim
n→∞
```
```
∑n
```
```
k=0
```
```
xk
k!
```
```
=ex.
```
Wir bezeichnen den Grenzwert von

```
∑n
```
```
k=0
```
```
xk
k!
```
```
, n= 0, 1 , 2 , 3 ,...
```
formal mit ∞
∑

```
k=0
```
```
xk
k!
```
und nennen ihn dieTaylorreihevonexim Entwicklunsgpunktx 0 = 0.
Dies motiviert die folgende allgemeine Definition.

Definition 25.8(Taylorreihe). Seif:D→Rbeliebig oft differenzierbar. DieTaylor-
reihevonfim Entwicklungspunktx 0 ∈Dist der Grenzwert der Taylorpolynome:

```
∑∞
```
```
k=0
```
```
f(k)(x 0 )
k!
```
```
(x−x 0 )k:= lim
n→∞
```
```
∑n
```
```
k=0
```
```
f(k)(x 0 )
k!
```
```
(x−x 0 )k.
```

```
Der Grenzwert
```
##### ∑∞

```
k=0
```
```
f(k)(x 0 )
k! (x−x^0 )
```
```
kexistiert und ist gleichf(x) genau dann, wenn
```
das Restglied

```
Rn(x) =f(x)−Tn(x) =f(x)−
```
```
∑n
```
```
k=0
```
```
f(k)(x 0 )
k!
(x−x 0 )k
```
f ̈urn→∞gegen 0 konvergiert.

```
Ein hinreichendes Kriterium f ̈ur die Konvergenz von
```
```
∑n
k=0
```
```
f(k)(x 0 )
k! (x−x^0 )
k,n=
```
0 , 1 , 2 ,..., gegenf(x) im Intervall ]a,b[ ist

```
|f(n)(x)|≤A·Bn f ̈ur allex∈]a,b[
```
mit KonstantenA,Bunabh ̈angig vonn. Dann ist n ̈amlich

```
|Rn(x)|=
|f(n+1)(ξ)|
(n+ 1)!
```
```
|x−x 0 |n+1≤
ABn+1
(n+ 1)!
```
```
|x−x 0 |n+1=A
(B|x−x 0 |)n+1
(n+ 1)!
```
##### → 0

nach Beispiel 18.7.

Bemerkung 25.9.Vorsicht:

```
1) Es kann passieren, dass
```
```
∑n
k=0
```
```
f(k)(x 0 )
k! (x−x^0 )
```
```
k,n= 0, 1 , 2 ,..., f ̈ur keinx 6 =x 0
konvergiert.
2) Es kann passieren, dass
```
```
∑n
k=0
```
f(k)(x 0 )
k! (x−x^0 )
k,n= 0, 1 , 2 ,..., zwar konvergiert,
abernichtgegenf(x).
Ein Beispiel ist die Funktionf(x) =e−^1 /x
2
,x 6 = 0, undf(0) = limx→ 0 f(x) = 0. Man kann zeigen,
dassf(k)(0) = 0 f ̈ur allek∈Nist. Daher ist die Taylorreihe die Nullfunktion, also ungleichf(x)
f ̈urx 6 = 0.
Das sind aber eher die Ausnahmen.

Beispiel 25.10. 1) F ̈urf(x) =existf(n)(x) =ex, alsof(n)(0) =e^0 = 1. Dies ergibt
im Entwicklungspunktx 0 = 0:

```
ex=
```
```
∑n
```
```
k=0
```
##### 1

```
k!
xk+Rn(x)→
```
##### ∑∞

```
k=0
```
```
xk
k!
f ̈urn→∞.
```
```
Die Reihe konvergiert f ̈ur jedesx ∈Rgegenex, wie zu Beginn des Abschnitts
nachgerechnet.
2) F ̈urf(x) = sin(x) ist
sin(2n)(x) = (−1)nsin(x), sin(2n+1)(x) = (−1)ncos(x).
Die Taylorreihe von Sinus im Entwicklungspunktx 0 = 0 ist dann
```
```
sin(x) =
```
##### ∑∞

```
k=0
```
```
(−1)k
(2k+ 1)!
x^2 k+1=x−
```
```
x^3
3!
```
##### +

```
x^5
5!
```
##### −

```
x^7
7!
```
##### ±....

```
Die Reihe konvergiert tats ̈achlich gegen sin(x) f ̈ur jedesx∈R: Es gilt|sin(n)(x)|≤
1 = 1· 1 nf ̈ur jedesx∈R, so dass das obige hinreichende Kriterium (mitA= 1
undB= 1) zeigt, dass die Taylorreihe gegen sin(x) konvergiert.
```

```
3) F ̈urf(x) = cos(x) ist
```
```
cos(2n)(x) = (−1)ncos(x), cos(2n+1)(x) = (−1)n+1sin(x).
```
```
Die Taylorreihe von cos im Entwicklungspunktx 0 = 0 ist
```
```
cos(x) =
```
##### ∑∞

```
k=0
```
```
(−1)k
(2k)!
```
```
x^2 k= 1−
x^2
2!
```
##### +

```
x^4
4!
```
##### −

```
x^6
6!
```
##### ±....

```
Die Taylorreihe von Cosinus konvergiert gegen cos(x) f ̈ur jedesx∈R, denn wie
beim Sinus gilt|cos(n)(x)| ≤1 = 1· 1 nfur alle ̈ x∈R, so dass das hinreichende
Kriterium f ̈ur jedesx∈Rdie Konvergenz der Taylorreihe gegen cos(x) liefert.
```
Die Reihen von Sinus und Cosinus k ̈onnen alternativ als Definition f ̈ur Sinus und
Cosinus genommen werden.
Mit den Reihendarstellungen von exp, cos und sin k ̈onnen wir nun die Euler-Formel
aus Vorlesung 7 nachrechnen: F ̈urx∈Rteilen wir die Reihe f ̈ureixin Summanden mit
geraden (k= 2`) und mit ungeraden (k= 2`+ 1) Summationsindizes:

```
eix=
```
##### ∑∞

```
k=0
```
##### 1

```
k!
(ix)k=
```
##### ∑∞

```
k=0
```
##### 1

```
k!
ikxk=
```
##### ∑∞

```
`=0
```
##### 1

##### (2`)!

```
i^2 `x^2 `+
```
##### ∑∞

```
`=0
```
##### 1

##### (2`+ 1)!

```
i^2 `+1x^2 `+1
```
##### =

##### ∑∞

```
`=0
```
##### 1

##### (2`)!

```
(i^2 )`x^2 `+
```
##### ∑∞

```
`=0
```
##### 1

##### (2`+ 1)!

```
i(i^2 )`x^2 `+1
```
##### =

##### ∑∞

```
`=0
```
##### (−1)`

##### (2`)!

```
x^2 `+i
```
##### ∑∞

```
`=0
```
##### (−1)`

##### (2`+ 1)!

```
x^2 `+1
```
```
= cos(x) +isin(x).
```

Vorlesung 26

## 26 Elementare Funktionen

Bisher haben wir die Exponentialfunktion und ihre Eigenschaften verwendet. Wir wollen
nun nachtr ̈aglich eine solide Definition geben, mit der sich die Eigenschaften der Expo-
nentialfunktion auch nachrechnen lassen. Weiter lernen wir in dieser und in der n ̈achsten
Vorlesung weitere elementare Funktionen kennen, die von der Exponentialfunktion ab-
geleitet sind.

### 26.1 Exponential- und Logarithmusfunktion

Ausgangspunkt: F ̈ur beliebigesx∈Rist die Reihe

```
1 +x+
x^2
2!
```
##### +

```
x^3
3!
```
##### +...=

##### ∑∞

```
k=0
```
```
xk
k!
```
konvergent. Die durch diesen Grenzwert definierte Funktion heißtExponentialfunktion:

```
exp(x):=
```
##### ∑∞

```
k=0
```
```
xk
k!
```
```
:= lim
n→∞
```
```
∑n
```
```
k=0
```
```
xk
k!
```
##### .

Die Zahle:= exp(1) heißt dieEulersche Zahl. Man berechnete≈ 2 .71828182845904.
Wir werden unten mit der allgemeinen Potenz sehen, dass exp(x) =exgilt.
Wir rechnen die Eigenschaften der Exponentialfunktion nach (vergleiche Satz 6.1
und Beispiel 21.7).
1) exp(x) ist differenzierbar mit (exp(x))′= exp(x).
Es gilt
d
dx

```
(n
∑
```
```
k=0
```
```
xk
k!
```
##### )

##### =

```
∑n
```
```
k=0
```
```
d
dx
```
```
xk
k!
```
##### =

```
∑n
```
```
k=1
```
```
kxk−^1
k!
```
##### =

```
∑n
```
```
k=1
```
```
xk−^1
(k−1)!
```
##### =

```
n∑− 1
```
```
k=0
```
```
xk
k!
```
##### .

```
Im vorletzten Schritt haben wirk! =k·(k−1)! verwendet, im letzten Schritt haben
wir eine Indexverschiebung durchgef ̈uhrt. Fur ̈ n→∞konvergiert dies gegen
d
dx
```
```
exp(x) =
d
dx
```
##### (∞

##### ∑

```
k=0
```
```
xk
k!
```
##### )

##### =

##### ∑∞

```
k=0
```
```
xk
k!
```
```
= exp(x).
```

Das wir hier Ableitung und Grenzwert vertauschen k ̈onnen, ben ̈otigt etwas Arbeit,
wir gehen da aber nicht drauf ein.
2) exp(0) = limn→∞

∑n
k=0
0 k
k! = limn→∞1 = 1.
3) Funktionalgleichung: F ̈ur allex 1 ,x 2 ∈Rgilt

```
exp(x 1 +x 2 ) = exp(x 1 ) exp(x 2 ).
Das rechnet man nach durch Ausmultiplizieren der beiden Reihen von exp(x 1 ) und
exp(x 2 ) und neu sortieren der Terme. Wir verzichten hier darauf.
Insbesondere ist
exp(x) exp(−x) = exp(x−x) = exp(0) = 1.
Daraus sehen wir: exp(x) 6 = 0 f ̈ur allex∈R, und
1
exp(x)
= exp(−x).
```
f ̈ur allex∈R.
4) exp(x)>0 f ̈ur allex∈R.
Wir wissen schon, dass exp(x) 6 = 0 f ̈ur jedesx∈R. Angenommen, es gibt ein
x 0 mit exp(x 0 )<0. Da exp(0) = 1>0 und exp stetig ist (da differenzierbar),
h ̈atte exp dann nach dem Zwischenwertsatz eine Nullstelle zwischenx 0 und 0, im
Widerspruch zu exp(x) 6 = 0 f ̈ur allex.
5) exp ist streng monoton wachsend, denn exp′(x) = exp(x)>0 f ̈ur allex∈R.
6) F ̈urx >0 ist

```
exp(x) =
```
##### ∑∞

```
k=0
```
```
xk
︸︷︷︸k!
≥ 0
```
##### ≥

```
xn+1
(n+ 1)!
```
##### ,

```
also
exp(x)
xn
```
##### ≥

```
x
(n+ 1)!
```
##### .

```
Da limx→+∞(n+1)!x = +∞, folgt
```
```
lim
x→+∞
```
```
exp(x)
xn
```
##### = +∞

```
f ̈ur allen. Daher gilt:
F ̈urx→+∞w ̈achstexp(x)schneller als jede Potenzxn.
F ̈urx→−∞ist hingegen
```
```
x→−∞lim exp(x) = limx→−∞
```
##### 1

```
exp(−x)
= limx→+∞
```
##### 1

```
exp(x)
```
##### = 0.

```
Insbesondere ist
exp :R→]0,∞[
injektiv (da streng monoton wachsend) und surjektiv (wegen limx→−∞exp(x) = 0
und limx→+∞exp(x) = +∞und dem Zwischenwertsatz), also bijektiv und damit
umkehrbar.
```

```
Die Umkehrfunktion heißt(nat ̈urliche) Logarithmusfunktion, ln : ]0,∞[→R.
```
```
x
```
```
y
```
##### 0

```
y=x
```
```
Graph von ln
```
```
Graph von exp
```
##### − 4 − 3 − 2 − 1 1 2 3 4

##### − 4

##### − 3

##### − 2

##### − 1

##### 1

##### 2

##### 3

##### 4

F ̈ur den Logarithmus gilt als Umkehrfunktion der Exponentialfunktion

```
exp(ln(x)) =x f ̈urx > 0 ,
ln(exp(x)) =x f ̈urx∈R.
```
Weiter gilt fur den Logarithmus (vergleiche Satz 6.2 und Beispiel 22.2): ̈
1) Funktionalgleichung: ln(xy) = ln(x) + ln(y) f ̈ur allex,y >0.
2) ln(1) = 0.
3) ln(x^1 ) =−ln(x) f ̈ur allex >0.
4) ln(x/y) = ln(x)−ln(y) f ̈ur allex,y >0.
5) ln(xn) =nln(x) f ̈ur allex >0 undn∈Z.
6) ln : ]0,∞[→Rist bijektiv.
7) ln′(x) =^1 x.
Zum Abschluss leiten wir eine Taylorentwicklung f ̈ur den Logarithmus her. Da ln(x)
inx= 0 nicht definiert ist, betrachten wir stattdessenf(x) = ln(1 +x), mitf(0) =
ln(1) = 0. Dann sind

```
f′(x) =
```
##### 1

```
1 +x
= (1 +x)−^1 , f′(0) = 1,
```
```
f′′(x) =−(1 +x)−^2 , f′′(0) =− 1 ,
f′′′(x) = (−1)^2 2(1 +x)−^3 , f′′′(0) = 2,
```
und allgemein (nachrechnen mit Induktion!):

```
f(k)(x) = (−1)k−^1 (k−1)!(1 +x)−k, f(k)(0) = (−1)k−^1 (k−1)!.
```
Im Entwicklungspunktx 0 = 0 ergibt dies die Taylorreihe

```
ln(1 +x) =
```
##### ∑∞

```
k=1
```
```
(−1)k−^1
k
xk=x−
```
```
x^2
2
```
##### +

```
x^3
3
```
##### −

```
x^4
4
```
##### ±...

Die Reihe konvergiert f ̈ur|x|<1 gegen ln(1 +x).


### 26.2 Allgemeine Potenzfunktion

Fur ̈ n∈Nhaben wirandefiniert alsa·a·...·a(nmal). Aber was ist zum Beispielaπ?
Dies kann man mit der allgemeinen Potenz erkl ̈aren.

Definition 26.1(allgemeine Potenz).Fur ̈ a >0 undb∈Rdefinieren wir

```
ab:= exp(bln(a)).
```
Dabei heißtaBasisundbExponent. Das ergibt insbesondere zwei Funktionen:

1) dieallgemeine Potenzax,
2) diePotenzfunktionxb.
Fur die allgemeine Potenz gelten ̈ ̈ahnliche Rechenregeln wie fur die Exponentialfunk- ̈
tion.

Satz 26.2(Rechenregeln f ̈ur die allgemeine Potenz).F ̈ur reellesa > 0 und allex,y∈R
gilt
1)a^0 = 1,
2)ax+y=axay,
3)a−x=a^1 x,
4)ax> 0 ,
5)ln(ax) =xln(a),
6)(ax)y=axy.

Beweis. Die Rechenregeln folgen aus den Eigenschaften der Exponentialfunktion. Es ist
a^0 = exp(0 ln(a)) = exp(0) = 1. Weiter ist

```
ax+y= exp((x+y) ln(a)) = exp(xln(a) +yln(a)) = exp(xln(a)) exp(yln(a)) =axay.
```
Insbesondere ist also axa−x = ax−x = a^0 = 1, alsoa−x = a^1 x. Weiter ist ax =
exp(xln(a))>0. Weiter ist ln(ax) = ln(exp(xln(a))) =xln(a), und damit

```
(ax)y= exp(yln(ax)) = exp(yxln(a)) = exp((xy) ln(a)) =axy.
```
Bemerkung 26.3.Ausax+y=axayfolgt durch vollst ̈andige Induktion f ̈urn∈N:

```
an=a︸·...︷︷·a︸
nmal
```
##### .

Das heißt, die Definition vonaxstimmt f ̈ur nat ̈urlichesnmit der alten Definitionuberein. ̈
Es gilta

(^1) n
= n

##### √

```
a, denna
```
(^1) n
ist eine positive L ̈osung vonxn=a, da (a
n^1
)n=a
(^1) nn
=a^1 =a.
Die Basis
a=e= exp(1) = 1 + 1 +

##### 1

##### 2!

##### +

##### 1

##### 3!

##### +...

(Eulersche Zahl,e≈ 2. 7183 ...) heißt dienat ̈urliche Basis. F ̈ur diese ist

```
ex= exp(xln(e)) = exp(xln(exp(1))) = exp(x),
```
also genau die Exponentialfunktion.


Satz 26.4. F ̈ura > 0 ist die Funktionaxauf ganzRdifferenzierbar mit

```
d
dx
```
```
ax= ln(a)ax.
```
Insbesondere gilt:

- F ̈ura > 1 istln(a)> 0 , alsoaxstreng monoton wachsend.
- F ̈ura= 1istln(a) = 0, alsoax= 1konstant.
- F ̈ura < 1 istln(a)< 0 , alsoaxstreng monoton fallend.
F ̈urb∈Rundx > 0 ist
d
dx
xb=bxb−^1.

Beweis. Mit der Kettenregel finden wir

```
d
dx
```
```
(ax) =
d
dx
```
```
exp(xln(a)) = exp(xln(a)) ln(a) = ln(a)ax
```
und
d
dx

```
(xb) =
d
dx
```
```
exp(bln(x)) = exp(bln(x))b
```
##### 1

```
x
```
```
=xbbx−^1 =bxb−^1.
```
Die Monotonieaussagen folgen aus dem Monotoniekriterium 23.6.

```
x
```
```
2 x
```
```
(^12 )x
```
```
1 x
```
##### − 2 − 1 0 1 2

##### 1

##### 2

##### 3

##### 4

F ̈ura >0,a 6 = 1, istaxsomit injektiv und damit umkehrbar. Die Umkehrfunktion ist
derLogarithmus zur Basisa:

```
loga: ]0,∞[→R, x7→loga(x).
```
Es ist per Definition
ay=x⇔loga(x) =y.

F ̈ura=eist loge(x) = ln(x) der nat ̈urliche Logarithmus. Der Logarithmus zur Basisa
l ̈asst sich durch den nat ̈urlichen Logarithmus darstellen: Ausaloga(x)=xfolgt

```
ln(x) = ln(aloga(x)) = loga(x) ln(a),
```

also

```
loga(x) =
```
```
ln(x)
ln(a)
```
##### .

Daher gelten f ̈ur logadie gleichen Rechenregeln wie f ̈ur ln. Diese lassen sich auch direkt
aus den Rechenregeln f ̈ur die allgemeine Potenz herleiten.

Satz 26.5(Rechenregeln fur den Logarithmus zur Basis ̈ a).Fur ̈ a,x,y > 0 gilt
1)loga(xy) = loga(x) + loga(y),
2)loga(xb) =bloga(x),
3)loga(^1 x) =−loga(x),
4)loga(xy) = loga(x)−loga(y).

```
x
```
```
y= ln(x)
```
```
y= log 10 (x)
```
##### 1 2 3 4 5

##### − 2

##### − 1

##### 0

##### 1

##### 2

### 26.3 Komplexe Exponentialfunktion

In Vorlesung 7 haben wir dieEuler-Formel

```
eiφ= cos(φ) +isin(φ), φ∈R,
```
f ̈ur die Eulerdarstellung von komplexen Zahlen verwendet. Mit Hilfe der Reihendarstellungen von exp, cos
und sin haben wir die Euler-Formel nachgerechnet (Abschnitt 25.4). Damit ist die Exponentialfunktion
auch f ̈ur komplexe Zahlen der Formiφdefiniert.
Allgemeiner kann man zeigen: Die Exponentialreihe konvergiert f ̈ur alle komplexen Zahlenz,

```
exp(z) =
```
```
∑∞
k=0
```
```
zk
k!= 1 +z+
```
```
z^2
2 +
```
```
z^3
3!+...,
```
wodurch diekomplexe Exponentialfunktionexp :C→Cdefiniert wird. F ̈urz=x+iymit reellenx,y
gilt allgemein
exp(z) = exp(x+iy) = exp(x) exp(iy) = exp(x)(cos(y) +isin(y)).

Die komplexe Exponentialfunktion erf ̈ullt die Gleichungez+2πi=eze^2 πi=ezf ̈ur allez∈C. Insbesondere
ist exp :C→Cnicht injektiv und damit nicht eindeutig umkehrbar. Das bereitet Probleme bei der
Definition eines komplexen Logarithmus, auf die wir hier nicht weiter eingehen. Mehr zur komplexen
Exponential- und Logarithmusfunktion in”Analysis III f ̈ur Ingenieurwissenschaften“.


Vorlesung 27

## 27 Elementare Funktionen

Wir behandeln die trigonometrischen und hyperbolischen Funktionen.

### 27.1 Trigonometrische Funktionen

Sinus und Cosinus hatten wir bereits in Vorlesung 6 als Funktionen am Einheitskreis
betrachtet und daraus ihren Graphen und elementare Eigenschaften hergeleitet.
Mit Hilfe der Exponentialfunktion sind die Additionstheoreme f ̈ur Sinus und Cosinus
leicht nachzurechnen.

Satz 27.1(Additionstheoreme von Sinus und Cosinus).Fur alle ̈ x,y∈Rgilt

```
cos(x+y) = cos(x) cos(y)−sin(x) sin(y),
sin(x+y) = sin(x) cos(y) + cos(x) sin(y).
```
Beweis. Man kann beide Additionstheoreme mithilfe der Reihendarstellung von sin und
cos beweisen (rechte Seite ausmultiplizieren und vereinfachen). Leichter geht es mit der
komplexen Exponentialfunktion (Abschnitt 26.3). Es sind

```
eix= cos(x) +isin(x), eiy= cos(y) +isin(y),
```
also

```
cos(x+y) +isin(x+y) =ei(x+y)=eixeiy= (cos(x) +isin(x))(cos(y) +isin(y))
= (cos(x) cos(y)−sin(x) sin(y)) +i(sin(x) cos(y) + cos(x) sin(y)).
```
Ein Vergleich von Real- und Imagin ̈arteil liefert beide Additionstheoreme.

```
Insbesondere ergeben sich
cos(x+ 2π) = cos(x) cos(2π)−sin(x) sin(2π) = cos(x),
sin(x+ 2π) = sin(x) cos(2π) + cos(x) sin(2π) = sin(x),
cos(x+π) = cos(x) cos(π)−sin(x) sin(π) =−cos(x),
sin(x+π) = sin(x) cos(π) + cos(x) sin(π) =−sin(x).
```
Dies zeigt noch einmal, dass Sinus und Cosinus 2π-periodisch sind.


Tangens und Cotangens. DerTangensist

```
tan(x) =
```
```
sin(x)
cos(x)
, x 6 =
```
```
π
2
+kπ,k∈Z,
```
und ist inRaußer in den Nullstellen des Cosinus definiert. DerCotangesist

```
cot(x) =
cos(x)
sin(x)
```
```
, x 6 =kπ,k∈Z,
```
und ist inRaußer in den Nullstellen des Sinus definiert. Wegen sin(x+π) =−sin(x) und
cos(x+π) =−cos(x) (siehe oben), sind dann tan(x+π) = tan(x) und cot(x+π) = cot(x)
fur alle ̈ xim Definitionsbereich, d.h.Tangens und Cotangens sindπ-periodisch. Das sieht
man auch an ihren Funktionsgraphen. Der Funktionsgraph des Tangens ist

```
x
```
```
y
```
```
tan
```
```
−^32 π −π 2 π 2 32 π
```
und der Funktionsgraph des Cotangens ist

```
x
```
```
y
```
```
cot
```
```
−π 0 π
```

Tangens und Cotangens haben folgende geometrische Interpretation am Einheitskreis
(Strahlensatz):

##### 0 1

##### 1

```
cos(x)
```
```
sin(x)
```
```
cot(x)
```
```
tan(x)
x
```
Die Ableitung des Tangens berechnet man mit der Quotientenregel (wie in Beispiel 23.9):

```
tan′(x) =
```
##### (

```
sin(x)
cos(x)
```
##### )′

##### =

```
sin′(x) cos(x)−sin(x) cos′(x)
cos(x)^2
```
```
=
cos(x) cos(x) + sin(x) sin(x)
cos(x)^2
```
##### =

```
cos(x)^2 + sin(x)^2
cos(x)^2
```
```
=
```
##### {

```
1
cos(x)^2
1 + tan(x)^2
```
##### .

Die erste Darstellung folgt mit dem trigonometrischen Pythagoras, die zweite durch
aufteilen des Bruchs. Je nach Situation ist mal die eine, mal die andere Darstellung
hilfreich.

### 27.2 Arcus-Funktionen – Umkehrfunktionen der Winkelfunktionen

### funktionen

Keine der Funktionen cos(x), sin(x), tan(x), cot(x) ist injektiv, sie sind vielmehr alle
periodisch. Aber wir k ̈onnen sie auf Teilintervalle einschr ̈anken, wo sie injektiv sind, und
dann gibt es zu den so eingeschr ̈ankten Funktionen eine Umkehrfunktion. Diese Funk-
tionen nennt manArcus-Funktionen(=Bogenfunktionen), weil sie zu einem gegebenen
Wert (z.B.y= cos(x)) die zugeh ̈orige Bogenl ̈angexliefern.
Wir beginnen mit dem Sinus.

```
x
```
```
y
```
```
−^32 π −π 2 π 2 32 π^52 π
− 1
```
##### 1

##### 0


Spiegelt man den Funktionsgraph an der Winkelhalbierendeny=x(um die Umkehr-
funktion zu finden, falls sie denn existiert), erh ̈alt man

```
y
```
```
x
```
und das ist keine Funktion, da jedemx-Wert mehrerey-Werte zugeordnet werden. Um
dennoch eine Umkehrfunktion zu erhalten, verkleinert man den Definitionsbereich so,
dass der gespiegelte Graph wieder eine Funktion definiert.

Der Sinus ist auf [−π 2 ,π 2 ] streng monoton wachsend, also injektiv und damit umkehr-
bar. Die Umkehrfunktion

```
arcsin : [− 1 ,1]→
```
##### [

##### −

```
π
2
```
##### ,

```
π
2
```
##### ]

heißtArcussinus. Den Graphen erhalten wir, indem wir den Graphen des Sinus an der
Winkelhalbierenden spiegeln:

```
x
```
```
y
```
```
sin
```
```
arcsin
```
##### − 1 1

```
−π 2
```
```
π
2
```
Fur ̈ x∈[− 1 ,1] gibt es unendlich viele Winkelymit sin(y) =x. Der Winkely∈[−π 2 ,π 2 ]
heißtHauptwert des Arcussinus. Schr ̈ankt man den Sinus auf ein anderes Intervall, wo
man ihn umkehren kann, ein, so erh ̈alt man eine andere Umkehrfunktion.
Analog verf ̈ahrt man f ̈ur den Cosinus, der streng monoton fallend auf [0,π] ist, also
umkehrbar. Die Umkehrfunktion

```
arccos : [− 1 ,1]→[0,π],
```
heißtArcuscosinusund hat den folgenden Graphen:


```
x
```
```
y
```
```
cos
```
```
arccos
```
```
− 1 1 π
− 1
```
##### 1

```
π
2
```
```
π
```
Der Tangens ist auf ]−π 2 ,π 2 [ streng monoton wachsend, denn tan′(x) = 1 + tan(x)^2 ≥
1, und damit umkehrbar. Die Umkehrfunktion

```
arctan :R→
```
##### ]

##### −

```
π
2
```
##### ,

```
π
2
```
##### [

heißtArcustangens.

```
x
```
```
y
```
```
arctan
```
##### − 5 − 4 − 3 − 2 − 1 0 1 2 3 4 5

```
−π 2
```
```
π
2
```
Den Arcustangens haben wir bereits in Vorlesung 7 zur Bestimmung des Arguments
einer komplexen Zahl eingefuhrt. Da der Arcustangens nur Werte zwischen ̈ −π 2 und π 2
annimmt, erhalten wir nur die Argumente von Zahlen in der rechten Halbebene (Re(z)>
0), f ̈ur komplexe Zahlen in der linken Halbebene oder auf der imagin ̈aren Achse mussten
wir die Formel anpassen. (Dabei wird der Tangens auf einem anderen Intervall als ]−π 2 ,π 2 [
umgekehrt.)

Ableitungen der Arcus-Funktionen. Die Ableitungen der Arcus-Funktionen be-
rechnen sich mit der Formel f ̈ur die Ableitung der Umkehrfunktion (Satz 22.1).

Satz 27.2(Ableitungen der Arcus-Funktionen). Im Inneren ihres jeweiligen Definiti-


onsbereichs sind

```
arcsin′(x) =
```
##### 1

##### √

```
1 −x^2
```
```
, − 1 < x < 1 ,
```
```
arccos′(x) =−
```
##### 1

##### √

```
1 −x^2
```
```
, − 1 < x < 1 ,
```
```
arctan′(x) =
```
##### 1

```
1 +x^2
, x∈R.
```
Beweis. F ̈ur arcsin rechnen wir: F ̈ur|x|<1 ist

```
arcsin′(x) =
```
##### 1

```
sin′(arcsin(x)
```
##### =

##### 1

```
cos(arcsin(x))
```
##### =

##### 1

##### √

```
1 −(sin(arcsin(x)))^2
```
##### =

##### 1

##### √

```
1 −x^2
```
Dabei haben wir verwendet, dass cos(y) = +

##### √

1 −sin(y)^2 f ̈ury ∈ [−π 2 ,π 2 ] gilt und
arcsin(x)∈[−π 2 ,π 2 ]. Analog ergibt sich arccos′. Fur den Arcustangens ist es einfacher: ̈

```
arctan′(x) =
```
##### 1

```
tan′(arctan(x))
```
##### =

##### 1

```
1 + (tan(arctan(x))^2
```
##### =

##### 1

```
1 +x^2
```
### 27.3 Hyperbolische Funktionen

Mit Hilfe der Exponentialfunktion erkl ̈aren wir die zwei folgenden, f ̈ur die Ingenieurma-
thematik wichtigen, Funktionen. Diese treten zum Beispiel in der Vorlesung
”
Mechanik
II“ auf.

Definition 27.3(Hyperbelfunktionen). Die Hyperbelfunktionen sind derCosinus hy-
perbolicuscosh :R→R,

```
cosh(x) =
ex+e−x
2
```
##### ,

und derSinus hyperbolicussinh :R→R,

```
sinh(x) =
```
```
ex−e−x
2
```
##### .

Der Graph des Cosinus Hyperbolicus hat die Form einer durchh ̈angenden Kette (Ket-
tenlinie):


```
x
```
```
y
```
```
ex
2
```
```
e−x
2
```
```
cosh
```
```
sinh
```
Die hyperbolischen Funktionen haben vieleAhnlichkeiten zu Sinus und Cosinus. Wir ̈
geben nur einige Eigenschaften an. Weitere, wie zum Beispiel Additionstheoreme, finden
Sie in Formelsammlungen.

Satz 27.4. F ̈ur die hyperbolischen Funktionen gilt:
1) cosh(x)^2 −sinh(x)^2 = 1f ̈ur allex∈R,
2) cosh′(x) = sinh(x),sinh′(x) = cosh(x),
3) coshist eine gerade Funktion:cosh(−x) = cosh(x)f ̈ur allex∈R,
4) sinhist eine ungerade Funktion:sinh(−x) =−sinh(x)f ̈ur allex∈R,
5) Es istsinh :R→Rbijektiv. Die Umkehrfunktion heißtArea Sinus hyperbolicus
und erf ̈ullt
arsinh(x) = ln(x+

##### √

```
x^2 + 1) f ̈ur allex∈R.
6) Es istcosh : [0,∞[ →[1,∞[bijektiv. Die Umkehrfunktion heißt Area Cosinus
hyperbolicusund erf ̈ullt
```
```
arcosh(x) = ln(x+
```
##### √

```
x^2 −1) f ̈ur allex≥ 1.
```
Beweis. Wir rechnen1)mit der Definition nach: Es ist

```
cosh(x)^2 −sinh(x)^2 =
```
##### (

```
ex+e−x
2
```
##### ) 2

##### −

##### (

```
ex−e−x
2
```
##### ) 2

##### =

```
e^2 x+ 2 +e−^2 x
4
```
##### −

```
e^2 x−2 +e−^2 x
4
= 1.
```
Ebenso rechnet man2)–4)nach.
Es ist sinh′(x) = cosh(x) = e
x+e−x
2 ≥^1 > 0 fur alle ̈ x∈R, also ist sinh streng
monoton wachsend und damit injektiv. Weiter sind limx→±∞sinh(x) =±∞, also ist


sinh surjektiv. Die Form der Umkehrfunktion findet man durch aufl ̈osen vony= sinh(x)
nachx. Alternativ kann man nachrechnen, dass arsinh und die angegebene Funktion die
gleiche Ableitung besitzen (dann sind sie gleich bis auf eine Konstante) und in 0 den
gleichen Wert besitzen.Ahnlich f ̈ ̈ur cosh.

Aus der Taylorreihe der Exponentialfunktion bekommen wir die Taylorreihen der
hyperbolischen Funktionen:

```
cosh(x) =
```
##### ∑∞

```
k=0
```
```
x^2 k
(2k)!
```
```
, sinh(x) =
```
##### ∑∞

```
k=0
```
```
x^2 k+1
(2k+ 1)!
```
##### .

Weiter kann man denTangens hyperbolicusundCotanges hyperbolicusdefinieren:

```
tanh(x) =
sinh(x)
cosh(x)
```
##### =

```
ex−e−x
ex+e−x
```
```
, coth(x) =
cosh(x)
sinh(x)
```
##### .

Der Tangens hyperbolicus ist eine bijektive Funktion vonRnach ]− 1 ,1[.

```
x
```
```
y
tanh
```
##### − 5 − 4 − 3 − 2 − 1 0 1 2 3 4 5

##### − 1

##### 1


```
Die Ableitungen der elementaren Funktionen
```
```
f(x) f′(x) Bemerkungen
```
```
xn nxn−^1 n= 1, 2 , 3 ,...
```
```
exp(x) exp(x)
ax axln(a) a >0,x∈R
ln(|x|)
```
##### 1

```
x
```
```
x 6 = 0
```
```
loga(|x|)
```
##### 1

```
xln(a)
```
```
a,x >0, loga(x) =ln(ln(xa))
```
```
xa axa−^1 x >0,a∈R
```
```
sin(x) cos(x)
cos(x) −sin(x)
tan(x) 1 + tan(x)^2 =
```
##### 1

```
cos(x)^2
```
```
x 6 =π 2 +kπ,k∈Z
```
```
cot(x) − 1 −cot(x)^2 =−
```
##### 1

```
sin(x)^2
x 6 =kπ,k∈Z
```
```
arcsin(x)
```
##### 1

##### √

```
1 −x^2
```
```
|x|< 1
```
```
arccos(x) −
```
##### 1

##### √

```
1 −x^2
```
```
|x|< 1
```
arctan(x)

##### 1

```
1 +x^2
arccot(x) −
```
##### 1

```
1 +x^2
cosh(x) sinh(x)
sinh(x) cosh(x)
tanh(x) 1 −tanh(x)^2 =
```
##### 1

```
cosh(x)^2
coth(x) 1 −coth(x)^2 =−
```
##### 1

```
sinh(x)^2
```
```
arsinh(x)
```
##### 1

##### √

```
1 +x^2
```
arcosh(x)

##### 1

##### √

```
x^2 − 1
```
```
x∈]1,∞[
```
artanh(x)

##### 1

```
1 −x^2
```
```
|x|< 1
```
```
Tabelle 27.1: Ableitungen der elementaren Funktionen.
```


Vorlesung 28

## 28 Das Integral

Wir lernen das Integral kennen.

### 28.1 Integraldefinition und Fl ̈achenberechnung

Wie groß ist der Fl ̈acheninhalt zwischen dem Graphen einer Funktionf: [a,b]→Rund
derx-Achse?

```
x
```
```
y
```
```
Fl ̈acheF=?
```
Dazu nehmen wir zun ̈achst an, dassfstetig und positiv ist (alsof(x)≥0 fur ̈ x∈[a,b]).
Die Idee ist, die Fl ̈ache durch Rechtecke zu approximieren. Dazu unterteilen wir [a,b]
innTeilintervalle [x 0 ,x 1 ], [x 1 ,x 2 ],..., [xn− 1 ,xn], wobei

```
a=x 0 < x 1 < x 2 < ... < xn=b.
```
Der Einfachheit halber w ̈ahlen wir alle Intervalle gleich lang, also mit der L ̈ange ∆x=
b−a
n , so dass
xj=a+j∆x=a+j
b−a
n

```
, j= 0, 1 , 2 ,...,n.
```
Die Punktex 0 ,x 1 ,...,xnsind dann ̈aquidistant (d.h., sie haben gleichen Abstand von-
einander).

a=x 0 x 1 x 2 x 3 xn− (^1) xn=b
∆x ∆x ∆x ∆x


Approximieref auf [xj− 1 ,xj] durch die Konstantef(xj− 1 ). Dann heißt

```
Fn=
```
```
∑n
```
```
j=1
```
```
f(xj− 1 )(xj−xj− 1 ) =
```
```
∑n
```
```
j=1
```
```
f(xj− 1 )∆x=
```
```
∑n
```
```
j=1
```
```
f(xj− 1 )
```
```
b−a
n
```
eineRiemannsche Summeder Funktionf: [a,b]→R. Diese Summe ist die Fl ̈ache aller
Rechtecke, mit der wir die Fl ̈ache unter dem Funktionsgraphen vonfapproximieren.

```
x
```
```
y
```
```
x
```
```
y
```
Erwartung: je gr ̈oßer nist, desto besser approximiert Fn den exakten WertF des
Fl ̈acheninhalts.

Satz 28.1. Istf: [a,b]→Rstetig oder monoton, so existiert der Grenzwert

```
nlim→∞Fn=:
```
```
∫b
```
```
a
```
```
f(x)dx.
```
Wir nennen ihn dasbestimmte Integralvonf ̈uber[a,b].

Bemerkung 28.2. 1) Bezeichnungen: In

```
∫b
af(x)dxheißt
```
- f derIntegrand
- xdieIntegrationsvariable
- adieuntere Integrationsgrenze
- bdieobere Integrationsgrenze.
2) Die Integrationsvariable kann beliebig benannt werden:
∫b

```
a
```
```
f(x)dx=
```
```
∫b
```
```
a
```
```
f(t)dt=
```
```
∫b
```
```
a
```
```
f(s)ds=
```
```
∫b
```
```
a
```
```
f(u)du.
```
```
3) Das bestimmte Integral
```
```
∫b
af(x)dxist eine reelle Zahl.
4) In der Schreibweise
∫b
```
```
a
```
```
f(x)dx= lim
n→∞
```
```
∑n
```
```
j=1
```
```
f(xj− 1 )·∆x
```
```
erinnert
```
##### ∫

```
(”S“ wie Summe) an
```
##### ∑

```
unddxan ∆x.
Vgl. auch die Schreibweisedfdx(x)= lim∆x→ 0 ∆∆f(xx)= lim∆x→ 0 f(x+∆∆xx)−f(x).
```

Beispiel 28.3.Istf: [a,b]→Rkonstant, alsof(x) =cf ̈ur allex∈[a,b], so ist

```
∫b
```
```
a
```
```
f(x)dx=c(b−a).
```
F ̈urc >0 ist das genau der Fl ̈acheninhalt zwischen dem Graphen vonfund derx-Achse.
F ̈urc <0 ist das Integral das negative des Fl ̈acheninhalts.

```
x
```
```
y
```
```
a b
```
```
c
```
Anwendungen:

```
1)Fl ̈achenbilanz und Fl ̈achenberechnung:Fallsf : [a,b]→Rstetig ist aber
nichtf(x) ≥0 f ̈ur allex∈[a,b] gilt, so ergibt das Integral dieFl ̈achenbilanz,
d.h. die Fl ̈ache ̈uber derx-Achse z ̈ahlt positiv, die Fl ̈ache unter derx-Achse z ̈ahlt
negativ. Um die Gesamtfl ̈ache zwischen dem Graph vonfund derx-Achse zu be-
stimmen, zerlege [a,b] in Teilintervalle, auf denenfdas Vorzeichen nicht wechselt.
Zum Beispiel:
```
##### F 1

##### F 2

x 0 =a x (^1) b=x 2
Die Fl ̈achenbilanz ist ∫
b
a
f(x)dx=F 1 −F 2 ,
und die Fl ̈ache zwischen dem Graph vonfund derx-Achse istF=F 1 +F 2 , wobei

##### F 1 =

```
∫x 1
```
```
x 0
```
```
f(x)dx und F 2 =
```
##### ∣∣

##### ∣∣

```
∫x 2
```
```
x 1
```
```
f(x)dx
```
##### ∣∣

##### ∣∣.

```
2)Streckenberechnung:Zum Beispiel ein Massenpunkt im freien Fall.
Wir der Massenpunkt zur Zeitt= 0 losgelassen, so ist die Geschwindigkeit pro-
portional zur Fallzeit:v(t) =gtmitg= 9.81 m/s^2.
Wie groß ist die zuruckgelegte Fallstrecke zum Zeitpunkt ̈ T >0? F ̈ur konstantes
vw ̈are diesvT.
```

```
Dav nicht konstant ist, zerlegen wie [0,T] in kleine Teilintervalle [tj− 1 ,tj] und
approximierenv(t) auf [tj− 1 ,tj] durchv(tj− 1 ):
```
```
0 =t 0 < t 1 < ... < tn=T, ∆t=
```
##### T− 0

```
n
```
##### =

##### T

```
n
```
```
, tj=j∆t=j
```
##### T

```
n
```
##### .

```
Daher ist die zur ̈uckgelegte StreckeSungef ̈ahr
```
##### S≈

```
∑n
```
```
j=1
```
```
v(tj− 1 )∆t,
```
```
ist also durch eine Riemannsche Summe gegeben. Im Grenzwert ergibt sich die
exakte Strecke als
```
```
S= limn→∞
```
```
∑n
```
```
j=1
```
```
v(tj− 1 )∆t=
```
##### ∫T

```
0
```
```
v(t)dt.
```
```
Wir berechnen nun das Integral:
∫T
```
```
0
```
```
v(t)dt= limn→∞
```
```
∑n
```
```
j=1
```
```
v(tj− 1 )∆t= limn→∞
```
```
∑n
```
```
j=1
```
```
g(j−1)
```
##### T

```
n
```
##### ·

##### T

```
n
= limn→∞
```
```
gT^2
n^2
```
```
∑n
```
```
j=1
```
```
(j−1)
```
```
= lim
n→∞
```
```
gT^2
n^2
```
```
n∑− 1
```
```
k=0
```
```
k= lim
n→∞
```
```
gT^2
n^2
```
```
(n−1)n
2
```
```
= lim
n→∞
```
```
gT^2
2
```
##### (

##### 1 −

##### 1

```
n
```
##### )

##### =

##### 1

##### 2

```
gT^2.
```
```
3)Massenberechnung: Zum Beispiel die Masse eines Drahtes der L ̈ange`mit
Massendichte ρ = ρ(x). Wie vernachl ̈assigen dabei die Ausdehnung iny- und
z-Richtung.
```
- Fallsρkonstant ist: Die Gesamtmasse istm=ρ`.
- Fallsρvariabel ist: Zerlege [0,`] innTeilintervalle [xj− 1 ,xj] und approximiere
    ρ(x) durchρ(xj− 1 ) auf [xj− 1 ,xj], wobeixj=j∆xund ∆x=`/nseien. Dann
    ist die Gesamtmasse

```
m= lim
n→∞
```
```
∑n
```
```
j=1
```
```
ρ(xj− 1 )∆x=
```
##### ∫`

```
0
```
```
ρ(x)dx.
```
Es kommt nicht so selten vor, dass man in den Ingenieurwissenschaften Funktionen
integrieren muss, die weder monoton noch stetig sind, etwa die Steuerspannung des
Kathodenstrahls einer Fernsehr ̈ohre:

```
x
− 2 − 1 0 1 2 3 4 5 6
− 1
```
##### 1


Wir erweitern daher unsere Integraldefinition noch etwas.

Definition 28.4(Integrierbare Funktion). 1) Wir nennenf : [a,b]→Rst ̈uckweise
stetig(bzw.st ̈uckweise monoton), falls es endlich viele Stellenx 0 ,x 1 ,...,xNgibt
mit
a=x 0 < x 1 < ... < xN=b,
so dassf : ]xj− 1 ,xj[ →R f ̈ur jedesj die Einschr ̈ankung einer stetigen (bzw.
monotonen) Funktionfj: [xj− 1 ,xj]→Rist.
2) Istfwie in 1), so definieren wir
∫b

```
a
```
```
f(x)dx:=
```
```
∫x 1
```
```
x 0
```
```
f 1 (x)dx+
```
```
∫x 2
```
```
x 1
```
```
f 2 (x)dx+...+
```
```
∫xN
```
```
xN− 1
```
```
fN(x)dx.
```
```
3) Funktionen wie in 1) heißenintegrierbar.
```
Bemerkung 28.5.Diese Definition von integrierbaren Funktionen deckt sich nicht ganz
mit der mathematischen Standardterminologie, dort ist die Klasse der integrierbaren
Funktionen etwas gr ̈oßer (und bildet einen Vektorraum).

Beispiel 28.6. 1) Die Funktionf: [0,1]→Rmit

```
f(x) =
```
##### {

```
0 f ̈urx= 0,
1
x f ̈ur 0< x≤^1
ist nicht stuckweise stetig, denn ̈ f: ]0,1[→Rl ̈asst sich nicht stetig auf [0,1] fortset-
zen und ist daher nicht die Einschr ̈ankung einer stetigen Funktionf 1 : [0,1]→R.
2) Die Funktionf: [0,2]→Rmit
```
```
f(x) =
```
##### 

##### 

##### 

```
1 f ̈urx∈[0,1[
2 f ̈urx= 1
3 f ̈urx∈]1,2]
ist stuckweise stetig, denn ̈
```
- f : ]0,1[→Rist die Einschr ̈ankung der stetigen Funktionf 1 : [0,1]→R,
    f 1 (x) = 1.
- f : ]1,2[→Rist die Einschr ̈ankung der stetigen Funktionf 2 : [1,2]→R,
    f 2 (x) = 3.

```
x
```
```
y
```
##### 1 2

##### 1

##### 2

##### 3

```
x
```
```
y
```
##### 1 2

##### 1

##### 2

##### 3

```
f 1
```
```
f 2
```

```
Dann ist
∫ 2
```
```
0
```
```
f(x)dx=
```
##### ∫ 1

```
0
```
```
f 1 (x)dx+
```
##### ∫ 2

```
1
```
```
f 2 (x)dx= 1(1−0) + 3(2−1) = 4.
```
```
Der Funktionswert inx= 1 geht nicht in das Integral ein.
```
Bemerkung 28.7. 1) Integrierbare Funktionen sind beschr ̈ankt und auf einem kom-
pakten Intervall [a,b] definiert.
2) Istf wie in Definition 28.4, so kannf inx 0 ,...,xN beliebige Werte annehmen,
die keinen Einfluss auf

```
∫b
af(x)dxhaben.
```
### 28.2 Rechenregeln

Wir vereinbaren ∫a

```
a
```
```
f(x)dx:= 0
```
(kein Fl ̈acheninhalt) und fur ̈ a < b
∫a

```
b
```
```
f(x)dx:=−
```
```
∫b
```
```
a
```
```
f(x)dx.
```
Satz 28.8(Rechenregeln).Seienf,g: [a,b]→Rintegrierbar undλ∈R. Dann gilt:

```
1) Linearit ̈at:
∫b
```
```
a
```
```
(f(x) +g(x))dx=
```
```
∫b
```
```
a
```
```
f(x)dx+
```
```
∫b
```
```
a
```
```
g(x)dx,
∫b
```
```
a
```
```
λf(x)dx=λ
```
```
∫b
```
```
a
```
```
f(x)dx.
```
```
2) Monotonie: Fallsf(x)≤g(x)f ̈ur allex∈[a,b], so ist
∫b
```
```
a
```
```
f(x)dx≤
```
```
∫b
```
```
a
```
```
g(x)dx.
```
```
3) Dreiecksungleichung: ∣
∣
∣∣
```
```
∫b
```
```
a
```
```
f(x)dx
```
##### ∣∣

##### ∣∣≤

```
∫b
```
```
a
```
```
|f(x)|dx.
```
```
4) F ̈ura < c < bist
∫b
```
```
a
```
```
f(x)dx=
```
```
∫c
```
```
a
```
```
f(x)dx+
```
```
∫b
```
```
c
```
```
f(x)dx.
```
Beweis. Die Aussagen lassen sich leicht fur Riemannsche Summen nachpr ̈ ̈ufen und blei-
ben im Grenzwert erhalten.


### 28.3 Das Integral als Mittelwert der Funktion

Aus der Monotonie des Integrals erhalten wir eine einfache Absch ̈atzung f ̈ur das Integral
vonf.

Satz 28.9(Absch ̈atzung des Integrals).Seif: [a,b]→Rintegrierbar undm≤f(x)≤
Mf ̈ur allex∈[a,b]. Dann gilt

```
m(b−a)≤
```
```
∫b
```
```
a
```
```
f(x)dx≤M(b−a).
```
Beweis. F ̈ur die konstante Funktiong: [a,b]→R,g(x) =m, istg(x)≤f(x) f ̈ur allex,
also

```
m(b−a) =
```
```
∫b
```
```
a
```
```
g(x)dx≤
```
```
∫b
```
```
a
```
```
f(x)dx.
```
Die zweite Ungleichung folgt genauso.

```
x
```
```
y
```
```
a b
```
```
m
```
##### M

```
M(b−a)
```
##### F

```
m(b−a)
```
Bemerkung 28.10. Es ist

```
1
b−a
```
```
∫b
```
```
a
```
```
f(x)dx= lim
n→∞
```
##### 1

```
b−a
```
```
∑n
```
```
j=1
```
```
f(xj− 1 )
b−a
n
```
```
= lim
n→∞
```
##### 1

```
n
```
```
∑n
```
```
j=1
```
```
f(xj− 1 ).
```
Die Summe^1 n

∑n
j=1f(xj−^1 ) ist genau der Mittelwert der Funktionswertef(x^0 ),f(x^1 ),
...,f(xn− 1 ). Daher kann das Integralb−^1 a

∫b
af(x)dxalsMittelwert der Funktionfange-
sehen werden. Der letzte Satz besagt also, dass das Integralmittel zwischen dem kleinsten
und gr ̈oßten Funktionswert liegt.

Satz 28.11 (Mittelwertsatz der Integralrechnung). Sei f : [a,b] → Rstetig. Dann
existiert einξ∈[a,b]mit

```
f(ξ) =
```
##### 1

```
b−a
```
```
∫b
```
```
a
```
```
f(x)dx.
```
Beweis. Als stetige Funktion nimmtf ihr Minimummund MaximumMin [a,b] an
(Satz 20.12). Mit der Integralabsch ̈atzung ist dann

```
m≤
```
##### 1

```
b−a
```
```
∫b
```
```
a
```
```
f(x)dx≤M.
```

Nach dem Zwischenwertsatz nimmtfjeden Wert zwischen Minimum und Maximum an,
d.h. es existiert einξ∈[a,b] mit

```
f(ξ) =
```
##### 1

```
b−a
```
```
∫b
```
```
a
```
```
f(x)dx,
```
wie behauptet.


Vorlesung 29

## 29 Integrationsregeln

Die Berechnung von Integralen mittels Riemann-Summen ist muhsam. Einfacher ist die ̈
Berechnung von Integralen mithilfe des Hauptsatzes der Differential- und Integralrech-
nung, der einen Zusammenhang zwischen Integration und Differentiation herstellt und
die Integration auf die Bestimmung einer sogenannten Stammfunktion zuruckf ̈ ̈uhrt.
Aus dem Hauptsatz erhalten wir n ̈utzliche Integrationsregeln zur Bestimmung von
Stammfunktionen zur Berechnung von Integralen: Dies sind die partielle Integration
(diese Vorlesung) und die Substitutionsregel (n ̈achste Vorlesung).

### 29.1 Stammfunktionen

Wir beginnen mit dem Begriff der Stammfunktion.

Definition 29.1 (Stammfunktion). Seif : D → Reine Funktion (D ⊆ R). Eine
differenzierbare FunktionF:D→Rheißt eineStammfunktionvonf, fallsF′(x) =f(x)
f ̈ur allex∈Dgilt.

Bemerkung 29.2. Auf Intervallen sind Stammfunktionen bis auf eine Konstante ein-
deutig:SeiF eine Stammfunktion vonf auf dem IntervallI. Ist dannGeine weitere
Stammfunktion vonf, so ist (G−F)′= G′−F′ =f−f = 0, also istG−F kon-
stant (Satz 23.7) und es existiert eine Konstantec ∈Rmit G(x)−F(x) = c, also
G(x) =F(x) +cf ̈ur allex∈I.

Definition 29.3(unbestimmtes Integral). Die Menge aller Stammfunktionen vonf :
[a,b]→Rwird mit

##### ∫

f(x)dxbezeichnet und heißt dasunbestimmte Integral vonf. Ist
F eineStammfunktion vonfist also
∫
f(x)dx={F+c|c∈R}.

Das unbestimmt Integral schreibt man oft nur als
∫
f(x)dx=F(x) +c, c∈Rbeliebig,

also ohne die Mengenklammern.


Das unbestimmte Integral ist also die L ̈osungsmenge der linearen GleichungdxdF(x) =
f(x). Wie bei linearen Gleichungssystemen hat man eine partikul ̈are L ̈osungFplus die
L ̈osungen der homogenen GleichungdxdF(x) = 0, die genau die Konstanten sind.

Beispiel 29.4. 1) Die Funktionf:R→R,f(x) =x, hat als eine Stammfunktion
F(x) =^12 x^2 , und dann auchG(x) =^12 x^2 + 3 oder H(x) =^12 x^2 −12000. Das
unbestimmte Integral vonfist
∫
f(x)dx=

##### 1

##### 2

```
x^2 +c, c∈Rbeliebig.
```
```
2) Wegen (ex)′=existexeine Stammfunktion vonex. Das unbestimmte Integral von
exist ∫
exdx=ex+c, c∈Rbeliebig.
```
```
3) Eine Stammfunktion von cos(x) ist sin(x). AuchF(x) = sin(x)+42 ist eine Stamm-
funktion von cos(x), dennF′(x) = cos(x).
```
### 29.2 Der Hauptsatz der Differential- und Integralrechnung

Wir kommen nun zum Hauptsatz der Differential- und Integralrechnung, dem zentralen
Satz zur Berechnung von Integralen. Wie sich herausstellt l ̈asst sich das bestimmte In-
tegral mit einer Stammfunktionen berechnen. Andersherum kann eine Stammfunktion
mittels Integration bestimmt werden.

Satz 29.5(Hauptsatz der Differential- und Integralrechnung).

```
1) Existenz einer Stammfunktion: Istf: [a,b]→Rstetig, so ist die durch
```
```
F: [a,b]→R, F(x) =
```
```
∫x
```
```
a
```
```
f(t)dt,
```
```
definierte Integralfunktion eine Stammfunktion vonf.
```
```
2) Berechnung des bestimmten Integrals: IstF : [a,b]→Reine Stammfunktion der
stetigen Funktionf: [a,b]→R, so gilt
∫b
```
```
a
```
```
f(x)dx=F
```
##### ∣

```
∣b
a:=F(b)−F(a).
```
Beweis. 1) Wir rechnen nach, dassF′(x) =f(x) gilt. Es ist

```
F(x+h)−F(x)
h
```
##### =

##### 1

```
h
```
```
(∫x+h
```
```
a
```
```
f(t)dt−
```
```
∫x
```
```
a
```
```
f(t)dt
```
##### )

##### =

##### 1

```
h
```
```
(∫x
```
```
a
```
```
f(t)dt+
```
```
∫x+h
```
```
x
```
```
f(t)dt−
```
```
∫x
```
```
a
```
```
f(t)dt
```
##### )

##### =

##### 1

```
h
```
```
∫x+h
```
```
x
```
```
f(t)dt.
```

```
Nach dem Mittelwertsatz der Integralrechnung (Satz 28.11) ist^1 h
```
```
∫x+h
x f(t)dt=
f(ξ) f ̈ur einξzwischenxundx+h. F ̈urh→0 ist dannξ→xund wir finden
```
```
F′(x) = lim
h→ 0
```
```
F(x+h)−F(x)
h
= lim
h→ 0
f(ξ) =f(x),
```
```
wobei wir noch einmal die Stetigkeit vonfverwendet haben.
```
```
2) DaF und
```
```
∫x
af(t)dtStammfunktionen vonfauf dem Intervall [a,b] sind, gibt es
eine Konstantec∈Rmit
```
```
F(x) =
```
```
∫x
```
```
a
```
```
f(t)dt+c.
```
```
F ̈urx=asehen wirF(a) = 0 +c. Setzen wir nunx=bein, erhalten wir
∫b
```
```
a
```
```
f(t)dt=F(b)−c=F(b)−F(a).
```
Fur ̈ F|bawird auch [F]baoderAhnliches verwendet. ̈
Der Hauptsatz der Differential- und Integralrechnung erm ̈oglicht es also, das be-
stimmte Integral einer stetigen Funktion ganz einfach durch auswerten einer Stamm-
funktion zu berechnen, falls man diese kennt.

Beispiel 29.6. 1) Eine Stammfunktion vonxist^12 x^2 , also

```
∫b
```
```
a
```
```
xdx=
```
##### 1

##### 2

```
x^2
```
##### ∣

```
∣b
a=
```
##### 1

##### 2

```
(b^2 −a^2 ).
```
```
Insbesondere ist
```
##### ∫ 1

```
0 xdx=
```
```
1
2 , was genau der Fl ̈acheninhalt des Dreiecks mit den
Ecken (0,0), (1,0) und (1,1) ist.
```
```
x
```
```
y
y=x
```
##### 1

##### 1

```
2) Eine Stammfunktion von sin ist−cos, daher ist
∫π
```
```
0
```
```
sin(x)dx=−cos(x)
```
##### ∣

```
∣π
0 =−cos(π) + cos(0) = 2.
```
```
Wenn Sie noch nicht ̈uberzeugt sein sollten, dass der Hauptsatz hilfreich ist, ver-
suchen Sie einmal, dieses Integral direkt mit der Definition zu berechnen, dann
m ̈ussen Sie also den folgenden Grenzwert berechnen:
∫π
```
```
0
```
```
sin(x)dx= lim
n→∞
```
```
∑n
```
```
j=1
```
```
sin
```
##### (

```
(j−1)
π
n
```
```
)π
n
```
##### .


##### 3)

```
∫b
ae
```
```
xdx=ex∣∣b
a=e
```
```
b−ea.
4)
```
```
∫b
a
```
```
1
1+x^2 dx= arctan(x)
```
##### ∣

```
∣b
a= arctan(b)−arctan(a). Speziell mita= 0 (undxstatt
b) erh ̈alt man eine Integraldarstellung f ̈ur den Arcus Tangens:
```
```
arctan(x) =
```
```
∫x
```
```
0
```
##### 1

```
1 +t^2
dt.
```
##### 5)

```
∫b
a
√^1
1 −x^2 dx= arcsin(x)
```
##### ∣

```
∣b
a= arcsin(b)−arcsin(a).
```
```
6) SeiH:R→R,H(x) =
```
##### {

```
0 f ̈urx < 0
1 f ̈urx≥ 0
```
```
, die Heavyside-Funktion. Seib >0 und
```
```
betrachteH: [−b,b]→R. Dann istHst ̈uckweise stetig, also integrierbar. Es gilt
```
```
F(x):=
```
```
∫x
```
```
−b
```
```
H(t)dt=
```
##### {

```
0 fur ̈ x < 0
x fur ̈ x≥ 0.
```
```
x
```
```
y
```
```
−b b
```
##### F

```
DaFeine Knickstelle inx= 0 hat, istFinx= 0 nicht differenzierbar, also istF
keine Stammfunktion vonH. Die Stetigkeit vonfim Hauptsatz der Differential-
und Integralrechnung ist also wesentlich.
```
### 29.3 Grundintegrale

Es bezeichneF:D→Reine Stammfunktion vonf:D→Rauf dem Definitionsbereich
D(Vereinigung von Intervallen).

```
1) F ̈urf(x) =xαmitα∈R\{− 1 }ist der Definitionsbereich
```
- D=R, fallsα∈N
- D=R\{ 0 }, fallsα∈Z\\N(z.B.x−^2 =x^12 )
- D= ]0,∞[ fallsα∈R\Z(allgemeine Potenz).
Dann istF(x) =α^1 +1xα+1, dennF′(x) =xα, also
∫
xαdx=

##### 1

```
α+ 1
xα+1+c, c∈R.
```
```
2) F ̈urf(x) =x−^1 =^1 xist der DefinitionsbereichD=R\{ 0 }. Eine Stammfunktion
istF(x) = ln|x|, denn
```
- f ̈urx >0 istF′(x) =dxd ln(x) =x^1 ,
- f ̈urx <0 ist mit der KettenregelF′(x) =dxdln(−x) =−^1 x(−1) =^1 x.


```
Daher ist ∫
1
x
dx= ln|x|+c, c∈R.
```
```
Achtung: ln(x) ist nur auf ]0,∞[ definiert und damit keine Stammfunktion auf
ganzD=R\{ 0 }.
```
```
3) F ̈urf(x) =x−^1 amita∈Rist der DefinitionsbereichD=R\{a}und dann
∫
1
x−a
dx= ln|x−a|+c mitc∈R.
```
Manche Stammfunktionen lassen sich nicht elementar ausdrucken (d.h. durch Poly- ̈
nome,ex, sin, cos, usw. und Addition, Multiplikation und Verkettung dieser Funktionen).
In diesem Fall geben die Integralfunktionen
”
neue“ Funktionen.

Beispiel 29.7. 1) Die Funktionf :R→R,f(x) = sin(x^2 ), ist stetig und besitzt
daher eine Stammfunktion. Diese l ̈asst sich nicht durch elementare Funktionen
ausdr ̈ucken, und f ̈uhrt auf ein sogenanntesFresnel-Integral, dass definiert ist durch

```
FesnelS(x) =
```
```
∫x
```
```
0
```
```
sin
```
```
(π
2
t^2
```
##### )

```
dt.
```
```
Das Fresnel-Integral spielt eine Rolle in der geometrischen Optik.
2) Ebenso kann die Stammfunktion vone−x
2
nicht durch elementare Funktionen an-
gegeben werden. Die Stammfunktion ist dieGaußsche Fehlerfunktion(
”
error func-
tion“) oderNormalverteilung:
```
```
erf(x) =
```
##### 2

##### √

```
π
```
```
∫x
```
```
0
```
```
e−t
2
dt.
```
```
Sie spielt eine wichtige Rolle in der Theorie von Diffusionsprozessen und in der
Statistik.
3)
```
```
∫sin(x)
x dxist ebenfalls nicht durch elementare Funktionen darstellbar und definiert
denIntegralsinus
Si(x) =
```
```
∫x
```
```
0
```
```
sin(t)
t
dt.
```
### 29.4 Partielle Integration

Zusammengesetzte Funktionen lassen sich recht einfach ableiten dank der Produkt-,
Quotienten- und Kettenregel. Bei der Integration gibt es keine so einfachen Rechenregeln,
und bereits”einfache“ Funktionen k ̈onnen auf”komplizierte“ Stammfunktionen f ̈uhren.
Dennoch k ̈onnen wir aus den Regeln f ̈ur das Ableiten auch Regeln f ̈ur die Integration
gewinnen, die bei der Berechnung von bestimmen und unbestimmten Integralen helfen
k ̈onnen.


Die partielle Integration (= Integration nach Teilen) entspricht der Produktregel
beim Ableiten. Sindu,v: [a,b]→Rdifferenzierbar mit stetigen Ableitungen, so ist mit
der Produktregel
(uv)′(x) =u′(x)v(x) +u(x)v′(x). (29.1)

Daher istuveine Stammfunktion vonu′v+uv′und es gilt f ̈ur die unbestimmten Integrale
∫
u′(x)v(x)dx=u(x)v(x)−

##### ∫

```
u(x)v′(x)dx.
```
Andererseits folgt durch Integration der Produktregel (29.1) mit dem Hauptsatz

```
(uv)
```
```
∣∣b
a=
```
```
∫b
```
```
a
```
```
(uv)′(x)dx=
```
```
∫b
```
```
a
```
```
u′(x)v(x)dx+
```
```
∫b
```
```
a
```
```
u(x)v′(x)dx,
```
also ∫
b
a

```
u′(x)v(x)dx= (uv)
```
```
∣∣b
a−
```
```
∫b
```
```
a
```
```
u(x)v′(x)dx.
```
Wir fassen das als Satz zusammen.

Satz 29.8 (Partielle Integration). Sind u,v : [a,b] →Rdifferenzierbar mit stetigen
Ableitungen, so gelten
∫
u′(x)v(x)dx=u(x)v(x)−

##### ∫

```
u(x)v′(x)dx
```
und ∫
b
a

```
u′(x)v(x)dx= (uv)
```
##### ∣

```
∣b
a−
```
```
∫b
```
```
a
```
```
u(x)v′(x)dx.
```
Beispiel 29.9. 1) Wir wollen

```
∫b
axe
```
```
xdxberechnen. Wir versuchen den Ansatz
```
```
u′(x) =x, v(x) =ex.
```
```
Dann istu(x) =^12 x^2 undv′(x) =exund partielle Integration liefert
∫b
```
```
a
```
```
xexdx=
```
##### 1

##### 2

```
x^2 ex
```
```
∣∣b
a−
```
```
∫b
```
```
a
```
##### 1

##### 2

```
x^2 exdx,
```
```
das neue Integral ist aber komplizierter als das erste. Wir versuchen nun
```
```
u′(x) =ex, v(x) =x,
```
```
dann sindu(x) =exundv′(x) = 1, also mit partieller Integration
∫b
```
```
a
```
```
xexdx=xex
```
##### ∣

```
∣b
a−
```
```
∫b
```
```
a
```
```
exdx= (x−1)ex
```
##### ∣

```
∣b
a.
```
```
Merke: Hier war es also hilfreich, dass Polynom (x) abzuleiten, da das resultierende
Integral leichter zu berechnen war. Das ist oft, aber nicht immer, so.
```

2) Berechne

##### ∫

```
x^2 cos(x)dx. Die Idee ist, mit partieller Integration dasx^2
”
wegzudif-
ferenzieren“: Mitv(x) =x^2 undu′(x) = cos(x) ist
∫
x^2 cos(x)dx=x^2 sin(x)−
```
##### ∫

```
2 xsin(x)dx,
```
```
und mit erneuter partieller Integration (u′(x) =−sin(x),v(x) = 2x) ist
∫
x^2 cos(x)dx=x^2 sin(x)+2xcos(x)−
```
##### ∫

```
2 cos(x)dx= (x^2 −2) sin(x)+2xcos(x)+c
```
```
mit beliebigemc∈R.
```
3) Wir berechnen

##### ∫

```
cos(x)^2 dxmit partieller Integration und u′(x) = cos(x) und
v(x) = cos(x), alsou(x) = sin(x) undv′(x) =−sin(x):
∫
cos(x)^2 dx= sin(x) cos(x) +
```
##### ∫

```
sin(x)^2 dx= sin(x) cos(x) +
```
##### ∫

```
(1−cos(x)^2 )dx
```
```
= sin(x) cos(x) +
```
##### ∫

```
1 dx−
```
##### ∫

```
cos(x)^2 dx
```
```
also
∫
cos(x)^2 dx=
```
##### 1

##### 2

##### (

```
sin(x) cos(x) +
```
##### ∫

```
1 dx
```
##### )

##### =

##### 1

##### 2

```
(sin(x) cos(x) +x+c)
```
```
mit beliebigemc∈R. Mit der Stammfunktion lassen sich bestimmte Integrale nun
ganz einfach berechnen, z.B. ̈uber [0, 2 π]:
∫ 2 π
```
```
0
```
```
cos(x)^2 dx=
```
##### 1

##### 2

```
(sin(x) cos(x) +x)|^20 π=π.
```
4) Wir berechnen eine Stammfunktion des Logarithmus. Dazu verwenden wir einen
praktischen Trick: Wir setzenu′(x) = 1 undv(x) = ln(x) und erhalten mitu(x) =
xundv′(x) =x^1
∫
ln(x)dx=xln(x)−

##### ∫

```
x
```
##### 1

```
x
```
```
dx=xln(x)−x+c,
```
```
mit einer Konstantenc∈R.
```

```
Einige bekannte Stammfunktionen
```
```
f(x) F(x) Bemerkungen
```
```
xn n^1 +1xn+1 n∈N
xn n^1 +1xn+1 x 6 = 0,n∈Zmitn 6 =− 1
xa a+1^1 xa+1 x >0,a∈Rmita 6 =− 1
1
x ln|x| x^6 = 0
eax^1 aeax
```
```
sin(x) −cos(x)
cos(x) sin(x)
1
cos(x)^2 tan(x) x^6 =
```
π
2 +kπ,k∈Z
−sin(^1 x) 2 cot(x) x 6 =kπ,k∈Z
1
√
1 −x^2

```
arcsin(x) |x|< 1
1
1 +x^2
```
```
arctan(x)
```
sinh(x) cosh(x)
cosh(x) sinh(x)
1
√
1 +x^2

```
arsinh(x) = ln(x+
```
##### √

1 +x^2 )
1
√
x^2 − 1

```
ln|x+
```
##### √

x^2 − 1 | |x|> 1
1
√
x^2 − 1

```
arcosh(x) = ln(x+
```
##### √

```
x^2 −1) x∈]1,∞[
1
1 −x^2
artanh(x) =^12 ln
```
```
(1+x
1 −x
```
##### )

```
|x|< 1
```
```
Tabelle 29.1: Einige Stammfunktionen.
```

## Vorlesung 30

# Integrationsregeln 2 und

# Integration komplexer Funktionen

Nachdem wir in der letzten Vorlesung bereits die partielle Integration kennen gelernt
haben, behandeln wir nun eine zweite wichtige Integrationsregel: Die Substitutionsregel.
Anschließend erkl ̈aren wir die Integration komplexwertiger Funktionen.

### 30.1 Substitutionsregel

Die Substitutionsregel (= Ersetzungsregel) entspricht der Kettenregel beim Ableiten.
SeienF : [c,d]→Rundg: [a,b]→[c,d] differenzierbar mit stetigen Ableitungen und
f=F′. Nach der Kettenregel gilt

```
d
dx
F(g(x)) =F′(g(x))g′(x) =f(g(x))g′(x).
```
Das unbestimmte Integral vonf(g(x))g′(x) ist dann
∫
f(g(x))g′(x)dx=F(g(x)) +c, c∈R,

und das bestimmte Integral ist
∫b

```
x=a
```
```
f(g(x))g′(x)dx=F(g(b))−F(g(a)) =
```
```
∫g(b)
```
```
t=g(a)
```
```
f(t)dt.
```
Merkregel: Man substituiertt=g(x), so dass dxdt=g′(x), also formaldt=g′(x)dx.

1. Version der Substitutionsregel zur Berechnung von

##### ∫

```
f(g(x))g′(x)dx:
```
```
1) Substituieret=g(x) unddt=g′(x)dx(Formaldxdt=g′(x) mitdxmultiplizieren):
∫
f(g(x))g′(x)dx=
```
##### ∫

```
f(t)dt.
```

```
2) Berechne eine Stammfunktion vonf:
```
##### ∫

```
f(t)dt=F(t) +c,c∈R.
3) Rucksubstitution ̈ t=g(x) ergibt
∫
f(g(x))g′(x)dx=F(g(x)) +c, c∈R.
```
Zur Berechnung des bestimmten Integrals

```
∫b
af(g(x))g
```
```
′(x)dxkann man die Grenzen di-
```
rekt mit transformieren und spart sich die R ̈ucksubstitution:

```
1) Substituieret=g(x) unddt=g′(x)dx:
∫b
```
```
x=a
```
```
f(g(x))g′(x)dx=
```
```
∫g(b)
```
```
t=g(a)
```
```
f(t)dt
```
```
2) Berechne eine StammfunktionFvonf, so dass
∫b
```
```
a
```
```
f(g(x))g′(x)dx=
```
```
∫g(b)
```
```
g(a)
```
```
f(t)dt=F
```
```
∣∣g(b)
g(a)=F(g(b))−F(g(a)).
```
Beispiel 30.1. 1) Berechne

##### ∫

```
cos(x^2 )2xdx. Substituieret=x^2 , alsodt= 2xdxund
∫
cos(x^2 )2xdx=
```
##### ∫

```
cos(t)dt= sin(t) +c= sin(x^2 ) +c,
```
```
mitc∈R.
2) Berechne
```
##### ∫ 2

```
0 cos(x
```
(^2) )2xdx. Substituieret=x (^2) , alsodt= 2xdxund
∫ 2
x=0
cos(x^2 )2xdx=

##### ∫ 4

```
t=0
```
```
cos(t)dt= sin(t)
```
##### ∣∣ 4

```
0 = sin(4).
```
```
3) Berechne
```
##### ∫ 3

```
0
```
```
x
(1+x^2 )^2 dx. Substituieret= 1 +x
```
(^2) , dann istdt= 2xdxund
∫ 3
x=0
x
(1 +x^2 )^2
dx=

##### ∫ 10

```
t=1
```
##### 1

```
t^2
```
##### 1

##### 2

```
dt=−
```
##### 1

##### 2

##### 1

```
t
```
##### ∣∣

##### ∣

```
10
1
```
##### =

##### 9

##### 20

##### .

```
Alternativ kann man aucht=x^2 substituieren.
4) Mit der Substitutiont=g(x) istdt=g′(x)dxund
∫
g′(x)
g(x)
```
```
dx=
```
##### ∫

##### 1

```
t
```
```
dt= ln|t|+c= ln|g(x)|+c, c∈R.
```
```
5) Berechne
```
∫ 4 xarcsin(x (^2) )
√
1 −x^4 dx. Wir substituierenφ=x
(^2) , haben alsodφ= 2xdx, und
∫
4 xarcsin(x^2 )
√
1 −x^4
dx=

##### ∫

```
2 arcsin(φ)
√
1 −φ^2
```
```
dφ.
```
```
Nun substituieren wirt= arcsin(φ), alsodt=√ 11 −φ 2 dφ, und
∫
4 xarcsin(x^2 )
√
1 −x^4
```
```
dx=
```
##### ∫

```
2 arcsin(φ)
√
1 −φ^2
```
```
dφ=
```
##### ∫

```
2 tdt=t^2 +c, c∈R.
```
```
Mit Rucksubstitution folgt ̈
```
∫ 4 xarcsin(x (^2) )
√
1 −x^4 dx= arcsin(x
(^2) ) (^2) +c,c∈R.


2. Version der Substitutionsregel zur Berechnung von

##### ∫

```
f(x)dx.
```
```
1) Substituierex=g(t) unddx=g′(t)dt, mit einemumkehrbaremg:
∫
f(x)dx=
```
##### ∫

```
f(g(t))g′(t)dt.
```
```
2) Berechne eine Stammfunktion:
```
##### ∫

```
f(g(t))g′(t)dt=H(t) +c,c∈R.
```
```
3) R ̈ucksubstitution: Aufl ̈osen vonx=g(t) nacht, alsot=g−^1 (x) (Umkehrfunktion),
und einsetzen ergibt
∫
f(x)dx=H(g−^1 (x)) +c, c∈R.
```
Berechnet man ein bestimmtes Integral, so kann man wieder die R ̈ucksubstitution sparen,
wenn man die Grenzen transformiert:

```
1) Substituierex=g(t) unddx=g′(t)dt:
∫b
```
```
x=a
```
```
f(x)dx=
```
∫g− (^1) (b)
t=g−^1 (a)
f(g(t))g′(t)dt.
2) Berechne eine StammfunktionH:
∫g− (^1) (b)
g−^1 (a)f(g(t))g
′(t)dt=H(g− (^1) (b))−H(g− (^1) (a)).
Beispiel 30.2. 1) Berechne die Fl ̈ache des Kreises mit Radiusr >0. Auf dem Kreis-
rand giltx^2 +y^2 =r^2 , alsoy=±

##### √

```
r^2 −x^2. Somit ist die Fl ̈ache der Kreisscheibe
```
```
I= 2
```
```
∫r
```
```
−r
```
##### √

```
r^2 −x^2 dx.
```
```
Substituierex=rsin(t), so dassdx=rcos(t)dtist und
```
##### I= 2

```
∫π/ 2
```
```
−π/ 2
```
##### √

```
r^2 −r^2 sin(t)^2 rcos(t)dt= 2r^2
```
```
∫π/ 2
```
```
−π/ 2
```
##### √

```
cos(t)^2 cos(t)dt.
```
```
Da cos(t)≥0 f ̈ur−π/ 2 ≤t≤π/2 ist
```
##### √

```
cos(t)^2 = cos(t) und somit
```
```
I= 2r^2
```
```
∫π/ 2
```
```
−π/ 2
```
```
cos(t)^2 dt=r^2 (cos(t) sin(t) +t)
```
##### ∣

```
∣π/^2
−π/ 2 =πr
```
(^2) ,
wobei wir die Stammfunktion von cos(t)^2 aus Beispiel 29.9 verwendet haben.
2) Berechne

##### ∫

```
arcsin(x)dx. Wir substituierenx = sin(t), alsot = arcsin(x), und
erhaltendx= cos(t)dtund
∫
arcsin(x)dx=
```
##### ∫

```
tcos(t)dt.
```

```
Dieses Integral k ̈onnen wir mit partieller Integration l ̈osen (u′(x) = cos(t) und
v(x) =t):
∫
tcos(t)dt= sin(t)t−
```
##### ∫

```
sin(t)dt= sin(t)t+cos(t)+c= sin(t)t+
```
##### √

```
1 −sin(t)^2 +c.
```
```
(Dabei haben wir cos(t)≥0 verwendet, was f ̈urt= arcsin(x)∈[−π/ 2 ,π/2] gilt.)
Mit Rucksubstitution erhalten wir ̈
∫
arcsin(x)dx=xarcsin(x) +
```
##### √

```
1 −x^2 +c, c∈R.
```
Als letztes betrachten wir noch ein Beispiel, das in Vorlesung 37 wichtig werden wird. Wir werden
die Aussage dort in Satz 37.4 auf anderem Wege nachrechnen.
Beispiel 30.3.SeiT >0 (Periode) undω=^2 Tπ(Kreisfrequenz). Dann gilt f ̈urk,`∈N:

```
2
T
```
```
∫T
0
```
```
cos(kωx) cos(`ωx)dx=
```
```



```
```
2 k=`= 0,
1 k=` > 0
0 k 6 =`
2
T
```
```
∫T
0
```
```
sin(kωx) sin(`ωx)dx=
```
```
{
1 k=` > 0
0 sonst
2
T
```
```
∫T
0
```
```
sin(kωx) cos(`ωx)dx= 0.
```
Wir beginnen mit den ersten beiden Formeln. Im Fallk=`= 0 ist
∫T
0 cos(0ωx)

(^2) dx=∫T
∫T^01 dt=Tund
0 sin(0ωx)
(^2) dx=∫T
0 0 dx= 0.
Sei nunk=` >0. Mit der Substitutiont=kωxistdt=kω dx, also
2
T
∫T
0
cos(kωx)^2 dx=T^2 kω^1
∫k 2 π
0
cos(t)^2 dt= 2 πk^212 (sin(t) cos(t) +t)
∣∣ 2 πk
0 = 1
und dann
2
T
∫T
0
sin(kωx)^2 dx=T^2
∫T
0
(1−cos(kωx)^2 )dx= 2−T^2
∫T
0
cos(kωx)^2 dx= 1.
Seien nunk 6 =`. Alle anderen F ̈alle erhalten wir wie folgt. Zun ̈achst ist fur ̈ m∈Z\{ 0 }:
2
T
∫T
0
cos(mωx)dx=T^2 mω^1 sin(mωx)
∣∣T
0 =
2
2 πm(sin(m^2 π)−sin(0)) = 0.
Aus den Additionstheorem von Sinus und Cosinus erhalten wir
cos(α) cos(β) =^12 (cos(α+β) + cos(α−β)),
sin(α) sin(β) =^12 (cos(α+β)−cos(α−β)).
(Additionstheoreme auf der rechten Seite anwenden und vereinfachen.) F ̈urα=kωxundβ=`ωxsind
dannα+β= (k+`)ωxundα−β= (k−`)ωx. Dak+` 6 = 0 undk−` 6 = 0 erhalten wir 0 nach
Integration. Damit haben wir die ersten beiden Formeln nachgerechnet.
Die letzte Formel ist noch einfacher. F ̈urm∈Z\{ 0 }ist
2
T
∫T
0
sin(mωx)dx=−T^2 mω^1 cos(mωx)
∣∣T
0 =−
2
2 πm(cos(m^2 π)−cos(0)) = 0
und das Ergebnis ist auch f ̈urm= 0 richtig. Da
sin(α) cos(β) =^12 (sin(α+β) + sin(α−β))
erhalten wir auch die dritte Formel.


### 30.2 Integration komplexer Funktionen

Komplexe Funktionen einer reellen Variablen integriert man nach Real- und Imagin ̈arteil
getrennt.

Definition 30.4(Integral komplexwertiger Funktionen). Seif: [a,b]→Cund seien
u,v : [a,b]→ Rmit u(x) = Re(f(x)) undv(x) = Im(f(x)) f ̈ur allex∈ [a,b], d.h.
f=u+iv. Sindu,vintegrierbar, so ist

```
∫b
```
```
a
```
```
f(x)dx:=
```
```
∫b
```
```
a
```
```
u(x)dx+i
```
```
∫b
```
```
a
```
```
v(x)dx.
```
Zum Beispiel ist f ̈urf(x) =eixdannu(x) = Re(eix) = cos(x) undv(x) = Im(eix) =
sin(x).

Bemerkung 30.5. 1) Der Hauptsatz der Differential- und Integralrechnung bleibt
richtig: IstF: [a,b]→Ceine Stammfunktion vonf(d.h.F′=f), so gilt:
∫b

```
a
```
```
f(x)dx=F(b)−F(a).
```
```
Weiter bleiben richtig: partielle Integration, die Substitutionsregel, sowie die Linea-
rit ̈at, Dreiecksungleichung und das Aufteilen des Integrals an einem Zwischenpunkt
(1),2)und4)in Satz 28.8).
```
```
2) Hingegen sind f ̈ur komplexwertige Funktionen nicht mehr gultig: die Monotonie ̈
(2)in Satz 28.8), die Integralabsch ̈atzung (Satz 28.9) und der Mittelwertsatz der
Integralrechnung (Satz 28.11).
```
Beispiel 30.6. 1) Seif:R→C,f(x) = (x+i)^2. Das unbestimmte Integral vonf
ist
∫
f(x)dx=

##### ∫

```
(x+i)^2 dx=
```
##### ∫

```
(x^2 +i 2 x−1)dx=
```
##### ∫

```
(x^2 −1)dx+i
```
##### ∫

```
2 xdx
```
```
=
```
##### 1

##### 3

```
x^3 −x+ix^2 +c, c∈C.
```
```
Alternativ istF(x) =^13 (x+i)^3 eine Stammfunktion vonf, dennF′(x) =dxd^13 (x+
i)^3 = (x+i)^2. Dies widerspricht nicht der ersten Rechnung, denn
```
```
F(x) =
```
##### 1

##### 3

```
(x+i)^3 =
```
##### 1

##### 3

```
(x^3 + 3ix^2 − 3 x−i) =
```
##### 1

##### 3

```
x^3 −x+ix^2 −
i
3
```
##### ,

```
also istFdie Stammfunktion mitc=− 3 i.
```
```
2) Seif:R→C,f(x) =x^1 −i. Naiv betrachtet istF(x) = ln|x−i|ein Kandidat f ̈ur
eine Stammfunktion vonf.
```

```
Probe: Es istF(x) =^12 ln(x^2 + 1), also
```
```
F′(x) =
```
##### 1

##### 2

```
2 x
x^2 + 1
```
##### =

```
x
x^2 + 1
```
```
6 =f(x),
```
```
d.h.Fist keine Stammfunktion vonf. (Das war auch vorher klar:Fist reellwertig,
so dass auchF′reellwertig ist, aberfist komplexwertig.)
Richtig geht es so:
∫
1
x−i
dx=
```
##### ∫

```
x+i
x^2 + 1
dx=
```
##### ∫

```
x
x^2 + 1
dx+i
```
##### ∫

##### 1

```
x^2 + 1
dx
```
```
=
```
##### 1

##### 2

```
ln(x^2 + 1) +iarctan(x) +c, c∈C.
```
Satz 30.7. Wie im Reellen ist f ̈ur allez∈C,z 6 = 0,
∫
ezxdx=

##### 1

```
z
ezx+c, c∈C.
```
Beweis. F ̈urz=a+ibmit reellena,bist

```
ezx=eax+ibx=eax(cos(bx) +isin(bx)),
```
also gilt
d
dxe

```
zx=aeax(cos(bx) +isin(bx)) +eaz(−bsin(bx) +ibcos(bx))
```
```
= (a+ib)eax(cos(bx) +isin(bx)) =zezx.
```
Daher ist^1 zezxeine Stammfunktion vonezx.

Die Integration komplexwertiger Funktionen eines reellen Arguments ist vor allem
bei der Fourieranalyse von Schwingungsvorg ̈angen in der Mechanik, Akustik oder Elek-
trotechnik wichtig. Darauf zielt das folgende Beispiel.

Beispiel 30.8.Seienk,`∈ZundT >0,ω=^2 Tπ. F ̈urk=`ist
∫T

```
0
```
```
eikωte−i`ωtdt=
```
##### ∫T

```
0
```
```
ei(k−`)ωtdt=
```
##### ∫T

```
0
```
```
1 dt=T.
```
Fur ̈ k 6 =`ist
∫T

```
0
```
```
eikωte−i`ωtdt=
```
##### ∫T

```
0
```
```
ei(k−`)ωtdt=
```
##### 1

```
i(k−`)ω
```
```
ei(k−`)ωt
```
##### ∣

##### ∣T

```
0
```
```
=
```
##### 1

```
i(k−`)ω
(ei(k−`)ωT−1) =
```
##### 1

```
i(k−`)ω
(ei(k−`)2π−1) = 0.
```
Daher ist
1
T

##### ∫T

```
0
```
```
eikωte−i`ωtdt=
```
##### {

```
1 k=`
0 k 6 =`.
```
Durch Vergleich von Real- und Imagin ̈arteil findet man die Formeln aus Beispiel 30.3 wie-
der. Vergleichen Sie noch einmal, wie einfach die Rechnung im Komplexen im Vergleich
zur Rechnung im Reellen ist.


## Vorlesung 31

# Uneigentliche Integrale und

# Integration rationaler Funktionen

Integrierbare Funktionen sind immer beschr ̈ankt und auf einem kompakten Intervall
[a,b] gegeben. Wir lernen, wie man Funktionen auf einem unbeschr ̈ankten Definitions-
bereich integriert oder unbeschr ̈ankte Funktionen integriert. Weiter besprechen wir die
Integration rationaler Funktionen.

### 31.1 Unbeschr ̈ankter Definitionsbereich

Seif: [0,∞[→R,f(x) =1+^1 x 2. Hat Die Fl ̈ache zwischen dem Graphen vonfund der
x-Achse endlichen Inhalt?

```
x
```
```
y
```
##### 1 2 3 4 5

```
1 Graph von1+^1 x 2
```
Definition 31.1(Uneigentliches Integraluber unbeschr ̈ ̈ankten Definitionsbereich).

```
1) Seif: [a,∞[→Rintegrierbar ̈uber jedem Intervall [a,b],b > a. Dann heißt
∫∞
```
```
a
```
```
f(x)dx:= lim
b→∞
```
```
∫b
```
```
a
```
```
f(x)dx
```
```
dasuneigentliche Integralvonf ̈uber [a,∞[. Falls der Grenzwert existiert, so heißt
funeigentlich integrierbarauf [a,∞[.
```
```
2) Seif: ]−∞,b]→Rintegrierbar ̈uber jedem Intervall [a,b] mita < b. Dann heißt
∫b
```
```
−∞
```
```
f(x)dx= lim
a→−∞
```
```
∫b
```
```
a
```
```
f(x)dx
```

```
dasuneigentliche Integral vonf ̈uber ]−∞,b]. Falls der Grenzwert existiert, so
heißtfuneigentlich integrierbarauf ]−∞,b].
```
Man sagt, das uneigentliche Integralexistiert, genau dann wenn der Grenzwert existiert.
Man sagt auch, dass ein uneigentliches Integralkonvergiert, falls der Grenzwert existiert,
andernfallsdivergiertes.

```
Das uneigentliche Integral kann existieren oder auch nicht.
```
Beispiel 31.2. 1) Es ist

```
∫∞
```
```
0
```
##### 1

```
1 +x^2
dx= lim
b→∞
```
```
∫b
```
```
0
```
##### 1

```
1 +x^2
dx= lim
b→∞
arctan(x)
```
##### ∣

```
∣b
0 = limb→∞arctan(b) =
```
```
π
2
```
##### .

```
Achtung: Schreiben Sie nicht arctan(x)
```
##### ∣

##### ∣∞

```
0 , sondern wie oben limb→∞arctan(x)
```
##### ∣

```
∣b
0.
Genauso ist
∫ 0
```
```
−∞
```
##### 1

```
1 +x^2
```
```
dx= lim
a→−∞
```
##### ∫ 0

```
a
```
##### 1

```
1 +x^2
```
```
dx= lim
a→−∞
arctan(x)
```
##### ∣

##### ∣^0

```
a= lima→−∞−arctan(a)
```
```
=
π
2
```
##### .

```
2) F ̈urα >0 ist
∫∞
```
```
0
```
```
e−αxdx= lim
b→∞
```
```
∫b
```
```
0
```
```
e−αxdx= lim
b→∞
```
##### −

##### 1

```
α
e−αx
```
```
∣∣b
0 = limb→∞
```
##### (

##### −

##### 1

```
α
e−αb+
```
##### 1

```
α
```
##### )

##### =

##### 1

```
α
```
##### ,

```
d.h. das uneigentliche Integral existiert.
```
```
3) Seif: [1,∞[→R,f(x) =x^1. Dann ist
∫∞
```
```
1
```
##### 1

```
x
```
```
dx= lim
b→∞
```
```
∫b
```
```
1
```
##### 1

```
x
```
```
dx= lim
b→∞
```
```
ln(x)
```
##### ∣

```
∣b
1 = limb→∞ln(b) =∞,
```
```
also existiert das uneigentliche Integral vonfnicht.
```
```
x
```
```
y
```
##### 1 2 3 4

##### 1

##### 2

```
Graph von^1 x
```

```
4) Seif : [1,∞[→R,f(x) =x−αmitα >0 undα 6 = 1. Eine Stammfunktion ist
1
1 −αx
1 −α. Dann ist
```
```
∫∞
```
```
1
```
##### 1

```
xα
```
```
dx= lim
b→∞
```
```
∫b
```
```
1
```
##### 1

```
xα
```
```
dx= lim
b→∞
```
##### 1

```
1 −α
```
```
x^1 −α
```
##### ∣∣

##### ∣

```
b
1
```
```
= lim
b→∞
```
##### (

##### 1

```
1 −α
```
```
b^1 −α+
```
##### 1

```
α− 1
```
##### )

##### .

```
F ̈ur 0< α <1 ist limb→∞b^1 −α=∞und fur ̈ α >1 ist limb→∞b^1 −α= 0. Daher
gilt
∫
1
xα
dx=
```
##### {

```
1
α− 1 fallsα >^1
existiert nicht falls 0< α < 1.
```
Integraleuber ganz ̈ R.

Definition 31.3. (Integral ̈uber ganzR) Istf:R→R, so setzen wir

```
∫∞
```
```
−∞
```
```
f(x)dx:=
```
##### ∫ 0

```
−∞
```
```
f(x)dx+
```
##### ∫∞

```
0
```
```
f(x)dx,
```
falls beide uneigentlichen Integrale existieren.

Bemerkung 31.4.Existiert

##### ∫∞

```
−∞f(x)dx, so gilt
∫∞
```
```
−∞
```
```
f(x)dx=
```
```
∫c
```
```
−∞
```
```
f(x)dx+
```
##### ∫∞

```
c
```
```
f(x)dx
```
f ̈ur jedesc∈R.

Beispiel 31.5. 1) Es gilt

```
∫∞
```
```
−∞
```
##### 1

```
1 +x^2
```
```
dx=
```
##### ∫ 0

```
−∞
```
##### 1

```
1 +x^2
```
```
dx+
```
##### ∫∞

```
0
```
##### 1

```
1 +x^2
```
```
dx=
π
2
```
##### +

```
π
2
```
```
=π,
```
```
vergleiche Beispiel 31.2.
```
```
2) Achtung: Zum Beispiel f ̈urf:R→R,f(x) =x, existiert der Grenzwert
```
```
lim
b→∞
```
```
∫b
```
```
−b
```
```
xdx= lim
b→∞
```
##### 1

##### 2

```
x^2
```
```
∣∣b
−b= 0,
```
```
aber das uneigentliche Integral
```
##### ∫∞

```
−∞xdxexistiert nicht, da
∫∞
```
```
0
```
```
xdx= lim
b→∞
```
```
∫b
```
```
0
```
```
xdx= lim
b→∞
```
##### 1

##### 2

```
x^2
```
```
∣∣b
0 =∞
```
```
nicht existiert. (Auch
```
##### ∫ 0

```
−∞xdxexistiert nicht.)
```

### 31.2 Unbeschr ̈ankte Funktion

Definition 31.6(Uneigentliches Integral von unbeschr ̈ankter Funktion).Seif: ]a,b]→
Rintegrierbar auf [c,a] f ̈ur allec∈]a,b]. Dann heißt

```
∫b
```
```
a
```
```
f(x)dx= lim
c↘a
```
```
∫b
```
```
c
```
```
f(x)dx
```
dasuneigentliches Integralvonfuber ] ̈ a,b]. Falls dieser Grenzwert existiert, so heißtf
uneigentlich integrierbarauf ]a,b].
Istf: [a,b[→Rintegrierbar auf [a,c] f ̈ur jedesc∈[a,b[, so definiert man genauso
∫b

```
a
```
```
f(x)dx= lim
c↗b
```
```
∫c
```
```
a
```
```
f(x)dx.
```
Man sagt, das uneigentliche Integralexistiert, genau dann wenn der Grenzwert existiert.
Man sagt auch, dass ein uneigentliches Integralkonvergiert, falls der Grenzwert existiert,
andernfallsdivergiertes.

```
Wie vorher kann das uneigentliche Integral vonfexistieren oder auch nicht.
```
Beispiel 31.7. 1) Es ist

```
∫ 1
```
```
0
```
##### 1

```
x
dx= lim
c↘ 0
```
##### ∫ 1

```
c
```
##### 1

```
x
dx= lim
c↘ 0
ln(x)
```
##### ∣∣ 1

```
c= limc↘ 0 −ln(c) =∞,
```
```
also existiert dieses uneigentliche Integral nicht.
```
```
x
```
```
y
```
##### 1 2

##### 1

##### 2

##### 3

##### 4

```
Graph von^1 x
```
```
2) F ̈urα >0,α 6 = 1, ist
∫ 1
```
```
0
```
##### 1

```
xα
```
```
dx= lim
c↘ 0
```
##### ∫ 1

```
c
```
##### 1

```
xα
```
```
dx= lim
c↘ 0
```
##### 1

```
1 −α
```
```
x^1 −α
```
##### ∣

##### ∣^1

```
c= limc↘ 0
```
##### 1

```
1 −α
```
```
(1−c^1 −α)
```
##### =

##### { 1

```
1 −α fur 0 ̈ < α <^1
existiert nicht fur ̈ α > 1.
```

```
Insbesondere existiert
∫∞
```
```
0
```
##### 1

```
xα
```
```
dx:=
```
##### ∫ 1

```
0
```
##### 1

```
xα
```
```
dx
︸ ︷︷ ︸
existiert f ̈urα< 1
```
##### +

##### ∫∞

```
1
```
##### 1

```
xα
```
```
dx
︸ ︷︷ ︸
existiert f ̈urα> 1
```
```
f ̈ur keinα.
```
```
Fur die Integrale ̈
```
##### ∫ 1

```
0
```
```
1
xαdxund
```
##### ∫∞

```
1
```
1
xαdxist alsoα= 1 der kritische Parameter:
F ̈urα= 1 existieren beide Integrale nicht. Ist aber die Funktion<x^1 , so existiert das
uneigentliche Integral. In ]0,1] ist das f ̈urα <1 der Fall, und in [1,∞[ fur ̈ α >1.

### 31.3 Integration rationaler Funktionen

Wir bestimmen Stammfunktionen von rationalen Funktionenf(x) = pq((xx)) mit reellen
Polynomenpundq.
Eine (komplexe) Polynomdivision und Partialbruchzerlegung (Vorlesung 9) liefert
p/qals Summe von Bestandteilen der Form

```
1) Polynom,
```
```
2) xA−zmit den Nullstellenz∈Cdes Nennersq,
```
```
3) (x−Az)k+1 mitk≥1, fallsz∈Cmehrfache Nullstelle des Nennersqist.
```
Die Stammfunktion eines Polynoms ist klar, die Stammfunktionen der Partialbr ̈uche
bestimmen wir im n ̈achsten Satz.

Satz 31.8(Stammfunktion von Partialbr ̈uchen). 1) Mehrfache Polstellen: F ̈urk≥
1 gilt wie im Reellen
∫
1
(x−z)k+1

```
dx=−
```
##### 1

```
k
```
##### 1

```
(x−z)k
```
```
+c mitc∈C.
```
```
2) Einfache Polstellen: F ̈urz=a+ib∈Cmita,b∈Rist
∫
1
x−z
dx=
```
##### {

```
ln|x−z|+c fallsb= 0
ln|x−z|+iarctan
```
```
(x−a
b
```
##### )

```
+c fallsb 6 = 0
```
```
mitc∈C.
```
Beweis. Wie im Reellen istdxd(x−^1 z)k=−k(x−z^1 )k+1, da die Ableitungsregeln auch f ̈ur komplexwertige
Funktionen gelten. Daraus folgt die Form der Stammfunktion in1).
Zu 2). F ̈urb= 0 ist alsoz=a∈R, und wir haben bereits in Abschnitt 29.3 nachgerechnet, dass
ln|x−z|eine Stammfunktion vonx−^1 zist (f ̈urz=a∈R).
Fur ̈ b 6 = 0 ist|x−z|^2 = (x−a)^2 +b^2 und
1
x−z=

```
x−z
(x−z)(x−z)=
```
```
x−a
(x−a)^2 +b^2 +i
```
```
b
(x−a)^2 +b^2.
```

Wie sieht eine Stammfunktion aus? Der erste Term ergibt (vergleiche Beispiel 30.1)
∫
x−a
(x−a)^2 +b^2 dx=

```
1
2 ln((x−a)
```
(^2) +b (^2) ) +c 1 = ln|x−z|+c 1 , c 1 ∈R.
Der zweite Term ergibt ∫
b
(x−a)^2 +b^2 dx=
∫ 1
b
1 +
(x−a
b
) 2 dx.
Substituieren wirt=x−ba, so istdt=^1 bdxund weiter
∫
1
(x−a)^2 +b^2 dx=
∫
1
1 +t^2 dt= arctan(t) +c= arctan
(x−a
b
)
+c 2 , c 2 ∈R.
Zusammen ist
∫ 1
x−zdx=
∫ x−a
(x−a)^2 +b^2 dx+i
∫ b
(x−a)^2 +b^2 dx= ln|x−z|+c^1 +iarctan
(x−a
b
)
+ic 2 ,
und mitc=c 1 +ic 2 ∈Chaben wir die Formel nachgerechnet.
Beispiel 31.9. Wir berechnen

##### ∫ 1

```
x^2 − 9 dx. Wegenx
```
(^2) −9 = (x−3)(x+ 3) (binomische
Formel, sonst mitpq-Formel) setzen wir an
1
x^2 − 9

##### =

##### A

```
x− 3
```
##### +

##### B

```
x+ 3
```
##### .

Multiplikation mit dem Nenner ergibt 1 =A(x+ 3) +B(x−3). Durch einsetzen von
x= 3 finden wirA= 1/6 und dannB=− 1 /6. (Alternativ Koeffizientenvergleich und
LGS l ̈osen.) Daher ist
∫
1
x^2 − 9

```
dx=
```
##### 1

##### 6

##### ∫

##### 1

```
x− 3
```
```
dx−
```
##### 1

##### 6

##### ∫

##### 1

```
x+ 3
```
```
dx=
```
##### 1

##### 6

```
ln|x− 3 |−
```
##### 1

##### 6

```
ln|x+3|+c=
```
##### 1

##### 6

```
ln
```
##### ∣

##### ∣∣

##### ∣

```
x− 3
x+ 3
```
##### ∣

##### ∣∣

```
∣+c
```
mitc∈R.

Istf(x) =pq((xx)) mit reellen Polynomenpundq, und istzeine nichtreelle Nullstelle
vonq, alsoz=a+ibmita,b∈Rundb 6 = 0, so ist auchz=a−ibeine Nullstelle von
q. In der reellen PBZ haben wir dann Terme der Form

```
A
x−z
```
##### +

##### B

```
x−z
```
##### .

Wir haben zwei M ̈oglichkeiten eine Stammfunktion zu bestimmen:

```
1) Wir verwenden die Stammfunktion aus Satz 31.8. Mitz=a+ibistz=a−ib, so
dass
∫ (
A
x−z
```
##### +

##### B

```
x−z
```
##### )

```
dx
```
```
=Aln|x−z|+iAarctan
```
##### (

```
x−a
b
```
##### )

```
+Bln|x−z|+iBarctan
```
##### (

```
x−a
−b
```
##### )

```
+c
```
```
= (A+B) ln|x−z|+ (A−iB) arctan
```
##### (

```
x−a
b
```
##### )

```
+c, c∈R,
```

```
wobei wir verwendet haben, dass ln|x−z|= ln|x−z|gilt und dass der Arcustan-
gens eine ungerade Funktion ist. (Hier ist die Integrationskonstante reell, da die
Funktion f ̈ur reellesxauch reelle Werte hat.)
```
```
2) Wir fassen beide Terme zu einem Bruch zusammen. Anschließend haben wir ty-
pischerweise zwei Integral zu berechnen: Eins f ̈uhrt auf einen ln, das andere auf
einen arctan (wie im Beweis von Satz 31.8).
```
Beispiel 31.10.Wir berechnen das unbestimmte Integral von

```
f(x) =
```
```
4 x^2 − 4 x
(x^2 + 1)^2
```
##### =

```
4 x^2 − 4 x
(x−i)^2 (x+i)^2
```
##### =

##### A

```
x+i
```
##### +

##### B

```
(x+i)^2
```
##### +

##### C

```
x−i
```
##### +

##### D

```
(x−i)^2
```
```
=
```
```
i
x+i
```
##### +

```
1 −i
(x+i)^2
```
##### +

```
−i
x−i
```
##### +

```
1 +i
(x−i)^2
```
##### .

(Koeffizientenvergleich fur ̈ A,B,C,D!) Fassen wir die beiden Pole erster Ordnung zu-
sammen kommt
f(x) =

##### 2

```
1 +x^2
```
##### +

```
1 −i
(x+i)^2
```
##### +

```
1 +i
(x−i)^2
```
##### .

Nun erhalten wir
∫
f(x)dx= 2 arctan(x)−
1 −i
x+i

##### −

```
1 +i
x−i
```
```
+c
```
```
= 2 arctan(x)−
(1−i)(x−i) + (1 +i)(x+i)
x^2 + 1
```
```
+c
```
```
= 2 arctan(x)−
2 x− 2
x^2 + 1
+c
```
mitc∈R. Man erh ̈alt das gleiche Ergebnis, wenn man die beiden einfachen Pole separat
integriert: Mitz=i= 0 + 1iist
∫ (
i
x+i

##### +

```
−i
x−i
```
##### )

```
dx=i
```
##### (

```
ln|x+i|+iarctan(−x)
```
##### )

```
−i
```
##### (

```
ln|x−i|+iarctan(x)
```
##### )

```
+c
```
```
=−i^2 arctan(x)−i^2 arctan(x) +c= 2 arctan(x) +c.
```


Vorlesung 32

## 32 Die Determinante

Die Determinante dient der Volumenberechnung. Sie findet Anwendungen

- bei der mehrdimensionalen Integration (
    ”
       Analysis II fur Ingenieurwissenschaften“) ̈
- bei der Herleitung und Berechnung von Eigenwerten (Vorlesung 33),
- um die Invertierbarkeit einer Matrix zu charakterisieren (jetzt).

### 32.1 Determinante und Volumenberechnung

F ̈urA=

##### [

```
a 11
```
##### ]

definiert man det(A):=a 11.
Mit Blick auf die Zahlengerade gibt|det(A)|also den Abstand zum Ursprung an
(= L ̈ange des Vektorsa 11 ∈R= eindimensionales Volumen), und das Vorzeichen der
Determinante gibt an, ob der Punkt rechts (+) oder links (−) vom Ursprung liegt.
Die Determinante einer 2×2-Matrix

##### A=

##### [

```
a 1 , 1 a 1 , 2
a 2 , 1 a 2 , 2
```
##### ]

ist
det(A) =a 1 , 1 a 2 , 2 −a 1 , 2 a 2 , 1.
Geometrische Deutung inR^2 (der Ebene):|det(A)|ist der Fl ̈acheninhalt des von den
Vektoren

```
~a 1 =
```
##### [

```
a 1 , 1
a 2 , 1
```
##### ]

```
, ~a 2 =
```
##### [

```
a 1 , 2
a 2 , 2
```
##### ]

aufgespannten Parallelogramms. Um dies nachzurechnen, betrachten wir die Fl ̈ache des
großen Rechtecks in der folgenden Skizze:

```
(a 1 , 1 +a 1 , 2 )(a 2 , 1 +a 2 , 2 ) =F+ 2F 1 + 2F 2 + 2F 3 ,
```
wobei

```
F 1 =a 1 , 2 a 2 , 1 , F 2 =
```
##### 1

##### 2

```
a 1 , 2 a 2 , 2 , F 3 =
```
##### 1

##### 2

```
a 1 , 1 a 2 , 1.
```

```
a 1 , 1
```
```
a 2 , 1
```
```
a 2 , 2
```
```
a 1 , 2
```
```
a 1 , 1 +a 1 , 2
```
```
a 2 , 1 +a 2 , 2
```
```
~a 1
```
```
~a 2
```
```
F
```
##### F 1

##### F 1

##### F 2

##### F 2

##### F 3

##### F 3

Daher folgt
F= (a 1 , 1 +a 1 , 2 )(a 2 , 1 +a 2 , 2 )− 2 F 1 − 2 F 2 − 2 F 3
=a 1 , 1 a 2 , 1 +a 1 , 1 a 2 , 2 +a 1 , 2 a 2 , 1 +a 1 , 2 a 2 , 2 − 2 a 1 , 2 a 2 , 1 −a 1 , 2 a 2 , 2 −a 1 , 1 a 2 , 1
=a 1 , 1 a 2 , 2 −a 2 , 1 a 2 , 1 = det(A).
Vertauscht man~a 1 und~a 2 in der Skizze, so bleibt der Fl ̈acheninhalt des Parallelogramms
gleich, eine ̈ahnliche Rechnung liefert aberF =a 1 , 2 a 2 , 1 −a 2 , 2 a 1 , 1 =−det(A), so dass
F=|det(A)|.
Wir haben jetzt zwei Dinge gelernt:|det(A)|=Fist der Fl ̈acheninhalt des von~a 1
und~a 2 aufgespannten Parallelogramms, und das Vorzeichen der Determinante gibt an,
wie die Vektoren~a 1 und~a 2 zueinander liegen. Ist det(A)>0, so ist die Drehung von
~a 1 zu~a 2 ̈uber den kleineren Winkel entgegen dem Uhrzeigersinn, ist det(A)<0 ist die
Drehung im Uhrzeigersinn.

```
~a 1
```
```
~a 2
```
```
det(A)> 0
~a 2
```
```
~a 1
```
```
det(A)< 0
```
```
Zum Beispiel sind
```
```
det
```
##### ([

##### 1 0

##### 0 1

##### ])

```
= 1· 1 − 0 ·0 = 1> 0 , det
```
##### ([

##### 0 1

##### 1 0

##### ])

##### = 0· 0 − 1 ·1 =− 1 < 0.

```
Drei Vektoren~a 1 ,~a 2 ,~a 3 ∈R^3 erzeugen ein Parallelotop (= Parallelepiped = Spat),
dessen Volumen mit dem Spatprodukt berechnet werden kann:
V =|~a 1 ·(~a 2 ×~a 3 )|,
wobei·das Standardskalarprodukt inR^3 bezeichnet und×das Vektorprodukt,
```
```
~a 2 ×~a 3 =
```
##### 

##### 

```
a 2 , 2 a 3 , 3 −a 3 , 2 a 2 , 3
a 3 , 2 a 1 , 3 −a 1 , 2 a 3 , 3
a 1 , 2 a 2 , 3 −a 2 , 2 a 1 , 3
```
##### 

##### 


Daher wird die Determinante vonA=

##### [

```
ai,j
```
##### ]

```
∈K^3 ,^3 formal als Spatprodukt definiert:
```
```
det(A) =a 1 , 1 (a 2 , 2 a 3 , 3 −a 3 , 2 a 2 , 3 ) +a 2 , 1 (a 3 , 2 a 1 , 3 −a 1 , 2 a 3 , 3 ) +a 3 , 1 (a 1 , 2 a 2 , 3 −a 2 , 2 a 1 , 3 ).
```
Die Ausdr ̈ucke in Klammern sind die Determinanten von 2×2-Matrizen:

```
det(A) =a 1 , 1 det
```
##### ([

```
a 2 , 2 a 2 , 3
a 3 , 2 a 3 , 3
```
##### ])

```
−a 2 , 1 det
```
##### ([

```
a 1 , 2 a 1 , 3
a 3 , 2 a 3 , 3
```
##### ])

```
+a 3 , 1 det
```
##### ([

```
a 1 , 2 a 1 , 3
a 2 , 2 a 2 , 3
```
##### ])

und die 2×2-Matrizen entstehen durch streichen einer Zeile und einer Spalte vonA.

Definition 32.1 (Streichungsmatrix). SeiA∈Kn,nmitn >1. Durch streichen der
i-ten Zeile undj-ten Spalte entsteht dieStreichungsmatrix

```
Ai,j=
```
##### 

##### 

##### 

##### 

##### 

##### 

```
a 1 , 1 ... a 1 ,j ... a 1 ,n
..
.
```
##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

```
ai, 1 ... ai,j ... ai,n
..
.
```
##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

```
an, 1 ... an,j ... an,n
```
##### 

##### 

##### 

##### 

##### 

##### 

```
∈Kn−^1 ,n−^1.
```
```
Mit den Streichungsmatrizen ist
```
```
det(A) =a 1 , 1 det(A 1 , 1 )−a 2 , 1 det(A 2 , 1 ), A∈K^2 ,^2 ,
det(A) =a 1 , 1 det(A 1 , 1 )−a 2 , 1 det(A 2 , 1 ) +a 3 , 1 det(A 3 , 1 ), A∈K^3 ,^3.
```
Die Determinante vonA l ̈asst sich also berechnen, indem man die erste Spalte ent-
langgeht, den Eintragai, 1 mit det(Ai, 1 ) multipliziert und aufsummiert (mit Vorzeichen
(−1)i+1). Auf diese Weise l ̈asst sich die Determinante rekursiv f ̈ur beliebige quadratische
Matrizen definieren.

Definition 32.2(Determinante).DieDeterminantevonA=

##### [

```
ai,j
```
##### ]

```
∈Kn,nist
```
```
det(A):=a 1 , 1 , fallsn= 1,
```
```
det(A):=
```
```
∑n
```
```
i=1
```
```
(−1)i+1ai, 1 det(Ai, 1 ), fallsn > 1.
```
Beispiel 32.3.Fur ̈

##### A=

##### 

##### 

##### 

##### 1 2 0 1

##### 4 −2 1 0

##### 0 5 3 − 1

##### 2 4 1 3

##### 

##### 

##### 

haben wir die Streichungsmatrizen

##### A 1 , 1 =

##### 

##### 

##### −2 1 0

##### 5 3 − 1

##### 4 1 3

##### 

##### ,A 2 , 1 =

##### 

##### 

##### 2 0 1

##### 5 3 − 1

##### 4 1 3

##### 

##### ,A 3 , 1 =

##### 

##### 

##### 2 0 1

##### −2 1 0

##### 4 1 3

##### 

##### ,A 4 , 1 =

##### 

##### 

##### 2 0 1

##### −2 1 0

##### 5 3 − 1

##### 

##### .


Damit ist

```
det(A) = 1 det(A 1 , 1 )−4 det(A 2 , 1 ) + 0 det(A 3 , 1 )−2 det(A 4 , 1 )
=
```
##### (

##### −2(3· 3 − 1 ·(−1))−5(1· 3 − 1 ·0) + 4(1·(−1)− 3 ·0))

##### )

##### − 4

##### (

##### 2(3· 3 − 1 ·(−1))−5(0· 3 − 1 ·1) + 4(0·(−1)− 3 ·1)

##### )

##### − 2

##### (

##### 2(1·(−1)− 3 ·0)−(−2)(0·(−1)− 3 ·1) + 5(0· 0 − 1 ·1)

##### )

##### =− 2 · 10 − 5 ·3 + 4·(−1)− 4

##### (

##### 2 ·10 + 5 + 4·(−3)

##### )

##### − 2

##### (

##### − 2 − 6 − 5

##### )

##### =− 65.

### 32.2 Berechnung von Determinanten

Fur Dreiecksmatrizen l ̈ ̈asst sich die Determinante ganz einfach berechnen.

Satz 32.4. F ̈ur eine obere Dreicksmatrix

##### A=

##### 

##### 

##### 

##### 

```
a 1 , 1 a 1 , 2 ... a 1 ,n
0 a 2 , 2 ... a 2 ,n
..
.
```
##### ... ... ..

##### .

```
0 ... 0 an,n
```
##### 

##### 

##### 

##### 

istdet(A) =a 1 , 1 a 2 , 2 ...an,ndas Produkt der Diagonaleintr ̈age.

Beweis. Mit der Definition ist

```
det(A) =a 1 , 1 det(A 1 , 1 ) + 0 det(A 2 , 1 ) +...+ 0 det(An, 1 )
```
```
=a 1 , 1 det
```
##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

```
a 2 , 2 a 2 , 3 ... a 2 ,n
0 a 3 , 3 ... a 3 ,n
..
.
```
##### ... ... ..

##### .

```
0 ... 0 an,n
```
##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

```
=a 1 , 1 a 2 , 2 det
```
##### 

##### 

##### 

##### 

##### 

##### 

```
a 3 , 3 ... a 3 ,n
0
```
##### ... ..

##### .

```
0 0 an,n
```
##### 

##### 

##### 

##### 

##### 

##### 

```
=...=a 1 , 1 a 2 , 2 ...an,n.
```
Fur die Determinante einer allgemeinen ̈ n×n-Matrix m ̈ussen hingegennDetermi-
nanten von (n−1)×(n−1)-Matrizen berechnet werden, was insgesamt aufn! Summan-
den mit Produkten vonnZahlen fuhrt. Da die Fakult ̈ ̈at schnell w ̈achst, ben ̈otigen wir
Rechenregeln, mit denen wir Determinanten besser berechnen k ̈onnen.

Satz 32.5(Rechenregeln f ̈ur die Determinante).SeiA∈Kn,nund sei~ajdiej-te Spalte
vonA.


```
1) Die Determinante ist linear in jeder Spalte vonA:
```
```
det
```
##### ([

```
~a 1 ... λ~aj ... ~an
```
##### ])

```
=λdet
```
##### ([

```
~a 1 ... ~aj ... ~an
```
##### ])

```
und
```
```
det
```
##### ([

```
~a 1 ... ~aj+~ ̃aj ... ~an
```
##### ])

```
= det
```
##### ([

```
~a 1 ... ~aj ... ~an
```
##### ])

```
+ det
```
##### ([

```
~a 1 ... ~ ̃aj ... ~an
```
##### ])

##### .

```
2) Die Determinante ist antisymmetrisch: Vertauscht man zwei Spalten, so ̈andert
sich das Vorzeichen: F ̈urk 6 =jist
```
```
det
```
##### ([

```
~a 1 ... ~aj ... ~ak ... ~an
```
##### ])

```
=−det
```
##### ([

```
~a 1 ... ~ak ... ~aj ... ~an
```
##### ])

##### .

```
3) HatAzwei gleiche Spalten, so istdet(A) = 0.
```
```
4) Addition einer Spalte zu einer anderen Spalte ̈andert die Determinante nicht: F ̈ur
k 6 =jist
```
```
det
```
##### ([

```
~a 1 ... (~aj+λ~ak) ... ~an
```
##### ])

```
= det
```
##### ([

```
~a 1 ... ~aj ... ~an
```
##### ])

##### .

```
5) det(A) = det(AT).
```
Wegen 5) gelten alle Aussagen auch f ̈ur die Zeilen der Matrix.

Beweis. Die ersten beiden Aussagen lassen sich zum Beispiel per Induktion ( ̈ubern) zeigen, wir f ̈uhren
das nicht aus. Auch5)rechnen wir nicht nach.
3)folgt unmittelbar aus2): Vertauschen wir die beiden gleichen Spalten, so bleibt die Matrix gleich,
aber das Vorzeichen der Determinante ̈andert sich: det(A) =−det(A), also ist det(A) = 0.
Fur ̈ 4)rechnen wir mit der Linearit ̈at in Spaltej:
det
([
~a 1 ... (~aj+λ~ak) ... ~ak ... ~an
])

```
= det
([
~a 1 ... ~aj ... ~ak ... ~an
])
+λdet
([
~a 1 ... ~ak ... ~ak ... ~an
])
,
```
und die letzte Determinante ist Null, da die Spaltenjundkgleich sind.

Beachten Sie: Die Determinante ist linear in jeder Spalte und Zeile, f ̈urn >1 ist
det :Kn,n→Kaber nicht linear”in der Matrix“, z.B. ist

```
det(2I 2 ) = 2·2 = 4 6 = 2 = 2 det(I 2 ).
```
Mit den elementaren Zeilen- und Spaltenoperationen k ̈onnen wir det(A) nun leichter
berechnen, indem wirAmit dem Gaußalgorithmus auf Dreiecksgestalt bringen, von der
wir die Determinante dann ablesen k ̈onnen.


Beispiel 32.6.Es ist

```
det
```
##### 

##### 

##### 

##### 

##### 

##### 

##### 1 2 1 − 1

##### 1 5 1 − 1

##### 2 5 4 − 1

##### 1 2 1 0

##### 

##### 

##### 

##### 

##### 

```
= det
```
##### 

##### 

##### 

##### 

##### 

##### 

##### 1 2 1 − 1

##### 0 3 0 0

##### 0 1 2 1

##### 0 0 0 1

##### 

##### 

##### 

##### 

##### 

```
= 3 det
```
##### 

##### 

##### 

##### 

##### 

##### 

##### 1 2 1 − 1

##### 0 1 0 0

##### 0 1 2 1

##### 0 0 0 1

##### 

##### 

##### 

##### 

##### 

##### 

```
= 3 det
```
##### 

##### 

##### 

##### 

##### 

##### 

##### 1 2 1 − 1

##### 0 1 0 0

##### 0 0 2 1

##### 0 0 0 1

##### 

##### 

##### 

##### 

##### 

##### = 6.

Die sogennanteLaplace-Entwicklung ist ein weiteres n ̈utzliches Hilfsmittel bei der
Berechnung der Determinante. Diese erlaubt die”Entwicklung nach der ersten Spalte“
auf andere Spalten und Zeilen zu ̈ubertragen.

Satz 32.7 (Laplace-Entwicklung nach Zeilen und Spalten). Fur ̈ A ∈ Kn,n gilt die
Laplace-Entwicklung nach derj-ten Spalte:

```
det(A) =
```
```
∑n
```
```
i=1
```
```
(−1)i+jai,jdet(Ai,j)
```
sowie die Laplace-Entwiklung nach deri-ten Zeile:

```
det(A) =
```
```
∑n
```
```
j=1
```
```
(−1)i+jai,jdet(Ai,j)
```
Beweis. Die MatrixB=

##### [

```
aj a 1 ... aj− 1 aj+1 ... an
```
##### ]

entsteht ausAdurch tau-
schen vonj−1 Spalten, also gilt det(A) = (−1)j−^1 det(B). Berechnet man det(B), so
erh ̈alt man die Formel f ̈ur die Laplace-Entwicklung nach derj-ten Spalte. Die Behaup-
tung f ̈ur die Zeilen folgt aus det(A) = det(AT).

```
Das Vorzeichenmuster bei der Entwicklung bildet ein”Schachbrett“:





```
##### + − + − + ...

##### − + − + − ...

##### + − + − + ...

##### − + − + − ...

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ..

##### .

##### ...

##### 

##### 

##### 

##### 

##### 

Beispiel 32.8.Mit einer Laplace-Entwicklung nach der dritten Spalte ist

```
det
```
##### 

##### 

##### 

##### 

##### 15 2 0

##### 12 15 4

##### 1 0 0

##### 

##### 

##### 

```
= (−1)·4 det
```
##### ([

##### 15 2

##### 1 0

##### ])

##### =−4(15· 0 − 1 ·2) = 8.


Berechnung der Determinante: Besonders hilfreich ist die Kombination von Zeilen-
und Spaltenoperationen mit der Laplace-Entwicklung. Das typische Vorgehen ist:

- Erzeuge m ̈oglichst viele Nullen in einer Zeile/Spalte durch elementare Zeilen- und
    Spaltenoperationen,
- Entwickle nach einer Zeile/Spalte mit m ̈oglichst vielen Nullen.

Wiederhole diese Schritte so lange, bis die Matrizen 2×2 sind, und die Determinanten
direkt berechnet werden k ̈onnen.

Beispiel 32.9.Es ist (2. Zeile−1. Zeile)

```
det(A) = det
```
##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 1 2 1 − 1

##### 1 3 1 − 1

##### 2 5 4 − 1

##### 1 2 1 0

##### 

##### 

##### 

##### 

##### 

##### 

##### 

```
= det
```
##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 1 2 1 − 1

##### 0 1 0 0

##### 2 5 4 − 1

##### 1 2 1 0

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

Laplace-Entwicklung nach der zweiten Zeile ergibt

```
det(A) = det
```
##### 

##### 

##### 

##### 

##### 1 1 − 1

##### 2 4 − 1

##### 1 1 0

##### 

##### 

##### 

```
= det
```
##### 

##### 

##### 

##### 

##### 1 0 − 1

##### 2 2 − 1

##### 1 0 0

##### 

##### 

##### 

```
= +1 det
```
##### ([

##### 0 − 1

##### 2 − 1

##### ])

##### = 2,

Dabei haben wir beim zweiten Gleichheitszeichen 2. Spalte−1. Spalte gerechnet, und
beim dritten Gleichheitszeichen nach der 3. Zeile entwickelt.

### 32.3 Der Determinantenmultiplikationssatz

Der folgende Determinantenmultiplikationssatz hat zahlreiche n ̈utzliche Folgerungen.

Satz 32.10 (Determinantenmultiplikationssatz). F ̈ur quadratische Matrizen A,B ∈
Kn,ngilt
det(AB) = det(A) det(B).

Bemerkung 32.11.Achtung: Es gibt keinen Determinantenadditionssatz, einfach weil
dieser falsch w ̈are! Im Allgemeinen ist det(A+B) 6 = det(A) + det(B). Rechnen Sie zum
Beispiel beide Seiten f ̈urA= [1 00 0] undB= [0 00 1] aus.

```
Aus dem Determinantenmultiplikationssatz ergeben sich viele n ̈utzliche Folgerungen.
```
Satz 32.12.F ̈ur quadratischeA,B∈Kn,ngilt

```
1) det(AB) = det(BA), selbst wennAB 6 =BA.
```
```
2) det(Ak) = det(A)kf ̈urk∈N.
```
```
3) det(A−^1 ) = (det(A))−^1 , fallsAinvertierbar ist.
```

```
4)det(C−^1 AC) = det(A)f ̈ur alle invertierbarenC∈Kn,n.
```
```
5) F ̈ur eine Blockdreiecksmatrix mit quadratischenBundDgilt
```
```
det
```
##### ([

##### B C

##### 0 D

##### ])

```
= det(B) det(D).
```
```
Die MatrixCmuss dabei nicht quadratisch sein.
```
Beweis. 1) Denn det(AB) = det(A) det(B) = det(B) det(A) = det(BA).
2) det(Ak) = det(A) det(Ak−^1 ) =...= det(A)k.
3) 1 = det(In) = det(AA−^1 ) = det(A) det(A−^1 ).
4) det(C−^1 AC) = det(C−^1 (AC)) = det((AC)C−^1 ) = det(A).
5) Man rechnet nach, dass (mitB∈Kn,n,D∈Km,m)
[
B C
0 D

##### ]

##### =

##### [

```
In 0
0 D
```
##### ][

##### B C

```
0 Im
```
##### ]

```
Dann liefert der Determinantenmultiplikationssatz
```
```
det
```
##### ([

##### B C

##### 0 D

##### ])

```
= det
```
##### ([

##### I 0

##### 0 D

##### ])

```
det
```
##### ([

##### B C

##### 0 I

##### ])

```
= det(D) det(B).
```
```
F ̈ur das letzte Gleichheitszeichen:nMal Laplace-Entwicklung nach der ersten Spal-
te fur det ̈
```
```
([In 0
0 D
```
##### ])

```
= det(D), undmMal Laplace-Entwicklung nach der letzten
Zeile f ̈ur die zweite Determinante.
```
### 32.4 Charakterisierung invertierbarer Matrizen

Mit Hilfe der Determinante erhalten wie eine weitere Charakterisierung, wann eine qua-
dratische Matrix invertierbar ist.

Satz 32.13.SeiA∈Kn,n. Dann sind ̈aquivalent:
1)Aist invertierbar
2)det(A) 6 = 0
3)Rang(A) =n
4)A~x=~ 0 ist eindeutig l ̈osbar (durch~x=~ 0 )
5) die Spalten vonAsind linear unabh ̈angig
6) die Zeilen vonAsind linear unabh ̈angig

Beweis. Wir wissen bereits (aus dem Gaußalgorithmus), dass

```
Ainvertierbar⇔Rang(A) =n⇔A~x=~0 ist eindeutig l ̈osbar.
```
Letzteres heißt aber genau, dass die Spalten vonAnur trivial zu Null linear kombiniert
werden k ̈onnen, d.h. dass die Spalten vonAlinear unabh ̈angig sind. Wegen Rang(A) =
Rang(AT) ist dies auch ̈aquivalent dazu, dass die Zeilen vonAlinear unabh ̈angig sind.


Bringen wirAin ZeilenstufenformC, so ̈andert sich bei Zeilentausch das Vorzeichen
der Determinante, beim Addieren des Vielfaches einer Zeile zu einer anderen bleibt
die Determinante gleich, zudem lassen sich Konstanten c 6 = 0 aus der Determinante
herausziehen, d.h. det(A) =αdet(C) f ̈ur einα 6 = 0.
Daher gilt also det(A) 6 = 0 genau dann, wenn det(C) 6 = 0. DaCeine Dreicksmatrix
ist, ist det(C) das Produkt der Diagonalelemente, und das ist 6 = 0 wenn alle Diagonal-
element 6 = 0 sind, also wenn Rang(A) =n.

Durch Negation aller Aussagen erhalten wir die Charakterisierung f ̈ur nicht inver-
tierbare Matrizen.

Satz 32.14.F ̈urA∈Kn,nsind ̈aquivalent:
1) Aist nicht invertierbar
2) det(A) = 0
3) Rang(A)< n
4) es existiert~x∈Knmit~x 6 = 0undA~x=~ 0
5) die Spalten vonAsind linear abh ̈angig
6) die Zeilen vonAsind linear abh ̈angig



Vorlesung 33

## 33 Eigenwerte und Eigenvektoren

Wir lernen Eigenwerte und Eigenvektoren von Matrizen kennen. Diese sind wichtige
Kenngr ̈oßen einer Matrix.
Zur Motivation betrachten wir das Bild des Quadrats mit den Ecken~0 = [^00 ],~e 1 = [^10 ],
~e 1 +~e 2 = [^11 ],~e 2 = [^01 ] bei Multiplikation mit verschiedenen MatrizenA∈R^2 ,^2.

```
1) F ̈urA= [2 00 1] ist
```
##### [^00 ] [^10 ]

##### [^01 ] [^11 ]

##### [^00 ] A[^10 ]

##### A[^01 ] A[^11 ]

```
Geometrisch bewirkt die Multiplikation mitAeine Streckung um den Faktor 2 in
x-Richtung, daA[^10 ] = 2 [^10 ], iny-Richtung bleibt alles wie es ist:A[^01 ] = [^01 ].
```
```
2) F ̈urA=^12 [3 11 3] ist
```
##### [^00 ] [^10 ]

##### [^01 ] [^11 ]

##### [^00 ]

##### A[^10 ]

##### A[^11 ]

##### A[^01 ]

```
Hier wird die Richtung des Vektors [^11 ] um 2 gestreckt,A[^11 ] = 2 [^11 ], w ̈ahrend die
Richtung des Vektors
```
##### [ 1

```
− 1
```
##### ]

```
um 1 gestreckt wird (also gleich bleibt):A
```
##### [ 1

```
− 1
```
##### ]

##### =

##### [ 1

```
− 1
```
##### ]

##### .

```
3) F ̈urA= [1 11 1] ist
```

##### [^00 ] [^10 ]

##### [^01 ] [^11 ]

##### [^00 ]

##### A[^10 ]

##### A[^11 ]

##### A[^01 ]

```
Das Quadrat wurde plattgedruckt! Multiplikation mit ̈ Af ̈uhrt zu einer Streckung
mit 2 in Richtung des Vektors [^11 ],A[^11 ] = 2 [^11 ], und zu einer Streckung mit 0 in
Richtung des Vektors
```
##### [ 1

```
− 1
```
##### ]

```
∈Kern(A).
```
```
4) F ̈urA=
```
##### [1 1

```
−1 1
```
##### ]

```
ist
```
##### [^00 ] [^10 ]

##### [^01 ] [^11 ]

##### [^00 ]

##### A[^10 ]

##### A[^11 ]

##### A[^01 ]

```
Multiplikation mitAf ̈uhrt eine Drehstreckung des Quadrats aus. Hier ist nicht
ersichtlich, obAVektoren in eine Richtung streckt.
```
### 33.1 Definition von Eigenwerten und Eigenvektoren

Definition 33.1(Eigenwerte und Eigenvektoren).SeiA∈Kn,neine quadratische Ma-
trix. Gilt
A~v=λ~v mitλ∈Kund~v∈Kn,~v 6 =~ 0 , (33.1)

so heißt

```
1)λeinEigenwertvonAmit zugeh ̈origem Eigenvektor~v,
```
```
2)~veinEigenvektorvonAzum Eigenwertλ.
```
Die Gleichung (33.1) wird auchEigenwertgleichunggenannt.

Geometrisch bewirkt die Multiplikation eines Eigenvektors mitAeine Streckung um
den Faktorλ.

Bemerkung 33.2.Beachten Sie:
1) Eigenwerte k ̈onnen 0 sein.
2) Eigenvektoren k ̈onnen nie~0 sein. (F ̈ur~v=~0 gilt n ̈amlichA~0 =~0 =λ~0 f ̈ur jedes
λ∈K, was nicht spannend ist.)


Beispiel 33.3. 1) Mit der MatrixA=

##### [

##### 3 − 4

##### 2 − 3

##### ]

```
ist
```
```
[
3 − 4
2 − 3
```
##### ][

##### 1

##### 1

##### ]

##### =

##### [

##### − 1

##### − 1

##### ]

##### = (−1)

##### [

##### 1

##### 1

##### ]

##### ,

##### [

##### 3 − 4

##### 2 − 3

##### ][

##### 2

##### 1

##### ]

##### =

##### [

##### 2

##### 1

##### ]

##### = 1

##### [

##### 2

##### 1

##### ]

##### ,

```
also hat die MatrixAdie beiden Eigenwerte 1 und−1. Der Vektor [^11 ] 6 = 0 ist
ein Eigenvektor vonAzum Eigenwert−1 und [^21 ] ist ein Eigenvektor vonAzum
Eigenwert 1.
2) F ̈urA=
```
##### [

##### 1 1

##### 1 1

##### ]

```
gilt
```
##### A

##### [

##### 1

##### 1

##### ]

##### = 2

##### [

##### 1

##### 1

##### ]

##### , A

##### [

##### 1

##### − 1

##### ]

##### =

##### [

##### 0

##### 0

##### ]

##### = 0

##### [

##### 1

##### − 1

##### ]

##### ,

```
daher hatAden Eigenwert 0 mit Eigenvektor
```
##### [ 1

```
− 1
```
##### ]

```
und den Eigenwert 2 mit
Eigenvektor [^11 ], vergleiche das motivierende Beispiel.
```
Bemerkung 33.4 (Reelle und komplexe Matrizen). Streng nach Definition sind die
Eigenwerte also aus dem gleichen Grundk ̈orper (RoderC) wie die Eintr ̈age der Matrix.
Insbesondere hat eine reelle MatrixA∈Rn,nnur reelle Eigenwerte. Es stellt sich heraus
(siehe Abschnitt 33.2 unten), dass die Eigenwerte Nullstellen eins Polynoms sind. Dieses
hat immer komplexe Nullstellen, die nicht notwendigerweise reell sein m ̈ussen. Um l ̈astige
Fallunterscheidungen zu vermeiden, fasst man ganz einfach die Matrix als komplexe
Matrix auf:A∈Rn,n⊆Cn,n, da ja reelle Zahlen auch komplexe Zahlen sind. Zu reellen
Eigenwerte gibt es dann reelle Eigenvektoren, zu nichtreellen Eigenwerten findet man
nichtreelle Eigenvektoren.
Ein Beispiel ist die MatrixA=

##### [0 1

```
−1 0
```
##### ]

```
∈R^2 ,^2 ⊆C^2 ,^2 in Beispiel 33.9 unten.
```
Definition 33.5(Eigenraum und geometrische Vielfachheit). SeiA∈Kn,nmit einem
Eigenwertλ∈K.
1) DerEigenraumvonAzum Eigenwertλist

```
Vλ=Vλ(A) ={~v∈Kn|A~v=λ~v}.
```
```
2) Diegeometrische Vielfachheitdes Eigenwertsλist die Dimension des Eigenraums:
```
```
g(λ) =g(λ,A) = dim(Vλ(A)).
```
Bemerkung 33.6. 1) Es istA~v=λ~vgenau dann, wenn (A−λIn)~v=~0. Daher ist

```
Vλ={~v∈Kn|A~v=λ~v}= Kern(A−λI).
```
```
Insbesondere istVλein Teilraum desKn. Das kann man auch mit dem Teilraum-
kriterium (Satz 10.4) nachrechnen.
2) Die Elemente~v 6 =~0 des Eigenraums sind genau die Eigenvektoren:
```
```
Vλ={Eigenvektoren zum Eigenwertλ}∪{~ 0 }.
```

```
3) Die geometrische Vielfachheit des Eigenwertsλist die Dimension des Eigenraums,
also die maximale Anzahl linear unabh ̈angiger Eigenvektoren zuλ.
```
Beispiel 33.7.Sei

```
A=
```
##### 

##### 

##### 1 2 0

##### 0 1 0

##### 0 0 1

##### 

##### ∈R^3 ,^3 ,

dann istA

##### [ 1

```
0
0
```
##### ]

##### =

##### [ 1

```
0
0
```
##### ]

```
, also ist 1 ein Eigenwert vonA. Wir bestimmen den Eigenraum:
```
```
V 1 = Kern(A− 1 I 3 ) = Kern
```
##### 

##### 

##### 

##### 

##### 0 2 0

##### 0 0 0

##### 0 0 0

##### 

##### 

##### 

##### =

##### 

##### 

##### 

##### 

##### 

```
a
0
b
```
##### 

##### 

##### ∣

##### ∣∣

##### ∣∣

##### ∣

```
a,b∈R
```
##### 

##### 

##### 

Die geometrische Vielfachheit istg(1,A) = dim(V 1 (A)) = 2.

### 33.2 Berechnung von Eigenwerten und Eigenvektoren

Die Eigenvektoren zu einem gegebene Eigenwert finden wir durch l ̈osen eines linearen
Gleichungssystems. Aber wie findet man die Eigenwerte? Wegen

```
A~v=λ~vmit~v 6 =~ 0 ⇔(A−λI)~v=~0 mit~v 6 =~ 0
⇔A−λIist nicht invertierbar
⇔det(A−λI) = 0
```
findet man die Eigenwerte vonAals Nullstellen des Polynoms

```
pA(z) = det(A−zI).
```
Man berechnet die Determinante und sortiert nach Potenzen vonz. Das ergibt ein Po-
lynom der Gestalt

```
pA(z) = (−z)n+an− 1 zn−^1 +...+a 1 z+a 0
```
vom Gradn. Nach dem Fundamentalsatz der Algebra hatpAimmernkomplexe Null-
stellen (mit Vielfachheiten gez ̈ahlt). Die Nullstellen k ̈onnen reell oder komplex sein, und
es kann vorkommen, dass es keine reellen Nullstellen gibt.

Definition 33.8(Charakteristisches Polynom, algebraische Vielfachheit).SeiA∈Kn,n.
1) Dascharakteristische PolynomvonAist

```
pA(z):= det(A−zIn).
```
```
2) Diealgebraische Vielfachheitdes EigenwertsλvonAist die Vielfachheit der Null-
stelleλim charakteristischen Polynom. Bezeichnung:a(λ) =a(λ,A).
```

Sind λ 1 ,...,λr die verschiedenen (komplexen) Nullstellen vonpA, so dasspA die
komplexe Zerlegung

```
pA(z) = (−1)n(z−λ 1 )a(λ^1 )(z−λ 2 )a(λ^2 )...(z−λr)a(λr)
```
hat, so gilt:
1)Ahat die Eigenwerteλ 1 ,...,λr(keine weiteren).
2) Die Summe der algebraischen Vielfachheiten istn:a(λ 1 ) +...+a(λr) =n.
3)Ahat h ̈ochstensnverschiedene Eigenwerte (fallsr=n).

Beispiel 33.9. 1) Das charakteristische Polynom vonA=

##### [0 1

```
−1 0
```
##### ]

```
∈R^2 ,^2 ⊆C^2 ,^2 ist
```
```
pA(z) = det(A−zI 2 ) = det
```
##### ([

```
−z 1
− 1 −z
```
##### ])

```
=z^2 + 1 = (z−i)(z+i),
```
```
also hatAdie Eigenwerteiund−i, beide mit algebraischer Vielfachheit 1. Be-
trachtet manAals reelle Matrix, so hatAkeine (reellen) Eigenwerte.
2) Das charakteristische Polynom von
```
##### A=

##### 

##### 

##### 

##### 2 1 −1 5

##### 1 2 7 2

##### 0 0 1 1

##### 0 0 −1 3

##### 

##### 

##### 

```
berechnen wir mit Hilfe von Satz 32.12 als
```
```
pA(z) = det(A−zI) = det
```
##### 

##### 

##### 

##### 

##### 

##### 

```
2 −z 1 − 1 5
1 2 −z 7 2
0 0 1 −z 1
0 0 − 1 3 −z
```
##### 

##### 

##### 

##### 

##### 

##### 

```
= det
```
##### ([

```
2 −z 1
1 2 −z
```
##### ])

```
det
```
##### ([

```
1 −z 1
− 1 3 −z
```
##### ])

```
= ((2−z)^2 −1)((1−z)(3−z) + 1) = (z^2 − 4 z+ 3)(z^2 − 4 z+ 4).
```
```
Multiplizieren sie hier auf keinen Fall aus, wenn sie die Nullstellen suchen.
Zerlegen Sie besser jeden der beiden Faktoren einzeln:
```
```
pA(z) = (z−1)(z−3)(z−2)^2.
```
```
Also hatAdie Eigenwerte 1, 2 und 3 mit algebraischen Vielfachheitena(1) = 1,
a(2) = 2 unda(3) = 1.
```
Satz 33.10.IstA∈Kn,neine obere oder untere Dreiecksmatrix, so sind die Eigenwerte
vonAgenau die Diagonaleintr ̈age.


Beweis. Mit Satz 32.4 berechnen wir das charakteristische Polynom der oberen Drei-
ecksmatrixA:

```
pA(z) = det
```
##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

```
a 1 , 1 −z a 1 , 2 ... a 1 ,n
0 a 2 , 2 −z
```
##### ... ..

##### .

##### ..

##### .

##### ... ...

```
an− 1 ,n
0 ... 0 an,n−z
```
##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

```
= (a 1 , 1 −z)(a 2 , 2 −z)...(an,n−z).
```
Die Eigenwerte vonAsind die Nullstellen vonpA, alsoa 1 , 1 ,...,an,n. F ̈ur untere Drei-
ecksmatrizen geht es genauso.

Berechnung von Eigenwerten und Eigenvektoren. Ist A ∈ Kn,ngegeben, so
berechnen wir die Eigenwerte und Eigenvektoren wie folgt:

```
1) Berechne das charakteristische PolynompA(z) = det(A−zIn).
```
```
2) Berechne die Nullstellen vonpA, dies sind die Eigenwerte vonA.
```
```
3) Berechne f ̈ur jeden Eigenwert die L ̈osung des homogenen LGS (A−λIn)~v=~0. Die
L ̈osungen~v 6 =~0 sind die Eigenvektoren zum Eigenwertλ.
```
Beispiel 33.11. 1) F ̈urA= [1 33 1] ist

```
pA(z) = det(A−zI 2 ) = det
```
##### ([

```
1 −z 3
3 1 −z
```
##### ])

```
= (1−z)^2 − 9
```
```
= 1− 2 z+z^2 −9 =z^2 − 2 z−8 = (z−4)(z+ 2),
also hatAdie Eigenwerte−2 und 4 je mit algebraischer Vielfachheit 1. Fur die ̈
geometrischen Vielfachheiten berechnen wir die Eigenr ̈aume: Wegen
```
```
A−(−2)I 2 =A+ 2I 2 =
```
##### [

##### 3 3

##### 3 3

##### ]

##### →

##### [

##### 3 3

##### 0 0

##### ]

```
ist
V− 2 (A) = Kern(A+ 2I 2 ) =
```
##### {

```
a
```
##### [

##### 1

##### − 1

##### ]∣∣

```
∣∣a∈K
```
##### }

```
= span
```
##### {[

##### 1

##### − 1

##### ]}

##### ,

```
also dim(V− 2 (A)) = 1, und
```
```
V 4 (A) = Kern(A− 4 I 2 ) = Kern
```
##### ([

##### − 3 3

##### 3 − 3

##### ])

```
= span
```
##### {[

##### 1

##### 1

##### ]}

##### ,

```
also dim(V 4 (A)) = dim(Kern(A− 4 I 2 )) = 1.
```
```
2) F ̈urA=
```
```
[ 2 i1+i
0 1
```
##### ]

```
istpA(z) = (z− 2 i)(z−1), also hatAdie Eigenwerte 2iund 1,
beide mit algebraischer Vielfachheit 1. Wir berechnen die Eigenr ̈aume. F ̈urλ 1 = 2i
ist
A− 2 iI 2 =
```
##### [

```
0 1 +i
0 1− 2 i
```
##### ]

##### →

##### [

##### 0 1

```
0 1− 2 i
```
##### ]

##### →

##### [

##### 0 1

##### 0 0

##### ]

##### ,


```
also ist
V 2 i(A) = span
```
##### {[

##### 1

##### 0

##### ]}

##### .

```
F ̈ur den Eigenwert 1 ist
```
##### A− 1 I 2 =

##### [

```
2 i−1 1 +i
0 0
```
##### ]

##### →

##### [

```
1 2 1+i−i 1
0 0
```
##### ]

##### =

##### [

```
1 1 − 53 i
0 0
```
##### ]

##### ,

```
also
V 1 (A) = span
```
```
{[−1+3i
5
1
```
##### ]}

##### .

```
3) F ̈urA= [1 10 1] ist
```
```
pA(z) = det(A−zI 2 ) = det
```
##### ([

```
1 −z 1
0 1 −z
```
##### ])

```
= (1−z)^2 ,
```
```
so dassAnur den Eigenwert 1 mit algebraischer Vielfachheit 2 besitzt. Fur die ̈
geometrische Vielfachheit rechnen wir
```
```
Kern(A− 1 I 2 ) = Kern
```
##### ([

##### 0 1

##### 0 0

##### ])

```
= span
```
##### {[

##### 1

##### 0

##### ]}

##### ,

```
alsog(1,A) = dim(Kern(A− 1 I 2 )) = 1<2 =a(1,A).
```
Die algebraische und geometrische Vielfachheit eines Eigenwerts sind nicht immer
gleich, wie das letzte Beispiel zeigt. Es gilt aber folgender Zusammenhang.

Satz 33.12.IstA∈Kn,nmit Eigenwertλ∈K, so gilt

```
1 ≤g(λ)≤a(λ),
```
d.h. die geometrische Vielfachheit ist immer kleiner als die algebraische Vielfachheit.

Beispiel 33.13. Die MatrixA=

##### [1 1 1

```
0 1 1
0 0 4
```
##### ]

```
hat die Eigenwerte 1 und 4 mit algebraischen
```
Vielfachheitena(1) = 2 unda(4) = 1. Die geometrischen Vielfachheiten sind

```
g(1) = dim(Kern(A− 1 I 3 )) = 3−Rang(A−I 3 ) = 3−Rang
```
##### 

##### 

##### 

##### 

##### 0 1 1

##### 0 0 1

##### 0 0 3

##### 

##### 

##### 

##### = 1

```
g(4) = dim(Kern(A− 4 I 3 )) = 3−Rang(A− 4 I 3 ) = 3−Rang
```
##### 

##### 

##### 

##### 

##### −3 1 1

##### −3 0 1

##### 0 0 0

##### 

##### 

##### 

##### = 1.

F ̈ur den Eigenwert 1 gilt also 1 =g(1) < a(1) = 2. Die geometrische Vielfachheit
ist echt kleiner als die algebraische Vielfachheit. F ̈ur den Eigenwert 4 gilt hingegen
1 =g(4) =a(4), d.h. beide Vielfachheiten sind gleich.


### 33.3 Eigenvektoren und lineare Unabh ̈angigkeit

Eigenvektoren zu verschiedenen Eigenwerten sind linear unabh ̈angig.
Zur Motivation seien~v 1 ,~v 2 Eigenvektoren vonA∈Kn,nzu den verschiedenen Eigen-
wertenλ 1 ,λ 2 (d.h. es giltλ 16 =λ 2 ). Wir rechnen nach, dass~v 1 und~v 2 linear unabh ̈angig
sind. Seienα 1 ,α 2 ∈Kmit
α 1 ~v 1 +α 2 ~v 2 =~ 0. (33.2)

Wir wollen nachrechnen, dassα 1 =α 2 = 0. Dazu multiplizieren wir (33.2) einmal mit
λ 1 und einmal mitAund erhalten:

```
~0 =α 1 λ 1 ~v 1 +α 2 λ 1 ~v 2
~0 =α 1 A~v 1 +α 2 A~v 2 =α 1 λ 1 ~v 1 +α 2 λ 2 ~v 2.
```
Die Differenz der beiden Gleichungen ergibt

```
~0 =α 2 (λ 1 −λ 2 )~v 2.
```
Da~v 26 = 0 (Eigenvektor!), istα 2 (λ 1 −λ 2 ) = 0, und daλ 16 =λ 2 istα 2 = 0. Eingesetzt
in (33.2) erhalten wirα 1 ~v 1 =~0, und da~v 16 =~0 (Eigenvektor!), istα 1 = 0.
Durch Induktion kann man dieses Resultat auf mehrere Eigenwerte verallgemeinern.
Dabei verwendet man im Induktionsschritt genau den Trick aus dem Beweis: Einmal
mitAund einmal mit einem Eigenwert multiplizieren.

Satz 33.14.IstA∈Kn,nmit Eigenvektoren~v 1 ,...,~vrzu den verschiedenen Eigenwer-
tenλ 1 ,...,λr, so sind~v 1 ,...,~vrlinear unabh ̈angig.


Vorlesung 34

## 34 Diagonalisierbarkeit

Diagonalmatrizen sind besonders einfach, an ihnen kann man den Rang und die Eigen-
werte mit Vielfachheiten direkt ablesen. SeiDeineDiagonalmatrix, also

##### D=

##### 

##### 

##### 

##### 

##### 

```
d 1 , 1 0 ... 0
0 d 2 , 2
0
0 ... 0 dn,n
```
##### 

##### 

##### 

##### 

##### 

```
=:diag(d 1 , 1 ,...,dn,n).
```
Die Diagonaleintr ̈aged 1 , 1 ,...,dn,nsind die Eigenwerte vonD(Satz 33.10) und

```
~e 1 =
```
##### 

##### 

##### 

##### 

##### 

##### 1

##### 0

##### 0

##### 0

##### 

##### 

##### 

##### 

##### 

```
,~e 2 =
```
##### 

##### 

##### 

##### 

##### 

##### 0

##### 1

##### 0

##### 0

##### 

##### 

##### 

##### 

##### 

```
,...,~en=
```
##### 

##### 

##### 

##### 

##### 

##### 0

##### 0

##### 0

##### 1

##### 

##### 

##### 

##### 

##### 

##### ,

sind Eigenvektoren vonDmitD~ej=dj,j~ej. F ̈ur jeden Eigenwert sind die algebraische
und geometrische Vielfachheit gleich, und gleich der Anzahl wie oftλjauf der Diagonalen
vonDvorkommt.

### 34.1 Definition und Charakterisierung

Wir sind daher daran interessiert, allgemeine Matrizen auf Diagonalform zu bringen.

Definition 34.1 (Diagonalisierbarkeit). Die MatrixA∈Kn,nheißtdiagonalisierbar,
wenn es eine invertierbare MatrixS∈Kn,ngibt, so dass

```
S−^1 AS=D= diag(λ 1 ,...,λn) =
```
##### 

##### 

##### 

```
λ 1
...
λn
```
##### 

##### 

#####  (34.1)

eine Diagonalmatrix ist. Die MatrixDnennt man eineDiagonalisierungvonA.


Bemerkung 34.2. 1) Wir k ̈onnen die Gleichung (34.1) auch schreiben als

```
A=SDS−^1.
```
```
2) In
```
##### 

##### 

##### 

```
λ 1
...
λn
```
##### 

##### 

```
= diag(λ 1 ,...,λn) sind alle Eintr ̈age außerhalb der Diagonalen
```
```
0.
```
```
3) Beachten Sie:Sist die Basiswechselmatrix von der BasisB={~s 1 ,...,~sn}(Spalten
vonS) in die StandardbasisB 0 , alsoS= idB,B 0. Dann istD=AB,B.
```
```
Wann l ̈asst sich eine Matrix diagonalisieren?
```
Satz 34.3. F ̈urA∈Kn,nsind ̈aquivalent:

```
1)Aist diagonalisierbar.
```
```
2) Es gibt eine Basis vonKnaus Eigenvektoren vonA.
```
```
3) Das charakteristische Polynom zerf ̈allt in Linearfaktoren unda(λ) =g(λ)f ̈ur jeden
Eigenwert vonA.
```
Beweis. 1)⇒2)SeiS−^1 AS=D= diag(λ 1 ,...,λn) diagonal. Mit den Spalten vonS,S=
[
~s 1 ... ~sn
]

ist [
A~s 1 ... A~sn
]
=AS=SD=
[
λ 1 ~s 1 ... λn~sn
]
,

alsoA~sj=λj~sjf ̈ur allej= 1,...,n. DaSinvertierbar ist, sind~sj 6 = 0, also Eigenvektoren vonA.
Weiter sind die Spalten vonSlinear unabh ̈angig und auch eine Basis vonKn. Damit ist~s 1 ,...,~sneine
Basis aus Eigenvektoren vonA.
[ 2)⇒1)Hat andersherumAeine Basis aus Eigenvektoren~s^1 ,...,~snmitA~sj=λj~sj, so istS=
~s 1 ... ~sn
]
invertierbar, und es gilt
AS=
[
A~s 1 ... A~sn
]
=
[
λ 1 ~s 1 ... λn~sn
]
=Sdiag(λ 1 ,...,λn),

also istS−^1 AS= diag(λ 1 ,...,λn) diagonal undAist diagonalisierbar.
2)⇒3)Es gebe eine Basis aus Eigenvektoren. Dann gibt es alsonlinear unabh ̈angige Eigenvektoren.
Da die geometrische Vielfachhheit gerade die maximale Anzahl linear unabh ̈angiger Eigenvektoren ist,
folgt
n≤

```
∑
g(λj)≤
```
```
∑
a(λj)≤n.
```
Dabei haben wirg(λ)≤a(λ) f ̈ur jeden Eigenwert verwendet, und dass die Summe der algebraischen
Vielfachheiten≤deg(pA) =nist. In der Ungleichung gilt dannuberall =, so dass ̈ pAnNullstellen
hat und damit in Linearfaktoren zerf ̈allt, undg(λ)≤a(λ) f ̈ur alle Eigenwerte. (Ein<w ̈urde zun < n
f ̈uhren, was nicht sein kann.)
3)⇒2)Seienλ 1 ,...,λrdie verschiedenen Eigenwerte vonA. DapAin Linearfaktoren zerf ̈allt ist
n=
∑r
j=1a(λr), und dag(λj) =a(λj) gilt, ist auch

∑r
j=1g(λj) =n. W ̈ahlen wir in jedem Eigenraum
eine Basis (g(λj) Elemente) und f ̈ugen alle Vektoren zusammen, so sind diese linear unabh ̈angig (Eigen-
vektoren zu verschiedenen Eigenwerten sind linear unabh ̈angig).nlinear unabh ̈angige Vektoren inKn
bilden aber eine Basis.


Bemerkung 34.4.Die Rechnung im Beweis zeigt:
1) Bilden die Spalten vonSeine Basis aus Eigenvektoren vonA, so istS−^1 ASdia-
gonal.
2) IstS−^1 AS=Ddiagonal, so sind die Spalten vonSEigenvektoren vonA, und auf
der Diagonalen vonDstehen die Eigenwerte vonA.

Satz 34.5. HatA∈Kn,ngenaunverschiedene Eigenwerte, so istAdiagonalisierbar.

Beweis. HatAdienverschiedenen Eigenwerteλ 1 ,...,λn, so hatpAdienverschiedenen
Nullstellenλ 1 ,...,λn, so dasspA in Linearfaktoren zerf ̈allt. Weiter ist 1≤ g(λj) ≤
a(λj) = 1 (dennpAkann nicht mehr alsnNullstellen haben), alsog(λj) =a(λj) f ̈ur alle
Eigenwerte, so dassAdiagonalisierbar ist.

Berechnung einer Diagonalisierung. SeiA∈Kn,ngegeben.
1) Berechne das charakteristische PolynompA.
2) Bestimme die Nullstellenλ 1 ,...,λkvonpA(Eigenwerte vonA) mit algebraischen
Vielfachheiten. (Gibt es keinenNullstellen, so istAnicht diagonalisierbar.)
3) Bestimme f ̈urj= 1,...,kdie Eigenr ̈aume
Vλj= Kern(A−λjIn) ={~x∈Kn|(A−λjIn)~x=~ 0 }
und bestimme jeweils eine BasisBjund die geometrische Vielfachheitg(λj).
4) Giltg(λj) =a(λj) f ̈ur allej= 1,...,k? (Wenn nicht, istAnicht diagonalisierbar.)
5)B := B 1 ∪ B 2 ∪...∪ Bk ist eine Basis aus Eigenvektoren, die als die Spalten
vonSgenommen werden k ̈onnen. SchreibeS=

##### [

```
~s 1 ... ~sn
```
##### ]

```
∈Kn,n. Dann ist
S−^1 AS=D= diag(μ 1 ,...,μn), wobeiμjder Eigenwert zum Eigenvektor~sj ist
(j-te Spalte vonS).
```
Beispiel 34.6.Wir berechnen (wenn m ̈oglich) eine Diagonalisierung von

##### A=

##### 

##### 

##### 1 0 0

##### −2 3 0

##### 0 0 1

##### 

##### .

```
1) Es istpA(z) = (1−z)^2 (3−z) (untere Dreiecksmatrix)
2) Somit hatAdie Eigenwerte 1 mita(1) = 2 und 3 mita(3) = 1.
3) F ̈ur den Eigenwert 1 ist
```
```
Kern(A−I 3 ) = Kern
```
##### 

##### 

##### 

##### 

##### 0 0 0

##### −2 2 0

##### 0 0 0

##### 

##### 

##### 

```
= span
```
##### 

##### 

##### 

##### 

##### 

##### 1

##### 1

##### 0

##### 

##### ,

##### 

##### 

##### 0

##### 0

##### 1

##### 

##### 

##### 

##### 

##### 

```
= span{~s 1 ,~s 2 },
```
```
somit istg(1) = 2 =a(1).
F ̈ur den Eigenwert 3 ist
```
```
Kern(A− 3 I 3 ) = Kern
```
##### 

##### 

##### 

##### 

##### −2 0 0

##### −2 0 0

##### 0 0 − 2

##### 

##### 

##### 

```
= span
```
##### 

##### 

##### 

##### 

##### 

##### 0

##### 1

##### 0

##### 

##### 

##### 

##### 

##### 

```
= span{~s 3 }.
```

```
4) Setze
```
```
S=
```
##### 

##### 

##### 1 0 0

##### 1 0 1

##### 0 1 0

##### 

```
 und D=
```
##### 

##### 

##### 1 0 0

##### 0 1 0

##### 0 0 3

##### 

##### .

```
(Beachte: Reihenfolge der Eigenwerte = Reihenfolge der Eigenvektoren!) Dann ist
S−^1 AS=D.
Zur Probe kann man am BestenAS=SDnachrechnen, dann braucht man die
MatrixSnicht zu invertieren.
```
Beispiel 34.7. 1) SeiA=

##### [

##### 0 − 1

##### 1 0

##### ]

. Dann istpA(z) =z^2 + 1 = (z−i)(z+i), so dass
Adie beiden Eigenwerteiund−ihat. Eigenvektoren:

```
A−iI=
```
##### [

```
−i − 1
1 −i
```
##### ]

```
→iI
```
##### [

```
1 −i
1 −i
```
##### ]

##### →

##### [

```
1 −i
0 0
```
##### ]

##### ,

```
A+iI=
```
##### [

```
i − 1
1 i
```
##### ]

```
I−→iII
```
##### [

##### 0 0

```
1 i
```
##### ]

##### →

##### [

```
1 i
0 0
```
##### ]

##### ,

```
also sind zum Beispiel~s 1 =
```
##### [

```
i
1
```
##### ]

```
∈Kern(A−iI),~s 2 =
```
##### [

```
−i
1
```
##### ]

```
∈Kern(A+iI). Mit
```
##### S=

##### [

```
i −i
1 1
```
##### ]

```
und D=
```
##### [

```
i 0
0 −i
```
##### ]

```
istS−^1 AS=D, oderA=SDS−^1.
```
```
2) Die MatrixA=
```
##### [

```
2 i 1 +i
0 1
```
##### ]

```
aus Beispiel 33.11 hat den Eigenwertλ 1 = 2imit
```
```
Eigenvektor~s 1 = [^10 ] und den Eigenwertλ 2 = 1 mit Eigenvektor~s 2 =
```
```
[−1+3i
5
1
```
##### ]

##### .

```
Setzen wir daher
S=
```
##### [

```
1 −1+3 5 i
0 1
```
##### ]

##### ,

```
so ist
S−^1 =
```
##### [

```
1 1 − 53 i
0 1
```
##### ]

```
und
```
```
S−^1 AS=
```
##### [

```
1 1 − 53 i
0 1
```
##### ][

```
2 i 1 +i
0 1
```
##### ][

```
1 −1+3 5 i
0 1
```
##### ]

##### =

##### [

```
2 i 6+2 5 i
0 1
```
##### ][

```
1 −1+3 5 i
0 1
```
##### ]

##### =

##### [

```
2 i 0
0 1
```
##### ]

##### =D.

```
DassDdiese Gestalt hat, wissen wir aber bereits, man h ̈atte sich die Rechnung
auch sparen k ̈onnen.
Tauscht man die Spalten vonS, betrachtet alsoX=
```
```
[−1+3i
5 1
1 0
```
##### ]

```
, so tauschen auch
die Eintr ̈age inD. Das kann man zurUbung nachrechnen: ̈ X−^1 AX= [1 00 2i].
```

### 34.2 Anwendungen

SeiA∈Kn,ndiagonalisierbar mitS−^1 AS=D= diag(λ 1 ,...,λn) diagonal, alsoA=
SDS−^1. Dann k ̈onnen wir ganz einfach berechnen:

```
1) Potenzen vonA:
```
```
A^2 =AA=SDS−^1 SDS−^1 =SD^2 S−^1 ,
A^3 =AA^2 =SDS−^1 SD^2 S−^1 =SD^3 S−^1 ,
```
```
und allgemein (Induktion!)
```
```
Ak=SDkS−^1 =S
```
##### 

##### 

##### 

```
λk 1
...
λkn
```
##### 

##### 

##### S−^1.

```
Vorteil: F ̈urAkbrauchen wirkMatrixmultiplikationen, mit der Diagonalisierung
nur zwei.
```
```
2) Polynome von Matrizen: Istp(z) =a 0 +a 1 z+...+amzmein Polynom, so ist
```
```
p(A) =a 0 In+a 1 A+...+amAm=a 0 SInS−^1 +a 1 SDS−^1 +...+amSDmS−^1
```
```
=Sp(D)S−^1 =S
```
##### 

##### 

##### 

```
p(λ 1 )
...
p(λn)
```
##### 

##### 

##### S−^1 ,

```
d.h. wir brauchen nur diep(λj) berechnen, sowie zwei Matrizenmultiplikationen.
```
```
3) Funktionen von Matrizen: Wir schreiben
```
```
f(A):=S
```
##### 

##### 

##### 

```
f(λ 1 )
...
f(λn)
```
##### 

##### 

##### S−^1 ,

```
falls die Funktion an den Eigenwerten definiert ist.
Das wird insbesondere f ̈urf(x) =exwichtig beim L ̈osen von Differentialgleichun-
gen in Vorlesung 43.
```
```
4) Rekursionen aufl ̈osen: F ̈ur die Fibonacci-Folgea 0 = 0,a 1 = 1 undan=an− 1 +an− 2
(n≥2) k ̈onnen wir schreiben
[
an
an− 1
```
##### ]

##### =

##### [

##### 1 1

##### 1 0

##### ][

```
an− 1
an− 2
```
##### ]

##### =A

##### [

```
an− 1
an− 2
```
##### ]

```
, n≥ 2 ,
```
##### [

```
a 1
a 0
```
##### ]

##### =

##### [

##### 1

##### 0

##### ]

##### .

```
Dann folgt [
an
an− 1
```
##### ]

```
=An−^1
```
##### [

```
a 1
a 0
```
##### ]

```
=An−^1
```
##### [

##### 1

##### 0

##### ]

##### ,


so dass wirandirekt berechnen k ̈onnen (ohne allea 0 ,a 1 ,...,an− 1 vorher zu be-

rechnen). Hier ist mitz+=1+

```
√ 5
2 undz−=
```
```
1 −√ 5
2
```
##### A=SDS−^1 =

##### [

```
z+ z−
1 1
```
##### ][

```
z+ 0
0 z−
```
##### ]

##### 1

##### √

##### 5

##### [

```
1 −z−
− 1 z+
```
##### ]

also

```
an=
```
##### 1

##### √

##### 5

```
(zn+−z−n) =
```
##### 1

##### √

##### 5

##### ((

##### 1 +

##### √

##### 5

##### 2

```
)n
−
```
##### (

##### 1 −

##### √

##### 5

##### 2

```
)n)
.
```
Allgemeiner kann manan=α 1 an− 1 +...+αkan−kmit gegebenena 0 ,a 1 ,...,ak− 1
schreiben als 

```



```
```
an
an− 1
..
.
an−k+1
```
##### 

##### 

##### 

##### 

##### =

##### 

##### 

##### 

##### 

```
α 1 α 2 ... αn
1 0 ... 0
0
```
##### ... ... ..

##### .

##### 0 0 1 0

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

```
an− 1
an− 2
..
.
an−k
```
##### 

##### 

##### 

##### 

L ̈asst sich die Matrix diagonalisieren, kann man wie eben eine explizite Formel f ̈ur
anfinden.


Vorlesung 35

## 35 Vektorr ̈aume mit Skalarprodukt

In der Ebene und im dreidimensionalen Raum haben wir einen Abstandsbegriff und
k ̈onnen Winkel zwischen Vektoren messen. Die zugrundeliegenden Begriffe sind die Norm
und das Skalarprodukt, die wir f ̈ur allgemeine Vektorr ̈aumeuber ̈ RoderCkennen lernen.

### 35.1 Norm

Wir wollen die L ̈ange von Vektoren und den Abstand zwischen zwei Vektoren messen.
InRoderCmessen wir Abst ̈ande und L ̈angen meist mit dem Absolutbetrag. InR^2 ist
die” ̈ubliche“ L ̈ange eines Vektors‖[xx^12 ]‖ 2 =

##### √

```
x^21 +x^22.
```
##### 0

```
[xx^12 ]
```
```
x^22
```
```
x^21
```
Die L ̈ange hat die folgenden Eigenschaften:

- L ̈angen sind immer positiv:‖~x‖ 2 ≥0, und‖~x‖ 2 = 0⇔~x=~0.
- Bei Streckung des Vektors umλwird seine L ̈ange mit|λ|multipliziert:

```
‖λ~x‖ 2 =
```
##### √

```
(λx 1 )^2 + (λx 2 )^2 =|λ|
```
##### √

```
x^21 +x^22 =|λ|‖~x‖ 2.
```
- Dreiecksungleichung:‖~x+~y‖ 2 ≤‖~x‖ 2 +‖~y‖ 2.

```
‖~x‖ 2
```
```
‖~y‖ 2
```
```
‖~x+~y‖ 2
```
```
Der Begriff der Norm verallgemeinert diesen Begriff der L ̈ange.
```
Definition 35.1(Norm).SeiVeinK-Vektorraum. EineNormaufV ist eine Abbildung
‖·‖:V →Rmit den folgenden drei Eigenschaften:


```
1) Positive Definitheit: F ̈ur allev∈V ist‖v‖≥0, und‖v‖= 0 nur f ̈urv= 0.
```
```
2) Homogenit ̈at: F ̈ur allev∈V undλ∈Kgilt‖λv‖=|λ|‖v‖.
```
```
3) Dreiecksungleichung: F ̈ur allev,w∈V gilt‖v+w‖≤‖v‖+‖w‖.
```
Beispiel 35.2. 1) Der Absolutbetrag aufV =RoderV =Cist eine Norm.

```
2) InKnist
```
```
‖~x‖ 2 =
```
##### √√

##### √

##### √

```
∑n
```
```
j=1
```
```
|xj|^2
```
```
eine Norm, die sogenannte 2-NormoderStandardnorm. IstK=Rso wird sie auch
euklidische Normgenannt. InR^2 oderR^3 ist das genau die
” ̈
ubliche L ̈ange“ von
Vektoren.
Allgemeiner ist fur reelles 1 ̈ ≤p <∞
```
```
‖~x‖p=
```
##### 

##### 

```
∑n
```
```
j=1
```
```
|xj|p
```
##### 

##### 

(^1) p
eine Norm aufKn, die sogenanntep-Norm.
3) DieMaximumsnorm
‖~x‖∞= max
j=1,...,n
|xj|= max{|x 1 |,...,|xn|}
ist eine Norm aufKn. Man kann zeigen, dass limp→∞‖~x‖p=‖~x‖∞f ̈ur jedes~x∈Kn
gilt. Daher wird die Maximumsnorm auch∞-Normgenannt.
4) Eine weitere Norm ist dieMaximumsnorm(von Funktionen)
‖f‖∞= max
x∈[a,b]
|f(x)|.

### 35.2 Skalarprodukte

Definition 35.3(Skalarprodukt).SeiV ein Vektorraumuber ̈ K. Eine Abbildung〈·,·〉:
V×V →Kheißt einSkalarproduktaufV falls f ̈ur alleu,v,w∈V undλ∈Kgilt

```
1) Linearit ̈at im ersten Argument:
```
```
〈u+v,w〉=〈u,w〉+〈v,w〉
〈λv,w〉=λ〈v,w〉,
```
```
2) Symmetrie:〈v,w〉=〈w,v〉
```

```
3) Positive Definitheit:〈v,v〉≥0 f ̈ur allev, und〈v,v〉= 0 genau dann wennv= 0.
```
IstK=Rso heißt ein Vektorraum mit Skalarprodukt eineuklidischer Vektorraum, ist
K=Cso nennt man ihn einenunit ̈aren Vektorraum.

Bemerkung 35.4. 1) F ̈ur das zweite Argument eines Skalarprodukts gilt:

```
〈v,u+w〉=〈u+w,v〉=〈u,v〉+〈w,v〉=〈u,v〉+〈w,v〉=〈v,u〉+〈v,w〉
```
```
und
〈v,λw〉=〈λw,v〉=λ〈w,v〉=λ〈w,v〉=λ〈v,w〉.
F ̈urK=Ristλ=λund das Skalarprodukt ist auch linear im zweiten Argument.
IstK=C, so ist das Skalarprodukt nicht linear im zweiten Argument, da man
zwar Summen auseinander ziehen kann, aber Skalare komplex konjugiert aus dem
zweiten Argument”herausgezogen“ werden. (Man sagt, das Skalarprodukt istse-
milinearoderantilinearim zweiten Argument.)
```
```
2) In einemreellenVektoraum mit Skalarprodukt ist〈w,v〉 ∈R, so dass man das
komplex Konjugieren bei der Symmetrie weglassen kann:〈v,w〉=〈w,v〉.
```
```
3) In manchen B ̈uchern (insbesondere in der Physik) wird gefordert, dass ein Skalar-
produkt auf einem komplexen Vektorraum linear im zweiten Argument ist, so dass
man dann〈v,λw〉=λ〈v,w〉=〈λv,w〉hat.
```
Beispiel 35.5. 1) DasStandardskalarproduktinKnist

```
〈~x,~y〉=
```
```
∑n
```
```
j=1
```
```
xjyj=~yH~x.
```
```
Speziell f ̈urR^2 haben wir das Skalarprodukt〈~x,~y〉=x 1 y 1 +x 2 y 2.
Dies ist tats ̈achlich ein Skalarprodukt, denn:
```
```
(a) Linearit ̈at im ersten Argument:
```
```
〈~x+~y,~z〉=
```
```
∑n
```
```
j=1
```
```
(xj+yj)zj=
```
```
∑n
```
```
j=1
```
```
xjzj+
```
```
∑n
```
```
j=1
```
```
yjzj=〈~x,~z〉+〈~y,~z〉,
```
```
〈λ~x,~y〉=
```
```
∑n
```
```
j=1
```
```
λxjyj=λ
```
```
∑n
```
```
j=1
```
```
xjyj=λ〈~x,~y〉.
```
```
(b) Symmetrie:
```
```
〈~x,~y〉=
```
```
∑n
```
```
j=1
```
```
xjyj=
```
```
∑n
```
```
j=1
```
```
xjyj=
```
```
∑n
```
```
j=1
```
```
xjyj=〈~y,~x〉.
```

```
(c) Positive Definitheit:〈~x,~x〉=
```
```
∑n
j=1xjxj=
```
```
∑n
j=1|xj|
```
(^2) ≥0. Weiter folgt aus
〈~x,~x〉= 0, dass alle Summanden|xj|^2 ≥0 auch = 0 sind, also|xj|= 0 und
dannxj= 0, so dass~x=~0 ist.
2) Gewichtetes Skalarprodukt aufKn: Sindw 1 ,...,wn>0, so definiert
〈~x,~y〉w=
∑n
j=1
wjxjyj
ein Skalarprodukt aufKn. Sind allewj= 1, so ist dies das Standardskalarprodukt.
3) DasL^2 -Skalarprodukt auf dem Vektorraum der stetigen FunktionenV =C([a,b])
ist
〈f,g〉=
∫b
a
f(x)g(x)dx.
Linearit ̈at im ersten Argument und Symmetrie rechnen sich genauso einfach nach
wie eben (versuchen Sie es!). Die positive Definitheit ist etwas komplizierter, wir
verzichten darauf.
Beim Standardskalarprodukt sehen wir, dass〈~v,~v〉=‖~v‖^22 ist, d.h. die 2-Norm l ̈asst
sich aus dem Skalarprodukt berechnen:‖~v‖ 2 =

##### √

〈~v,~v〉. Allgemein erh ̈alt man so aus
einem Skalarprodukt eine Norm.

Satz 35.6. IstV einK-Vektorraum mit Skalarprodukt〈·,·〉, so ist

```
‖·‖:V →R, ‖v‖:=
```
##### √

```
〈v,v〉,
```
eine Norm aufV. Diese heißtvom Skalarprodukt induzierte Norm(= zum Skalarprodukt
zugeh ̈orige Norm).

Beweis. Nachrechnen der drei Normeigenschaften:
1) Positive Definitheit: Wegen〈v,v〉 ≥0 und = 0 nur f ̈urv= 0, folgt dass‖v‖=
√
〈v,v〉 ≥0 und
‖v‖= 0 ist ̈aquivalent zu〈v,v〉= 0, also zuv= 0.
2) Homogenit ̈at: Es ist

```
‖λv‖=
```
```
√
〈λv,λv〉=
```
```
√
λ〈v,λv〉=
```
```
√
λλ〈v,v〉=
```
```
√
λλ
```
```
√
〈v,v〉=|λ|‖v‖.
```
```
3) Dreiecksungleichung: F ̈ur die Dreiecksungleichung ben ̈otigt man dieCauchy-Schwarz-Ungleichung
|〈v,w〉|≤‖v‖‖w‖, die wir nicht nachrechnen. F ̈urv,w∈V ist
‖v+w‖^2 =〈v+w,v+w〉=〈v,v〉+〈v,w〉+〈w,v〉+〈w,w〉
=‖v‖^2 +〈v,w〉+〈v,w〉+‖w‖^2 =‖v‖^2 + 2 Re(〈v,w〉) +‖w‖^2
≤‖v‖^2 + 2|〈v,w〉|+‖w‖^2 ≤‖v‖^2 + 2‖v‖‖w‖+‖w‖^2
= (‖v‖+‖w‖)^2 ,
woraus durch Wurzelziehen die Dreiecksungleichung folgt.
```

Beispiel 35.7. 1) Die 2-Norm wird vom Standardskalarprodukt induziert:

```
‖~x‖ 2 =
```
##### √√

##### √

##### √

```
∑n
```
```
j=1
```
```
|xj|^2 =
```
##### √

```
〈~x,~x〉.
```
```
2) DieL^2 -Norm wird vomL^2 -Skalarprodukt induziert:
```
```
‖f‖L 2 =
```
##### √

```
〈f,f〉=
```
##### √∫

```
b
a
```
```
f(x)f(x)dx=
```
##### √∫

```
b
a
```
```
|f(x)|^2 dx.
```
```
3) Nicht von einem Skalarprodukt induziert sind: Diep-Norm fallsp 6 = 2, die Maxi-
mumsnorm.
```
### 35.3 Orthogonale Vektoren

Aus demR^2 mit dem Standardskalarprodukt wissen Sie: zwei Vektoren sind senkrecht
zueinander, genau dann wenn ihr Skalarprodukt Null ist. Zum Beispiel sind~x=

##### [ 2

```
− 1
```
##### ]

und~y= [^12 ] senkrecht zueinander:

```
x 1
```
```
x 2
```
```
~x
```
```
~y
```
Das Skalarprodukt ist〈~x,~y〉= 2·1 + (−1)·2 = 0. Dies nimmt man als Ausgangspunkt
f ̈ur die folgende Definition.

Definition 35.8(orthogonale und orthonormale Vektoren).SeiV ein Vektorraum mit
Skalarprodukt〈·,·〉. Die Vektorenv 1 ,...,vk∈V heißen

```
1)orthogonal(=senkrecht), falls
```
```
〈vi,vj〉= 0 fur ̈ i 6 =j.
```
```
2)orthonormal, falls
```
```
〈vi,vj〉=
```
##### {

```
1 fur ̈ i=j
0 fur ̈ i 6 =j,
d.h. wenn sie orthogonal sind und Norm (L ̈ange) 1 haben.
```

```
Zwei Vektorenu,vsind also orthogonal genau dann, wenn〈u,v〉= 0.
Fur eine von einem Skalarprodukt induzierte Norm gilt der Satz des Pythagoras. ̈
```
Satz 35.9 (Satz des Pythagoras). SeiV ein Vektorraum mit Skalarprodukt〈·,·〉und
induzierter Norm‖·‖. Sindu,v∈V orthogonal, so gilt

```
‖u+v‖^2 =‖u‖^2 +‖v‖^2.
```
Allgemeiner gilt: Sindv 1 ,...,vk∈V orthogonal, so gilt

```
‖v 1 +...+vk‖^2 =‖v 1 ‖^2 +...+‖vk‖^2 ,
```
Beweis. Das rechnen wir direkt nach:

```
‖u+v‖^2 =〈u+v,u+v〉=〈u,u+v〉+〈v,u+v〉=〈u,u〉+〈u,v〉
︸︷︷︸
=0
```
```
+〈v,u〉
︸︷︷︸
=0
```
```
+〈v,v〉
```
```
=‖u‖^2 +‖v‖^2.
```
Fur ̈ kVektoren geht es ̈ahnlich.

```
u
```
```
v
```
```
u+v
```
```
‖u‖
```
```
‖u+v‖ ‖v‖
```
### 35.4 Orthonormalbasen

Was macht kartesische Koordinatensysteme besser als andere Koordinatensysteme?

```
vs.
```
Ein Vorteil ist, dass wir einen Vektor in der ersten (kartesischen) Basis ganz einfach
als Linearkombination der beiden Basisvektoren schreiben k ̈onnen, w ̈ahrend das in der
zweiten komplizierter ist.

Definition 35.10(Orthonormalbasis). SeiV ein Vektorraum mit Skalarprodukt〈·,·〉.
Eine Basis{u 1 ,u 2 ,...,un}vonV heißtOrthonormalbasis(kurz ONB), falls die Vektoren
orthonormal sind.

Satz 35.11. 1) Orthonormale Vektoren sind linear unabh ̈angig.

```
2) Sindu 1 ,...,unorthonormal und istn= dim(V), so ist{u 1 ,...,un}eine ONB
vonV.
```

Beweis. Seienu 1 ,...,un orthonormal und seienλ 1 ,...,λn ∈ Kmit 0 =

∑n
j=1λjuj.
Dann ist

```
0 =〈 0 ,uk〉=
```
```
〈n
∑
```
```
j=1
```
```
λjuj,uk
```
##### 〉

##### =

```
∑n
```
```
j=1
```
```
λj 〈uj,uk〉
︸ ︷︷ ︸
=0 f ̈urj 6 =k
```
```
=λk〈uk,uk〉
︸ ︷︷ ︸
=1
```
```
=λk.
```
Da dies f ̈ur jedesk= 1,...,ngilt, sindλ 1 =...= λn= 0 und u 1 ,...,un sind li-
near unabh ̈angig. Die zweite Aussage gilt, danlinear unabh ̈angige Vektoren einesn-
dimensionalen Vektorraums eine Basis bilden (Satz 11.9).

Beispiel 35.12. 1) Die Standardbasis imK^2 ist eine Orthonormalbasis bez ̈uglich des
Standardskalarprodukts, denn
〈[
1
0

##### ]

##### ,

##### [

##### 1

##### 0

##### ]〉

##### = 1,

##### 〈[

##### 1

##### 0

##### ]

##### ,

##### [

##### 0

##### 1

##### ]〉

##### =

##### 〈[

##### 0

##### 1

##### ]

##### ,

##### [

##### 1

##### 0

##### ]〉

##### = 0,

##### 〈[

##### 0

##### 1

##### ]

##### ,

##### [

##### 0

##### 1

##### ]〉

##### = 1.

```
2) Allgemeiner ist die Standardbasis imKneine Orthonormalbasis bez ̈uglich des Stan-
dardskalarprodukts imKn.
```
```
3) SeiV =R^2 mit dem Standardskalarprodukt〈·,·〉. Die Vektoren~u 1 = √^12 [^11 ] und
~u 2 =√^12
```
##### [ 1

```
− 1
```
##### ]

```
sind orthonormal, denn
```
```
〈~u 1 ,~u 1 〉=
```
##### 1

##### √

##### 2

##### ·

##### 1

##### √

##### 2

##### +

##### 1

##### √

##### 2

##### ·

##### 1

##### √

##### 2

##### =

##### 1

##### 2

##### +

##### 1

##### 2

##### = 1,

```
〈~u 2 ,~u 2 〉=
```
##### 1

##### √

##### 2

##### ·

##### 1

##### √

##### 2

##### + (−

##### 1

##### √

##### 2

##### )·(−

##### 1

##### √

##### 2

##### ) =

##### 1

##### 2

##### +

##### 1

##### 2

##### = 1,

```
〈~u 1 ,~u 2 〉=〈~u 2 ,~u 1 〉=
```
##### 1

##### √

##### 2

##### ·

##### 1

##### √

##### 2

##### +

##### 1

##### √

##### 2

##### ·(−

##### 1

##### √

##### 2

##### ) = 0.

```
Daher ist{~u 1 ,~u 2 }eine Orthonormalbasis vonR^2.
```
Orthonormalbasen haben viele Vorz ̈uge. Insbesondere lassen sich die Koordinaten
eines Vektors ganz einfach berechnen.

Satz 35.13.SeiV ein Vektorraum mit Skalarprodukt〈·,·〉und einer Orthonormalbasis
B={u 1 ,...,un}. Dann gilt f ̈ur jeden Vektorv∈V

```
v=
```
```
∑n
```
```
j=1
```
```
〈v,uj〉uj,
```
d.h. die Koordinaten vonvin der OrthonormalbasisBlassen sich durch Berechnung der
Skalarprodukte〈v,uj〉bestimmen.


Beweis. Das l ̈asst sich leicht nachrechnen. DaBeine Basis ist, l ̈asst sichvdarstellen als

```
v=
```
```
∑n
```
```
j=1
```
```
αjuj
```
mit Skalarenαj∈K. Dann ist

```
〈v,uk〉=
```
```
〈 n
∑
```
```
j=1
```
```
αjuj,uk
```
##### 〉

##### =

```
∑n
```
```
j=1
```
```
αj 〈uj,uk〉
︸ ︷︷ ︸
=0 fur ̈ j 6 =k
```
```
=αk〈uk,uk〉
︸ ︷︷ ︸
=1
```
```
=αk.
```
### 35.5 Orthogonale Matrizen

Wir untersuchen nun Matrizen, deren Spalten eine Orthonormalbasis bilden. Wie sich
herausstellt werden zum Beispiel Drehungen und Spiegelungen durch solche Matrizen
beschrieben.

Definition 35.14(Orthogonale Matrix).Eine MatrixQ∈Rn,nheißtorthogonal, falls
QTQ=In.

```
SchreibeQ=
```
##### [

```
q 1 ... qn
```
##### ]

```
mit den Spaltenq 1 ,...,qn∈Rn. Dann gilt
```
##### QTQ=

##### 

##### 

##### 

```
qT 1
..
.
qTn
```
##### 

##### 

##### 

##### [

```
q 1 ... qn
```
##### ]

##### =

##### 

##### 

##### 

```
qT 1 q 1 ... qT 1 qn
..
.
```
##### ... ..

##### .

```
qTnq 1 ... qTnqn
```
##### 

##### 

##### 

also

```
QTQ=In ⇔ qTiqj=
```
##### {

```
1 f ̈uri=j
0 f ̈uri 6 =j.
```
Dies zeigt:Qist orthogonal genau dann, wenn die Spalten vonQeine Orthonormalbasis
(bzgl. des Standardskalarprodukts) vonRnsind.

Beispiel 35.15. 1) Die Einheitsmatrix ist orthogonal:InTIn=InIn=In.

```
2) Die Matrix
Q=
```
##### [

```
cos(α) −sin(α)
sin(α) cos(α)
```
##### ]

##### ∈R^2 ,^2

```
ist orthogonal, denn
```
```
QTQ=
```
##### [

```
cos(α) sin(α)
−sin(α) cos(α)
```
##### ][

```
cos(α) −sin(α)
sin(α) cos(α)
```
##### ]

##### =

##### [

```
cos(α)^2 + sin(α)^2 −cos(α) sin(α) + sin(α) cos(α)
−sin(α) cos(α) + cos(α) sin(α) sin(α)^2 + cos(α)^2
```
##### ]

##### =I 2.

```
Multiplikation mit der MatrixQfuhrt eine Drehung um den Winkel ̈ αaus (ma-
thematisch positiv, also entgegen dem Uhrzeigersinn):
```

```
x 1
```
```
x 2
```
```
~e 1
```
```
~e 2
```
```
Q~e 1 =
```
##### [

```
cos(α)
sin(α)
```
```
Q~e 2 = ]
```
##### [

```
−sin(α)
cos(α)
```
##### ]

```
α
```
```
α
```
Satz 35.16.SeiQ∈Rn,northogonal. Dann gilt:

```
1) det(Q) =± 1
```
```
2) Qist invertierbar undQ−^1 =QT. Insbesondere gilt auchQQT=In.
```
```
3) F ̈ur alle~x,~y∈Rngilt〈Q~x,Q~y〉=〈~x,~y〉(Standardskalarprodukt).
```
```
4) F ̈ur alle~x∈Rngilt‖Q~x‖ 2 =‖~x‖ 2.
```
Die letzten beiden Eigenschaften bedeuten, dass die Multiplikation mit einer orthogonalen
Matrix sowohl Winkel als auch die euklidische L ̈ange erh ̈alt.

Beweis. 1)Ubungsaufgabe. ̈
2) Da det(Q) 6 = 0 ist, istQinvertierbar, und daQTQ=InistQT=Q−^1 die Inverse.
3) Mit dem Standardskalarprodukt ist

```
〈Q~x,Q~y〉= (Q~y)TQ~x=~yTQTQ
︸︷︷︸
=In
```
```
~x=~yT~x=〈~x,~y〉.
```
```
4) Mit 3) ist
‖Q~x‖ 2 =
```
##### √

```
〈Q~x,Q~x〉=
```
##### √

```
〈~x,~x〉=‖~x‖ 2.
```
Beispiel 35.17.Sei~u∈R^2 und~u 6 =~0. Wir wollen die Spiegelung an der zu~uorthogonalen Ursprungs-
geraden bestimmen. Wir nehmen zuerst~uals normiert an (‖~u‖ 2 = 1), so dass〈~v,~u〉~uder Anteil von~v
in Richtung~uist.

```
~u
```
```
~v
〈~v,~u〉~u
```
```
~v−〈~v,~u〉~u
```
```
~v− 2 〈~v,~u〉~u
```

Die Spiegelung von~van der zu~uorthogonalen Ursprungsgeraden ist daher

```
~v− 2 〈~v,~u〉~u=~v− 2 ~u〈~v,~u〉=~v− 2 ~u~uT~v= (I 2 − 2 ~u~uT)~v.
```
F ̈ur allgemeines~u 6 =~0 ist‖~u~u‖ 2 normiert und die Spiegelung von~vist dann
(
I 2 − 2 ~u~u

```
T
‖~u‖^22
```
```
)
~v=
```
```
(
I 2 − 2 ~u~u
```
```
T
~uT~u
```
```
)
```
```
︸ ︷︷ ︸
=:H(~u)
```
```
~v.
```
DieHouseholder-MatrixH(~u) beschreibt die Spiegelung an der Ursprungsgeraden, die senkrecht zu~u
ist.
Allgemeiner beschreibt f ̈ur~u∈Rn,~u 6 =~0, dieHouseholder-Matrix

```
H(~u) =I− 2 ~u~u
```
```
T
~uT~u
```
die Spiegelung an der zu~uorthogonalen HyperebeneU ={~v∈Rn| 〈~v,~u〉= 0}, die ein (n−1)-
dimensionaler Teilraum vonRnist.

### 35.6 Unit ̈are Matrizen

Unit ̈are Matrizen sind wie orthogonale Matrizen, nur komplex.

Definition 35.18(Unit ̈are Matrix). Eine MatrixU∈Cn,nheißtunit ̈ar, fallsUHU=
In.

Wie f ̈ur orthogonale Matrizen rechnet man nach:Uist unit ̈ar genau dann, wenn die
Spalten vonU eine Orthonormalbasis (bzgl. des Standardskalarprodukts) vonCnsind.
Auch Satz 35.16 gilt entsprechend f ̈ur unit ̈are Matrizen.

Satz 35.19.SeiU∈Cn,nunit ̈ar. Dann gilt:

```
1)|det(U)|= 1, d.h.det(U) =eiφf ̈urφ∈R.
```
```
2)U ist invertierbar undU−^1 =UH. Insbesondere gilt auchUUH=In.
```
```
3) F ̈ur alle~x,~y∈Cngilt〈U~x,U~y〉=〈~x,~y〉(Standardskalarprodukt).
```
```
4) F ̈ur alle~x∈Cngilt‖U~x‖ 2 =‖~x‖ 2.
```
```
5) Die Eigenwerte vonUliegen auf dem Einheitskreis:|λ|= 1.
```
Beweis. Nur 5): DaU∈Cn,neine komplexe Matrix ist, hatUEigenwerte, etwaU~x=λ~x
mitλ∈Cund~x∈Cn,~x 6 =~0. Dann gilt mit 4)

```
‖~x‖ 2 =‖U~x‖ 2 =‖λ~x‖ 2 =|λ|‖~x‖ 2.
```
Da~x 6 =~0 ist auch‖~x‖ 26 = 0, also folgt|λ|= 1, d.h.λliegt auf dem Einheitskreis.


Vorlesung 36

## 36 Vektorr ̈aume mit Skalarprodukt

Wir lernen wichtige Anwendungen von Skalarprodukten kennen. Dies sind einmal die
Berechnung des k ̈urzesten Abstands von einem Punkt zu einer Geraden oder zu einem
Teilraum. Zum Anderen lernen wir das Gram-Schmidt-Verfahren kennen, mit dem sich
Orthonormabasen berechnen lassen, und einige Anwendungen kennen.

### 36.1 Kurzeste Abst ̈ ̈ande und orthogonale Projektion

Frage:Was ist der kurzeste Abstand zwischen einem Punkt ̈ vund einer Geraden? Anders
gefragt suchen wir den Punktv∗auf der Geraden, der am n ̈achsten am Punktvliegt.
Anschaulich mussv−v∗senkrecht zur Geraden sein:

```
Gerade
```
```
v
```
```
v∗
```
Wie siehtv∗dann aus? Dazu seiu∈V mit‖u‖= 1 der Richtungsvektor der Ursprungs-
geradenU = span{u}. Wir machen den Ansatzv∗ =αu. Die Bedingungv−v∗⊥u
ergibt

```
0 =〈v−v∗,u〉=〈v,u〉−〈v∗,u〉=〈v,u〉−〈αu,u〉=〈v,u〉−α〈u,u〉=〈v,u〉−α,
```
alsov∗=αu=〈v,u〉u. Andersherum zeigt die Rechnung: Wennv∗=αu=〈v,u〉uist,
dann istv−v∗⊥u.

```
span{u}
```
##### 0

```
u
```
```
v
```
```
〈v,u〉u
```
```
v−〈v,u〉u
```

```
Daher zerlegen wirvals
```
```
v=v−v∗+v∗= (v−〈v,u〉u) +〈v,u〉u.
```
Dabei stehtv− 〈v,u〉u senkrecht zur vonu aufgespannten Geraden, und〈v,u〉uist
dieorthogonale Projektion(= dasLot) vonvauf diese Gerade. Dies legt nahe, dass
v∗=〈v,u〉uder Punkt der Geraden mit kleinstem Abstand zuvist.

Satz 36.1. SeiV ein Vektorraum mit Skalarprodukt undU = span{u}mit‖u‖= 1.
F ̈ur v∈V ist v∗ =〈v,u〉udie orthogonale Projektion vonvaufU. Dann istv∗ der
Punkt ausUmit kleinstem Abstand zuv. Der Abstand vonvzuU ist dann genau

```
‖v−〈v,u〉u‖=
```
##### √

```
‖v‖^2 −|〈v,u〉|^2.
```
Beweis. Der Abstand vonvzum Punktαu∈Uist‖v−αu‖. Wie wir gerade nachgerechnet haben, sind
v−〈v,u〉uunduorthogonal, also gilt mit dem Satz des Pythagoras

```
‖v−αu‖^2 =‖v−〈v,u〉u+〈v,u〉u−αu‖^2 =‖v−〈v,u〉u+ (〈v,u〉−α)u‖^2
=‖v−〈v,u〉u‖^2 +‖(〈v,u〉−α)u‖^2 =‖v−〈v,u〉u‖^2 +|〈v,u〉−α|^2 ‖u‖^2
=‖v−〈v,u〉u‖^2 +|〈v,u〉−α|^2.
```
Der zweite Term ist≥0 und = 0 genau dann, wennα=〈v,u〉. D.h.‖v−αu‖^2 ist am kleinsten f ̈ur
α=〈v,u〉. Weiter gilt

```
‖v‖^2 =‖v−〈v,u〉u+〈v,u〉u‖^2 Pythagoras= ‖v−〈v,u〉u‖^2 +‖〈v,u〉u‖^2
=‖v−〈v,u〉u‖^2 +|〈v,u〉|^2 ‖︸u︷︷‖^2 ︸
=1
```
```
.
```
Durch Umstellen folgt die Formel f ̈ur‖v−〈v,u〉u‖.

Genauso findet man den Abstand von einem Punkt zu einer Ebene, und allgemeiner
zu einem Teilraum. Wir halten das als Satz fest, versuchen Sie diesen nachzurechnen!

Satz 36.2.SeiV ein Vektorraum mit Skalarprodukt undU= span{u 1 ,...,uk}mit einer
Orthonormalbasisu 1 ,...,uk vonU. F ̈urv∈V istv∗=

∑k
j=1〈v,uj〉ujdie orthogonale
Projektion vonvaufU. Dann istv∗der Punkt ausUmit kleinstem Abstand zuv. Der
Abstand vonvzuUist dann genau

```
‖v−
```
```
∑k
```
```
j=1
```
```
〈v,uj〉uj‖=
```
##### √

```
‖v‖^2 −‖v∗‖^2 =
```
##### √√

##### √√

```
‖v‖^2 −
```
```
∑k
```
```
j=1
```
```
|〈v,uj〉|^2.
```
Die orthogonale Projektion vonv∈V aufU sieht also fast so aus wie eine”ONB-
Entwicklung“ vonv∈V im Teilraum (statt im ganzen Raum).


### 36.2 Das Gram-Schmidt-Verfahren

Das Gram-Schmidt-Verfahren erlaubt es, aus einer gegebenen Basis eine Orthonormal-
basis zu berechnen.
SeiV ein Vektorraum mit Skalarprodukt〈·,·〉und induzierter Norm‖·‖. SeiB=
{b 1 ,...,bn}eine Basis vonV. Das Gram-Schmidt-Verfahren liefert eine ONB vonV:

```
1) Normiereb 1 :u 1 :=‖b^11 ‖b 1.
```
```
2) F ̈urk= 2,...,n:
```
```
(a) Orthogonalisierebk: Entferne den Anteil vonbk in die bereits gefundenen
Richtungenu 1 ,...,uk− 1 :
```
```
̂uk=bk−
```
```
k∑− 1
```
```
j=1
```
```
〈bk,uj〉uj
```
```
(Dann ist〈̂uk,u`〉= 0 f ̈ur`= 1, 2 ,...,k−1.)
(b) Normiereûk:
uk=
```
##### 1

```
‖ûk‖
̂uk.
```
Dann ist{u 1 ,...,un}eine ONB vonV mit der Eigenschaft, dass

```
span{u 1 ,...,uk}= span{b 1 ,...,bk} f ̈urk= 1, 2 ,...,n.
```
Beispiel 36.3. SeiV = R^3 mit dem Standardskalarprodukt. Gegeben sei die Basis
B={b 1 ,b 2 ,b 3 }mit

```
b 1 =
```
##### 

##### 

##### 1

##### 1

##### 0

##### 

```
, b 2 =
```
##### 

##### 

##### 2

##### 2

##### 1

##### 

```
, b 3 =
```
##### 

##### 

##### 1

##### − 1

##### 1

##### 

##### .

Dann ist
‖b 1 ‖=

##### √

##### 1 + 1 + 0 =

##### √

##### 2 ,

also

```
u 1 =
```
##### 

##### 

##### 

```
√^1
12
√ 2
0
```
##### 

##### 

##### .

Wir othogonalisierenb 2 :

```
û 2 =b 2 −〈b 2 ,u 1 〉u 1 =
```
##### 

##### 

##### 2

##### 2

##### 1

##### 

##### −

##### 〈

##### 

##### 2

##### 2

##### 1

##### 

##### ,

##### 

##### 

##### 

```
√^1
12
√
2
0
```
##### 

##### 

##### 

##### 〉

##### 

##### 

```
√^1
12
√
2
0
```
##### 

##### 

##### =

##### 

##### 

##### 2

##### 2

##### 1

##### 

##### − 2

##### √

##### 2

##### 

##### 

##### 

```
√^1
12
√
2
0
```
##### 

##### 

##### =

##### 

##### 

##### 0

##### 0

##### 1

##### 

##### .


Normierêu 2 : Wegen‖̂u 2 ‖= 1 ist

```
u 2 =
```
##### 1

##### 1

```
̂u 2 =
```
##### 

##### 

##### 0

##### 0

##### 1

##### 

##### .

Orthogonalisiereb 3 :

```
û 3 =b 3 −〈b 3 ,u 1 〉u 1 −〈b 3 ,u 2 〉u 2
```
##### =

##### 

##### 

##### 1

##### − 1

##### 1

##### 

##### −

##### 〈

##### 

##### 1

##### − 1

##### 1

##### 

##### ,

##### 

##### 

##### 

```
√^1
12
√
2
0
```
##### 

##### 

##### 

##### 〉

##### ︸ ︷︷ ︸

```
=0
```
##### 

##### 

##### 

```
√^1
12
√
2
0
```
##### 

##### 

##### −

##### 〈

##### 

##### 1

##### − 1

##### 1

##### 

##### ,

##### 

##### 

##### 0

##### 0

##### 1

##### 

##### 

##### 〉

##### ︸ ︷︷ ︸

```
=1
```
##### 

##### 

##### 0

##### 0

##### 1

##### 

##### =

##### 

##### 

##### 1

##### − 1

##### 0

##### 

##### .

Normierêu 3 : Mit‖̂u 3 ‖=

##### √

```
2 ist
```
```
u 3 =
```
##### 1

##### √

##### 2

##### 

##### 

##### 1

##### − 1

##### 0

##### 

##### =

##### 

##### 

##### 

```
√^1
2
−√^12
0
```
##### 

##### 

##### .

Dann ist{u 1 ,u 2 ,u 3 }eine ONB vonR^3.

### 36.3 QR-Zerlegung

Die QR-Zerlegung ist eine fur die Anwendung sehr wichtige Matrix-Zerlegung. ̈

Satz 36.4(QR-Zerlegung).SeiA∈Rm,nmitm≥n. Dann ist

```
A=QR
```
mit einer orthogonalen MatrixQ∈Rm,mund einer oberen DreiecksmatrixR∈Rm,n,
also

##### R=

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### ∗ ∗ ... ∗

##### 0 ∗ ... ∗

##### ..

##### .

##### ... ... ..

##### .

##### 0 ... 0 ∗

##### 0 0 ... 0

##### ..

##### .

##### ..

##### .

##### ... ..

##### .

##### 0 0 ... 0

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 

IstA∈Cm,nmitm≥n, so ist
A=QR

mit einer unit ̈aren MatrixQ∈Cn,nund einer oberen DreiecksmatrixR∈Cm,n.


Beweis. Wir betrachten nur den Fall einer quadratischen MatrixAmit linear unabh ̈angigen Spalten
(also ein invertierbaresA).
SeiA=
[
a 1 ... an
]
∈Kn,ninvertierbar mit den Spaltena 1 ,...,an∈Kn. Anwendung des
Gram-Schmidt-Verfahrens (mit dem Standardskalarprodukt und der 2-Norm) aufa 1 ,...,anergibt or-
thonormale Vektorenu 1 ,...,unmit

```
span{a 1 ,...,aj}= span{u 1 ,...,uj}, j= 1,...,n.
```
Daher gilt insbesondere

```
aj=
```
```
∑j
i=1
```
```
ri,jui.
```
Aus dem Gram-Schmidt-Verfahren sehen wir, dass

```
aj=
```
```
j∑− 1
```
```
i=1
```
```
〈aj,ui〉ui+ûj=
```
```
j∑− 1
```
```
i=1
```
```
〈aj,ui〉ui+‖̂uj‖ 2 uj.
```
Dann istri,j=〈aj,ui〉f ̈uri= 1,...,j−1. Weiter ist〈aj,uj〉=‖ûj‖ 2 und fur ̈ i > jistri,j=〈aj,ui〉= 0,
dau 1 ,...,unorthogonal sind. Insgesamt ist alsori,j=〈aj,ui〉f ̈ur allei,jund

```
A=QR
```
mit orthogonalem (bzw. unit ̈arem)Q=

```
[
u 1 ... un
]
und einer oberen DreicksmatrixR=
```
```
[
ri,j
]
.
```
Berechnung der QR-Zerlegung f ̈ur invertierbaresA∈Rn,n(bzw.Cn,n):

```
1) Wende das Gram-Schmidt-Verfahren (mit dem Standardskalarprodukt) auf die
Spaltena 1 ,...,anvonAan, um die ONBu 1 ,...,unzu erhalten.
```
```
2) SetzeQ:=
```
##### [

```
u 1 ... un
```
##### ]

. Dann istQorthogonal (bzw. unit ̈ar).
(Probe:QTQ=Inbzw.QHQ=In.)

```
3) Berechnung vonR=
```
##### [

```
ri,j
```
##### ]

```
: Zwei M ̈oglichkeiten:
```
- berechneri,j=〈aj,ui〉(wird bei Gram-Schmidt mit berechnet), oder
- berechneR=QTA(bzw.R=QHA).
Dann istReine obere Dreiecksmatrix.

Dann istA=QReine QR-Zerlegung vonA.

Beispiel 36.5.Sei

```
A=
```
##### 

##### 

##### 1 2 1

##### 1 2 − 1

##### 0 1 1

##### 

##### ∈R^3 ,^3.

In Beispiel 36.3 haben wir die Spalten vonAmit Gram-Schmidt orthonormalisiert. Daher
ist

```
Q=
```
##### [

```
u 1 u 2 u 3
```
##### ]

##### =

##### 

##### 

##### 

```
√^1
2 0
√^1
1 2
√
2 0 −
√^1
2
0 1 0
```
##### 

##### 

##### 


orthogonal (rechnen Sie zur ProbeQTQ=I 3 nach). Die MatrixRerhalten wir als

##### R=QTA=

##### 

##### 

##### 

```
√^1
2
√^1
2 0
0 0 1
√^1
2 −
√^1
2 0
```
##### 

##### 

##### 

##### 

##### 

##### 1 2 1

##### 1 2 − 1

##### 0 1 1

##### 

##### =

##### 

##### 

##### √

##### 2 2

##### √

##### 2 0

##### 0 1 1

##### 0 0

##### √

##### 2

##### 

##### 

oder aus den berechneten Koeffizienten:

##### R=

##### 

##### 

```
‖̂u 1 ‖ 2 〈a 2 ,u 1 〉 〈a 3 ,u 1 〉
0 ‖̂u 2 ‖ 2 〈a 3 ,u 2 〉
0 0 ‖û 3 ‖ 2
```
##### 

##### =

##### 

##### 

##### √

##### 2 2

##### √

##### 2 0

##### 0 1 1

##### 0 0

##### √

##### 2

##### 

##### .

Bemerkung 36.6.FallsAnicht quadratisch ist oder die Spalten linear abh ̈angig sind, modifiziert man
das vorgehen wie folgt: Man rechnet Gram-Schmidt auf den Spalten vonA. Ist einer der Vektoren̂uj= 0
(dies passiert genau dann, wennajlinear abh ̈angig vonu 1 ,...,uj− 1 ist), so l ̈asst manûjweg, beh ̈alt
aber die berechneten Eintr ̈ageri,j=〈aj,ui〉,i= 1,...,j−1 vonR.
Am Ende hat manA=QRmit einer MatrixQmit orthonormalen Spalten und einer oberen
DreiecksmatrixR. Nun erg ̈anzt manQzu einer orthogonalen (unit ̈aren) Matrix und h ̈angt gen ̈ugend
Nullen unten anRan.

Beispiel 36.7.SeiA=

```
[
1 1
1 1
```
```
]
```
. Hier ist‖a 1 ‖=

```
√
2, also
```
```
u 1 =√^1
2
```
```
[
1
1
```
```
]
=
```
```
[ 1
√ 2
√^1
2
```
```
]
.
```
Dann ist

```
̂u 2 =a 2 −〈a 2 ,u 1 〉u 1 =
```
```
[
1
1
```
```
]
−√^2
2
```
```
[ 1
√ 2
√^1
2
```
```
]
=
```
```
[
1
1
```
```
]
−
```
```
[
1
1
```
```
]
=
```
```
[
0
0
```
```
]
.
```
Das ergibt

```
A=
[
u 1
][
r 1 , 1 r 1 , 2
]
=
```
```
[ 1
√ 2
√^1
2
```
```
][
√
2 √^22
```
```
]
=
```
```
[ 1
√ 2 √^12
√^1
2 −
√^1
2
```
```
]
```
```
︸ ︷︷ ︸
=Q
```
```
[√
2 √^22
0 0
```
```
]
```
```
︸ ︷︷ ︸
=R
```
```
.
```
Dabei haben wir im letzten SchrittQzu einer orthogonalen Matrix erg ̈anzt. (Sieht man nicht direkt,
wie manQerg ̈anzen kann, kann man nach und nach die Standardbasisvektoren versuchen und zum
Orthonormalisieren noch einmal Gram-Schmidt anwenden.)


### 36.4 Lineare Regression

Gegeben seien Daten oder Messwerte: (ti,yi),i= 1,...,m.
Gesucht ist eine Geradey(t) =a 1 t+a 2 die diese Punkte”am Besten“ repr ̈asentiert.
Wegen Messfehlern gilt typischerweiseyi≈a 1 ti+a 2 fur alle ̈ i, aber nicht Gleichheit.

```
t
```
```
y
```
```
t 1 t 2 t 3 t 4 t 5 ... tm
```
```
Gesucht sind also die Koeffizientena 0 ,a 1 so dass
```
```
a 1 ti+a 2 −yi=
```
##### [

```
ti 1
```
```
][a 1
a 2
```
##### ]

```
−yi, i= 1,...,m,
```
m ̈oglichst klein wird. Den Abstand messen wir in der 2-Norm (kleinste-Quadrate-Appro-
ximation; engl. least squares approximation), und suchen alsoa 1 ,a 2 so dass
∥
∥∥
∥∥
∥∥
∥
∥

##### 

##### 

##### 

##### 

```
t 1 1
t 2 1
..
.
```
##### ..

##### .

```
tm 1
```
##### 

##### 

##### 

##### 

##### [

```
a 1
a 2
```
##### ]

##### −

##### 

##### 

##### 

##### 

```
y 1
y 2
..
.
ym
```
##### 

##### 

##### 

##### 

##### ∥

##### ∥∥

##### ∥∥

##### ∥∥

##### ∥

##### ∥

```
2
```
##### =

##### √√

```
√√∑m
```
```
i=1
```
```
(a 1 ti+a 2 −yi)^2
```
m ̈oglichst klein wird. Setzen wir

##### A=

##### 

##### 

##### 

```
t 1 1
..
.
```
##### ..

##### .

```
tm 1
```
##### 

##### 

```
,a=
```
##### [

```
a 1
a 2
```
##### ]

```
,y=
```
##### 

##### 

##### 

```
y 1
..
.
ym
```
##### 

##### 

##### ,

so wollen wir also‖Aa−y‖ 2 m ̈oglichst klein bekommen. Mit der QR-ZerlegungA=QR
ist dann

```
‖Aa−y‖ 2 =‖QRa−y‖ 2 =‖Q(Ra−QTy)‖ 2 =‖Ra−QTy‖ 2.
```
DaAnur zwei Spalten hat, istRvon der Form

##### R=

##### 

##### 

##### 

##### 

##### 

```
r 1 , 1 r 1 , 2
0 r 2 , 2
0 0
..
.
```
##### ..

##### .

##### 0 0

##### 

##### 

##### 

##### 

##### 

##### =

##### [ ̃

##### R

##### 0

##### ]

```
mitR ̃=
```
##### [

```
r 1 , 1 r 1 , 2
0 r 2 , 2
```
##### ]

##### .


Teilen wir dannb=QTy=

##### [

```
b 1
b 2
```
##### ]

```
auf mitb 1 ∈R^2 undb 2 ∈Rn−^2 , so ist
```
```
‖Aa−y‖^22 =‖Ra−b‖^22 =
```
##### ∥

##### ∥∥

##### ∥

##### [ ̃

```
Ra
0
```
##### ]

##### −

##### [

```
b 1
b 2
```
##### ]∥∥

##### ∥

##### ∥

```
2
```
```
2
```
##### =

##### ∥

##### ∥∥

##### ∥

##### [ ̃

```
Ra−b 1
−b 2
```
##### ]∥∥

##### ∥

##### ∥

```
2
```
```
2
```
```
=‖Ra ̃ −b 1 ‖^22 +‖b 2 ‖^22 ,
```
und das wird am kleinsten wenn‖Ra ̃ −b 1 ‖ 2 = 0, alsoRa ̃ −b 1 = 0, alsoa=R ̃−^1 b 1 gilt.
Fur dieses ̈ aist der verbleibende Fehler also‖Aa−y‖ 2 =‖b 2 ‖ 2. Wir fassen zusammen.

L ̈osung des kleinste-Quadrate-Problems: Gegeben Messwerte (ti,yi),i= 1, 2 ,...,
m, setze:

##### A=

##### 

##### 

##### 

```
t 1 1
..
.
```
##### ..

##### .

```
tm 1
```
##### 

##### 

```
, y=
```
##### 

##### 

##### 

```
y 1
..
.
ym
```
##### 

##### 

##### .

```
1) Berechne die QR-ZerlegungA=QR.
```
```
2) ErhalteR ̃=
```
##### [

```
r 1 , 1 r 1 , 2
0 r 2 , 2
```
##### ]

```
undb 1 =
```
##### [

```
q 1 T
q 2 T
```
##### ]

```
y.
```
```
3) L ̈ose das LGSR ̃
```
##### [

```
a 1
a 2
```
##### ]

```
=b 1.
```
Dann isty(t) =a 1 t+a 2 die Gerade durch die Messwerte, deren Abweichung den kleinsten
Fehler hat (in der 2-Norm).

Beispiel 36.8.Der Plot am Anfang des Abschnitts illustriert die Datenpunkte

##### A=

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 1 1

##### 2 1

##### 3 1

##### 4 1

##### 5 1

##### 6 1

##### 7 1

##### 

##### 

##### 

##### 

##### 

##### 

##### 

```
,y=
```
##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### 1

##### 2

##### 1. 8

##### 1. 9

##### 3

##### 2. 5

##### 3. 2

##### 

##### 

##### 

##### 

##### 

##### 

##### 

##### .

Fur diese berechnet man ̈ a 1 = 0.2714 unda 2 = 1.0286 (auf 4 Nachkommastellen gerun-
det), was die eingezeichnete Gerade ergibt.


Vorlesung 37

## 37 Reelle Fourieranalysis

- Differenzierbare Funktionen lassen sich lokal gut durch (Taylor-)Polynome appro-
    ximieren.
- Wir lernen, wie man periodische Funktionen sogar global durch Sinus- und Cosi-
    nusfunktionen approximiert.
- Die Fourierapproximation ist das mathematische Werkzeug f ̈ur die Frequenzanalyse
    von periodischen Schwingungen.

Die Taylorformel gibt einem die M ̈oglichkeit, eine (gen ̈ugend oft differenzierbare)
Funktionfin der N ̈ahe eines Punktesx 0 durch ein Polynom zu approximieren:

```
f(x) =
```
```
∑n
```
```
k=0
```
```
ak(x−x 0 )k mitak=
f(k)(x 0 )
k!
```
Die Taylor-Approximation eines Polynoms liefert wieder dasselbe Polynom. F ̈ur andere
Funktionen gibt der Satz von Taylor (Satz 24.2) Auskunft ̈uber den Approximationsfeh-
ler.
In der Taylor-Theorie approximiert man also”komplizierte“ Funktionen mit einfa-
chen Bausteinen, n ̈amlich Potenzen vonx, genauer mit Linearkombinationen von Poten-
zen vonx, also mit Polynomen.
Bei den in der Mechanik und Elektrotechnik h ̈aufig auftretenden periodischen Funk-
tionen (Schwingungen, Wechselstrom) ist es nahe liegend, andere Bausteine zu nehmen,
n ̈amlich solche die selber schon periodisch sind: Sinus- und Cosinusschwingungen zum
Beispiel, sogenannteharmonische Schwingungen(vgl. Beispiel 6.6). Wir bezeichnen die
Variable in diesem Abschnitt mitt, weil sie oft die Zeit repr ̈asentiert.

### 37.1 Trigonometrische Polynome

Definition 37.1(periodische Funktion). Die Funktionf:R→Rheißtperiodisch mit
der PeriodeT >0 oder kurzT-periodisch, wennf(t+T) =f(t) f ̈ur allet∈Rgilt.


Eine Periode muss nicht die kleinste Zahl mit der Eigenschaftf(t+T) =f(t) f ̈ur
alletsein. IstTeine Periode vonf, so sind auch 2T, 3T, 4T,... Perioden der Funktion,
zum Beispiel istf(t+ 2T) =f((t+T) +T) =f(t+T) =f(t).

Beispiel 37.2.SeiT >0 undω=^2 Tπdie zugeh ̈orige Frequenz. Die Funktionen

```
cos(kωt), sin(kωt)
```
sind f ̈ur jedesk∈Nperiodisch mit der PeriodeT:

```
cos(kω(t+T)) = cos(kωt+kωT) = cos(kωt+k 2 π) = cos(kωt).
```
Genauso f ̈ur sin(kωt). Dann sind auch die Linearkombinationen

```
∑n
```
```
k=0
```
```
(akcos(kωt) +bksin(kωt))
```
T-periodisch. Dabei sind sin(0ωt) = 0 und cos(0ωt) = 1, und man schreibt typischerweise

```
a 0
2
```
##### +

```
∑n
```
```
k=1
```
```
(akcos(kωt) +bksin(kωt))
```
(Der Faktor^12 ist Konvention, wir werden sp ̈ater sehen, warum dies sinnvoll ist.)

```
t
T
```
##### 1

##### − 1

```
sin(ωt) sin(2ωt)
```
Auch sin(2ωt) ist periodisch mit PeriodeT. Die kleinste Periode von sin(2ωt) istT/2.

Definition 37.3(Trigonometrische Polynome). Funktionen der Form

```
a 0
2
```
##### +

```
∑n
```
```
k=1
```
```
(akcos(kωt) +bksin(kωt))
```
heißentrigonometrische Polynomevom Grad oder von der Ordnungn.

Physikalische Interpretation.Physikalisch besteht ein trigonometrisches Polynom aus
derUberlagerung einer ̈ Grundschwingungder Frequenzω=^2 TπundOberschwingungen
der Frequenzenkω. Die Koeffizienten geben an, mit welcher Amplitude, d.h. wie stark,
die Oberschwingungen vertreten sind.


Umrechnung der Periode: SeienT,S >0. Istf:R→RT-periodisch, dann ist

```
f ̃(t):=f
```
##### (

##### T

##### S

```
t
```
##### )

S-periodisch, denn:

```
f ̃(t+S) =f
```
##### (

##### T

##### S

```
(t+S)
```
##### )

```
=f
```
##### (

##### T

##### S

```
t+T
```
##### )

```
=f
```
##### (

##### T

##### S

```
t
```
##### )

```
=f ̃(t).
```
### 37.2 Reelle Fourierapproximation

F ̈ur die Bausteine der trigonometrischen Polynome gelten die folgenden Orthogona-
lit ̈atsrelationen.

Satz 37.4 (Orthogonalit ̈atsrelationen). Sei T > 0 undω =^2 Tπ. Dann gilt f ̈ur alle
k,`∈N:

##### 2

##### T

##### ∫T

```
0
```
```
cos(kωt) cos(`ωt)dt=
```
##### 

##### 

##### 

```
2 k=`= 0,
1 k=` > 0
0 k 6 =`
2
T
```
##### ∫T

```
0
```
```
sin(kωt) sin(`ωt)dt=
```
##### {

```
1 k=` > 0
0 sonst
2
T
```
##### ∫T

```
0
```
```
sin(kωt) cos(`ωt)dt= 0.
```
Beweis. Das haben wir bereits in Beispiel 30.3 nachgerechnet. Ebenso folgt die Aussage
aus Beispiel 30.8, wenn man Real- und Imagin ̈arteile der Integrale dort vergleicht.
Wir k ̈onnen das auch noch auf einem dritten Weg mit der komplexen Exponential-
funktion nachrechnen: F ̈urm∈Z,m 6 = 0, ist
∫T

```
0
```
```
eimωtdt=
```
##### 1

```
imω
eimωt
```
##### ∣∣T

##### 0 =

##### 1

```
imω
(eimωT−e^0 ) =
```
##### 1

```
imω
(eim^2 π−1) = 0.
```
Daher rechnet man f ̈urk,`≥0:

```
2
T
```
##### ∫T

```
0
```
```
cos(kωt) cos(`ωt)dt=
```
##### 2

##### T

##### ∫T

```
0
```
```
eikωt+e−ikωt
2
```
##### ·

```
ei`ωt+e−i`ωt
2
dt
```
##### =

##### 1

##### 2 T

##### ∫T

```
0
```
##### (

```
ei(k+`)ωt+ei(k−`)ωt+ei(−k+`)ωt+ei(−k−`)ωt
```
##### )

```
dt
```
F ̈urk 6 =`sindk+`,k−`,−k+`,−k−`alle 6 = 0, so dass das Integral verschwindet. Ist
k=`= 0, so ist der Integrand 4, und wir erhalten^42 TT= 2, wie behauptet. Ist hingegen
k=` >0, so sindk+`,−k−` 6 = 0, aberk−`= 0 =−k+`. Dann ist der Integrand 2,
und das Integral ergibt 1.
Die anderen beiden Formeln findet man genauso, indem man auch den Sinus mit der
komplexen Exponentialfunktion schreibt.


Die Orthogonalit ̈atsrelationen besagen, dass die Funktionen cos(kωt) und sin(`ωt)
orthogonalbezuglich des ̈ L^2 -Skalarprodukts

```
〈f,g〉=
```
##### 2

##### T

##### ∫T

```
0
```
```
f(t)g(t)dt (37.1)
```
sind. Fur ̈ k≥1 sind cos(kωt) und sin(kωt) sogarorthonormal, denn

```
〈cos(kωt),cos(kωt)〉=
```
##### 2

##### T

##### ∫T

```
0
```
```
cos(kωt) cos(kωt)dt= 1,
```
```
〈sin(kωt),sin(kωt)〉=
```
##### 2

##### T

##### ∫T

```
0
```
```
sin(kωt) sin(kωt)dt= 1.
```
Nur f ̈urk= 0 haben wir den Sonderfall cos(0ωt) = 1, f ̈ur den

```
〈cos(0ωt),cos(0ωt)〉=
```
##### 2

##### T

##### ∫T

```
0
```
```
1 dt= 2
```
ist. Dieses Wissen machen wir uns gleich zunutze, um Funktionen durch trigonometrische
Polynome zu approximieren.

St ̈uckweise Monotonie. Im Folgenden haben wir es oft mit Integralen zu tun, an de-
nen eineT-periodische Funktionf:R→Rbeteiligt ist. Damit diese Integrale existieren,
machen wir folgende Generalvoraussetzung.
Generalvoraussetzung:Alle im Zusammenhang mit der Fourieranalysis betrachteten
reellen Funktionen seien auf dem Periodenintervall [0,T] stuckweise monoton (vergleiche ̈
Definition 28.4). Komplexwertige Funktionen sollen st ̈uckweise monotonen Real- und
Imagin ̈arteil haben.
Das impliziert insbesondere, dassfbeschr ̈ankt ist. Beispiele sind nat ̈urlich die Sinus-
und Cosinusfunktionen aber auch S ̈agezahn- oder Rechtecksfunktionen.
Aus Satz 35.13 wissen wir, dass ein Vektor bez ̈uglich einer Orthonormalbasis die
Darstellung

```
v=
```
```
∑n
```
```
j=1
```
```
〈v,uj〉uj
```
hat. Da cos(kωt) und sin(kωt) orthonormal bzgl. desL^2 -Skalarprodukts (37.1) sind,
definieren wir die Approximation durch trigonometrische Polynome wie folgt.

Definition 37.5(Fourierkoeffizienten und Fourierpolynom). Seif :R→R(oderf :
R→C) eine stuckweise monotone Funktion der Periode ̈ T >0 undω=^2 Tπ. Dann heißen

```
ak=〈f,cos(kωt)〉=
```
##### 2

##### T

##### ∫T

```
0
```
```
f(t) cos(kωt)dt,
```
```
bk=〈f,sin(kωt)〉=
```
##### 2

##### T

##### ∫T

```
0
```
```
f(t) sin(kωt)dt,
```

die reellenFourierkoeffizientenvonf, und

```
φn(t) =
a 0
2
```
##### +

```
∑n
```
```
k=1
```
```
(akcos(kωt) +bksin(kωt))
```
heißt dasn-te Fourierpolynomoder dasFourierpolynom der Ordnungnvonf. Manchmal
schreibt man auchφfn(t) oderAhnliches. ̈

Bemerkung 37.6. 1) Die Fourierkoeffizienten sind genau wie die Koeffizienten eines
Vektors bez ̈uglich einer ONB definiert.
2) Der konstante Term ista 0 /2 und nichta 0 , da〈cos(0ωt),cos(0ωt)〉= 2 und nicht 1
ist (s.o.), was hier ausgeglichen wird.
3) Es ist immerb 0 = 0, da sin(0ωt) = 0 ist.

```
4) F ̈ur die Fourierkoeffizienten vonφngilt
2
T
```
##### ∫T

```
0
```
```
φn(t) cos(kωt)dt=ak=
```
##### 2

##### T

##### ∫T

```
0
```
```
f(t) cos(kωt)dt,
```
```
2
T
```
##### ∫T

```
0
```
```
φn(t) sin(kωt)dt=bk=
```
##### 2

##### T

##### ∫T

```
0
```
```
f(t) sin(kωt)dt.
```
Das rechnet man leicht mit den Orthogonalit ̈atsrelationen nach (oder noch einfa-
cher mit dem Skalarprodukt).
Mit Blick auf Satz 35.13 und Satz 36.2 (ONB-Entwicklung und beste Approximation)
hat man die Hoffnung, dass die Fourierpolynome die Funktionf approximieren. Es ist
aber nicht klar, ob das Fourierpolynomn-ter Ordnung (fur hinreichend großes ̈ n) wirklich
eine gute Approximation f ̈urf(t) darstellt. Wie groß ist der Fehler? Wir kommen auf
diese Frage in Abschnitt 38.2 zuruck. Zun ̈ ̈achst betrachten wir ein Beispiel.

Beispiel 37.7.S ̈agezahnkurve

```
t
− 2 − 1 0 1 3
− 1
```
##### 1

Die Funktionf:R→Rist 2-periodisch (d.h.T= 2) mit

```
f(t) =
```
##### {

```
1 −t f ̈ur 0< t < 2
0 f ̈urt= 0.
```
```
Berechnung der Fourierkoeffizienten:DaT= 2 istω=^2 Tπ=π. Dann ist f ̈urk≥1:
```
```
bk=
```
##### 2

##### 2

##### ∫ 2

```
0
```
```
(1−t) sin(kπt)dt=
```
##### ∫ 2

```
0
```
```
sin(kπt)dt−
```
##### ∫ 2

```
0
```
```
tsin(kπt)dt.
```

Das zweite Integral berechnen wir mit partieller Integration:u′(t) = sin(kπt) undv(t) =
t, also

```
bk=−
```
##### 1

```
kπ
cos(kπt)
```
##### ∣

##### ∣^2

##### 0 −

##### (

##### −

```
t
kπ
cos(kπt)
```
##### ∣

##### ∣^2

##### 0 −

##### ∫ 2

```
0
```
##### 1

```
kπ
cos(kπt)dt
```
##### )

##### =−

##### 1

```
kπ
```
##### (1−1)−

##### (

##### −

##### 2

```
kπ
```
##### +

##### 0

```
kπ
```
##### −

##### 1

```
kπ
sin(kπt)
```
##### ∣∣ 2

```
0
```
##### )

##### =

##### 2

```
kπ
```
##### .

Weiter ist

```
a 0 =
```
##### ∫ 2

```
0
```
```
(1−t) cos(0πt)
︸ ︷︷ ︸
=1
```
```
dt=
```
##### (

```
t−
```
##### 1

##### 2

```
t^2
```
##### )∣

##### ∣∣^2

```
0
```
##### = 0

und fur ̈ k≥ 1

```
ak=
```
##### ∫ 2

```
0
```
```
(1−t) cos(kπt)dt=
```
##### ∫ 2

```
0
```
```
cos(kπt)dt
︸ ︷︷ ︸
=0
```
##### −

##### ∫ 2

```
0
```
```
tcos(kπt)dt
```
##### =−

##### (

```
t
```
##### 1

```
kπ
```
```
sin(kπt)
```
##### ∣

##### ∣^2

##### 0 −

##### ∫ 2

```
0
```
##### 1

```
kπ
```
```
sin(kπt)dt
```
##### )

##### = 0.

Damit ist dasn-te Fourierpolynom vonf

```
φn(t) =
```
```
a 0
2
```
##### +

```
∑n
```
```
k=1
```
```
(akcos(kωt) +bksin(kωt)) =
```
```
∑n
```
```
k=1
```
##### 2

```
kπ
sin(kπt).
```
Die Fourierpolynome approximieren die Funktion f ̈ur wachsendesnimmer besser:

##### − 2 0 2 4

##### − 1

##### 0

##### 1

```
1-tes Fourierpolynom
```
##### − 2 0 2 4

##### − 1

##### 0

##### 1

```
5-tes Fourierpolynom
```
##### − 2 0 2 4

##### − 1

##### 0

##### 1

```
10-tes Fourierpolynom
```
##### − 2 0 2 4

##### − 1

##### 0

##### 1

```
50-tes Fourierpolynom
```

```
Frequenzspektrum:
```
```
k
```
```
bk
```
##### 1 2 3 4

##### 1

```
2 /π
k
```
```
ak
```
##### 1 2 3 4

##### 1

Anwendungen. DieFourieranalyseeiner periodischen Funktionf(t), d.h. die Berech-
nung derakundbk, und dieFouriersynthese, das Zusammenfugen der Koeffizienten zu ̈
einem trigonometrischen Polynom, haben viele Anwendungen.

- Eine benutzen Sie st ̈andig: Im menschlichen Ohr sind die Haarzellen des Corti-
    schen Organs jeweils f ̈ur bestimmte Frequenzen empfindlich. Das Ohr ̈ubermittelt
    dem Gehirn also die Fourierkoeffizienten der von ihm aufgenommenen akustischen
    Signale.
- Rauschunterdruckungs- oder Kompressionsverfahren (etwa f ̈ ur MP3) zerlegen Si- ̈
    gnale mit der Fourieranalyse in ihr Frequenzspektrum, filtern die unerwunschten ̈
    oder ̈uberflussigen Frequenzen heraus und setzen das Signal dann wieder zusam- ̈
    men.
- Die Wirkung linearer Systeme etwa in der Regelungstechnik l ̈aßt sich an harmoni-
    schen Schwingungen testen und mit der Fourieranalyse f ̈ur beliebigen periodischen
    Input vorhersagen.
- W ̈ahrend Generatoren naturlicherweise harmonische Spannungen liefern, ist man ̈
    bei technischen Anwendungen zum Beipiel an linearen S ̈agezahnspannungen inter-
    essiert (etwa fur die Zeilensteuerung des Elektronenstrahls in einer Bildr ̈ ̈ohre). Die
    Fourieranalyse liefert Auskunft daruber, wie man durch ̈ Uberlagerung harmoni- ̈
    scher Schwingungen solche”willk ̈urlichen“ Spannungen”synthetisieren“ kann.
- Auf ̈uberraschend andere Weise dient die Fourieranalyse bei Rand-Anfangswert-
    Problemen partieller Differentialgleichungen und damit bei sehr vielen Problemen
    der Verfahrens-, Energie- oder Elektrotechnik als wichtiges Hilfsmittel.



## Vorlesung 38

## 38 Approximation im quadratischen Mittel

# Mittel

In dieser Vorlesung untersuchen wir, wie gut die Fourierpolynomeφneine periodische
Funktionfapproximieren. Hier gibt es zwei wesentliche Aussagen: F ̈ur festesnist das
Fourierpolynom dasjenige trigonometrische Polynom, dasfam Besten in derL^2 -Norm
approximiert, d.h. unter den trigonometrischen Polynomen mit vorgegebener Ordnung
ist das Fourierpolynom am n ̈achsten anf(Bestapproximation; siehe Abschnitt 38.2).
Fur ̈ n→ ∞konvergiert‖f−φn‖L 2 in der L^2 -Norm (dem quadratischen Mittel)
gegen Null. Aber hier gilt noch mehr: Die Fourierpolynome konvergieren nicht nur im
quadratischen Mittel gegenf, sondern wir k ̈onnen sogar genaue Aussagenuber die Funk- ̈
tionswerte in jedem Punkt treffen (Satz 38.6).
Vorweg behandeln wir einige Aussagen, die die Berechnung der Fourierkoeffizienten
im Falle von geraden oder ungeraden Funktionen vereinfachen.

### 38.1 Fourierkoeffizienten f ̈ur gerade und ungerade Funktionen

### tionen

Istfeine gerade oder ungeradeT-periodische Funktion, so lassen sich die Fourierkoef-
fizienten leichter berechnen.
Wir beginnen mit einigen elementaren Beobachtungen:

```
1) F ̈ur alleω∈Rund allek∈Nist
```
- cos(kωt) eine gerade Funktion: cos(kω(−t)) = cos(kωt),
- sin(kωt) eine ungerade Funktion: sin(kω(−t)) =−sin(kωt).

```
2) Seieng:R→Reine gerade Funktion undu:R→Reine ungerade Funktion.
Dann gilt:
```
- Istfgerade, so istfggerade undfuungerade.
- Istfungerade, so istfgungerade undfugerade.


```
Beweis. F ̈ur geradesfsind
```
```
(fg)(−t) =f(−t)g(−t) =f(t)g(t) = (fg)(t),
(fu)(−t) =f(−t)u(−t) =f(t)(−u(t)) =−(fu)(t).
```
```
F ̈ur ungeradesfgeht es analog.
```
3) Seigeine gerade undueine ungerade Funktion. Dann gilt
∫a

```
−a
```
```
u(t)dt= 0,
```
```
∫a
```
```
−a
```
```
g(t)dt= 2
```
```
∫a
```
```
0
```
```
g(t)dt.
```
```
Beweis. F ̈uruist
∫a
```
```
−a
```
```
u(t)dt=
```
##### ∫ 0

```
−a
```
```
u(t)dt+
```
```
∫a
```
```
0
```
```
u(t)dt,
```
```
t
```
```
u
```
```
−a a
```
```
und Substitutions=−tergibt
∫ 0
```
```
−a
```
```
u(t)dt=−
```
##### ∫ 0

```
a
```
```
u(−s)ds=
```
```
∫a
```
```
0
```
```
u(−s)ds
uungerade
= −
```
```
∫a
```
```
0
```
```
u(s)ds,
```
```
woraus die erste Formel folgt. Genauso ist f ̈urg
∫a
```
```
−a
```
```
g(t)dt=
```
##### ∫ 0

```
−a
```
```
g(t)dt+
```
```
∫a
```
```
0
```
```
g(t)dt
```
```
t
```
```
g
```
```
−a 0 a
```
```
und ∫ 0
```
```
−a
```
```
g(t)dt=−
```
##### ∫ 0

```
a
```
```
g(−s)ds=
```
```
∫a
```
```
0
```
```
g(s)ds.
```
```
Das ergibt die zweite Formel.
```

```
4) Istf T-periodisch, so k ̈onnen wir statt ̈uber [0,T]uber ein beliebiges anderes ̈
Intervall der L ̈angeT integrieren. In Zeichen: F ̈ur jedesa∈Rgilt
∫T
```
```
0
```
```
f(t)dt=
```
```
∫a+T
```
```
a
```
```
f(t)dt.
```
```
Speziell f ̈ura=−T 2 ist
∫T
```
```
0
```
```
f(t)dt=
```
##### ∫ T 2

```
−T 2
```
```
f(t)dt.
```
```
t
0 a T a+T
```
```
t
−a=−T 2 0 a=T 2 T
```
Mit diesen Vorbereitungen erhalten wir die folgenden Aussagen ̈uber die Fourier-
koeffizienten von geraden oder ungeraden Funktionen. (Nat ̈urlich gibt es auch andere
Funktionen, wie 1 +t, die weder gerade noch ungerade sind.)

Satz 38.1. Seif:R→ReineT-periodische Funktion undω=^2 Tπ> 0.

```
1) Istfungerade, so gilt f ̈ur allek∈N
```
```
ak= 0, bk=
```
##### 4

##### T

##### ∫ T 2

```
0
```
```
f(t) sin(kωt)dt.
```
```
2) Istfeine gerade Funktion, so gilt f ̈ur allek∈N
```
```
ak=
```
##### 4

##### T

##### ∫ T

```
2
0
```
```
f(t) cos(kωt)dt, bk= 0.
```
Beweis. Nach 4) k ̈onnen wir die Fourierkoeffizienten vonfschreiben als

```
ak=
```
##### 2

##### T

##### ∫ T 2

```
−T 2
```
```
f(t) cos(kωt)dt, bk=
```
##### 2

##### T

##### ∫ T 2

```
−T 2
```
```
f(t) sin(kωt)dt.
```
Nach 1) ist cos(kωt) gerade und sin(kωt) ungerade.
Fur ungerades ̈ fistf(t) cos(kωt) ungerade, alsoak= 0, undf(t) sin(kωt) ist gerade,
also

```
bk=
```
##### 4

##### T

##### ∫ T

```
2
0
```
```
f(t) sin(kωt)dt.
```
Ist hingegenf gerade, so istf(t) sin(kωt) ungerade, alsobk = 0, undf(t) cos(kωt) ist
gerade, also

```
ak=
```
##### 4

##### T

##### ∫ T

```
2
0
```
```
f(t) cos(kωt)dt.
```

Beispiel 38.2(Rechteckspannung).Seif:R→R 2 π-periodisch (T= 2π,ω=^2 Tπ= 1),

```
f(t) =
```
##### 

##### 

##### 

```
−1 fur ̈ −π < t < 0
0 fur ̈ t= 0,π
1 fur 0 ̈ < t < π.
```
```
t
−π 0 π
```
##### 1

##### − 1

Der Funktionswert an den Punkten 0 undπist nicht wesentlich, aber mit dem Wert
0 wird die Funktion ungerade, was die Berechnung der Fourierkoeffizienten vereinfacht:
Fur alle ̈ k∈Nsindak= 0 und

```
bk=
```
##### 4

```
2 π
```
```
∫π
```
```
0
```
```
f(t) sin(kωt)dt=
```
##### 2

```
π
```
```
∫π
```
```
0
```
```
sin(kt)dt=−
```
##### 2

```
π
```
##### 1

```
k
cos(kt)
```
##### ∣

```
∣π
0
```
##### =−

##### 2

```
kπ
((−1)k−1) =
```
##### {

```
0 f ̈urkgerade
4
kπ f ̈urkungerade.
```
Damit ist das (2n+ 1)-te Fourierpolynom

```
φ 2 n+1(t) =
```
(^2) ∑n+1
k=1
bksin(kωt) =

##### 4

```
π
```
```
∑n
```
```
m=0
```
##### 1

```
2 m+ 1
sin((2m+ 1)t)
```
##### =

##### 4

```
π
```
##### (

```
sin(t) +
```
##### 1

##### 3

```
sin(3t) +
```
##### 1

##### 5

```
sin(5t) +...+
```
##### 1

```
2 n+ 1
sin((2n+ 1)t)
```
##### )

##### .

Im Folgenden sind die Fourierpolynome der Ordnungenn= 5, 11 , 21 ,51 geplottet.

##### − 4 − 2 0 2 4

##### − 1

##### 0

##### 1

```
5-tes Fourierpolynom
```
##### − 4 − 2 0 2 4

##### − 1

##### 0

##### 1

```
11-tes Fourierpolynom
```

##### − 4 − 2 0 2 4

##### − 1

##### 0

##### 1

```
21-tes Fourierpolynom
```
##### − 4 − 2 0 2 4

##### − 1

##### 0

##### 1

```
51-tes Fourierpolynom
```
Der”Uberschuss“ an den Sprungstellen ist ein nach Gibbs benanntes typisches Ph ̈ ̈ano-
men bei der Approximation von Sprungfunktionen durch Fourierpolynome. Er betr ̈agt
knapp 9% des Sprungs.

### 38.2 Approximation im quadratischen Mittel

Wir untersuchen, wie gut das Fourierpolynom die gegebene Funktion ann ̈ahert.
Erinnerung: Wenn wir eine GeradeU = span{u}mit‖u‖= 1 in der Ebene haben
und einen Punktv, so ist〈v,u〉uder Punkt der Geraden, der am n ̈achsten anv ist
(gemessen in der zum Skalarprodukt geh ̈orenden Norm). Genauso verh ̈alt es sich wenn
U eine Ebene oder gar ein Teilraum ist (vergleiche Satz 36.1 und Satz 36.2). Nun ist
das Fourierpolynom genau solch eine orthogonale Projektion der Funktionf auf den
Teilraum der trigonometrischen Polynome. Das Skalarprodukt ist dabei

```
〈f,g〉=
```
##### 2

##### T

##### ∫T

```
0
```
```
f(t)g(t)dt (38.1)
```
mit induzierter Norm

```
‖f‖L 2 =
```
##### √

##### 2

##### T

##### ∫T

```
0
```
```
f(t)^2 dt.
```
Diese wird auch alsquadratisches Mittelder Funktion bezeichnet.

Satz 38.3(Approximation im quadratischen Mittel). Seif eine st ̈uckweise monotone
Funktion mit Periode T =^2 ωπ > 0 und den Fourierkoeffizientenak, bk. Unter allen
trigonometrischen Polynomen der Ordnungnliefert dasn-te Fourierpolynom vonf

```
φn(t) =
```
```
a 0
2
```
##### +

```
∑n
```
```
k=1
```
```
(akcos(kωt) +bksin(kωt))
```
die beste Approximation im quadratischen Mittel. F ̈ur dieses ist der”quadratische Feh-
ler“‖f−φn‖^2 L 2 =‖f‖^2 L 2 −‖φn‖^2 L 2 , also

```
2
T
```
##### ∫T

```
0
```
```
(f(t)−φn(t))^2 dt=
```
##### 2

##### T

##### ∫T

```
0
```
```
f(t)^2 dt−
```
##### (

```
a^20
2
```
##### +

```
∑n
```
```
k=1
```
```
(a^2 k+b^2 k)
```
##### )

##### .


Dieser Fehler konvergiert fur ̈ n→ ∞gegen 0 , d.h.f(t)kann im quadratischen Mittel
beliebig gut durch Fourierpolynome approximiert werden.

Beweis. Die Konvergenzaussage beweisen wir nicht, den Anfang des Satzes schon.
Die Eigenschaft der Bestapproximation und die Fehlerformel sind Satz 36.2 (das
Fourierpolynom ist genau die orthogonale Projektion vonfauf den Teilraum der trigo-
nometrischen Polynome, fur den cos( ̈ kωt) und sin(kωt) eine ONB bzgl. (38.1) bilden).
Das kann man auch noch einmal direkt nachrechnen. Um den Schreibaufwand zu reduzieren, rech-
nen wir das nur f ̈ur eine ungerade Funktion nach und suchen unter allen ungeraden trigonometrischen
Polynomen der Ordnungn

```
φ(t) =
```
```
∑n
k=1
```
```
βksin(kωt)
```
dasjenige, fur das ̈

```
‖f−φ‖^2 L 2 =T^2
```
```
∫T
0
```
```
(f(t)−φ(t))^2 dt= min.
```
Wegen der Orthogonalit ̈atsrelationen (Satz 37.4) ist

```
2
T
```
```
∫T
0
```
```
φ(t)^2 dt=
```
```
∑n
k=1
```
```
∑n
`=1
```
```
βkβ`T^2
```
```
∫T
0
```
```
sin(kωt) sin(`ωt)dt=
```
```
∑n
k=1
```
```
β^2 k
```
und, wennbkdie Fourierkoeffizienten vonfbezeichnen,

```
〈f,φ〉=T^2
```
```
∫T
0
```
```
f(t)φ(t)dt=
```
```
∑n
k=1
```
```
βkT^2
```
```
∫T
0
```
```
f(t) sin(kωt)dt=
```
```
∑n
k=1
```
```
βkbk.
```
Wir erhalten

```
‖f−φ‖^2 L 2 =T^2
```
```
∫T
0
```
```
(f(t)−φ(t))^2 dt
```
```
=T^2
```
```
∫T
0
```
f(t)^2 dt− (^2) T^2
∫T
0
f(t)φ(t)dt+T^2
∫T
0
φ(t)^2 dt
=T^2
∫T
0
f(t)^2 dt− 2
∑n
k=1
βkbk+
∑n
k=1
β^2 k
=T^2
∫T
0
f(t)^2 dt−
∑n
k=1
b^2 k
︸ ︷︷ ︸
konstant
+
∑n
k=1
(bk−βk)^2
︸ ︷︷ ︸
≥ 0
Die rechte Seite wird also am kleinsten f ̈urβk=bkf ̈ur allek, d.h. wennφ(t) =φn(t) dasn-te Fourier-
polynom vonfist.

### 38.3 Fourierreihen

Wir wollen nun untersuchen, wie gut die Fourierpolynomeφn(t) fur ̈ n→∞die Funktion
fann ̈ahern.Ahnlich zu Taylorreihen definieren wir Fourierreihen. ̈

Definition 38.4(Fourierreihe). Seif :R→ReineT-periodische Funktion mit den
Fourierkoeffizientenakundbkund seiω=^2 Tπ. Dann heißt

```
a 0
2
```
##### +

##### ∑∞

```
k=1
```
```
(akcos(kωt) +bksin(kωt)):= lim
n→∞
```
##### (

```
a 0
2
```
##### +

```
∑n
```
```
k=1
```
```
(akcos(kωt) +bksin(kωt))
```
##### )


dieFourierreihevonf.

Bemerkung 38.5. 1) Die Fourierreihe vonfentspricht der ONB-Entwicklung (siehe
Satz 35.13), nur mit unendlich vielen Termen.

```
2) Die Fourierpolynome entsprechen der Bestapproximation durch trigonometrische
Polynome (siehe Satz 36.2).
```
Satz 38.3 besagt, dass die Fourierreihe im quadratischen Mittel gegenfkonvergiert:
Der Approximationsfehler

```
‖f−φn‖L 2 =
```
##### √

##### 2

##### T

##### ∫T

```
0
```
```
(f(t)−φn(t))^2 dt
```
konvergiert gegen 0 f ̈urn→∞.
Der folgende Satz gibt Auskunft, wann die Fourierpolynome in einem Punkt (und
nicht nur im quadratischen Mittel) gegen die Funktionfkonvergiert.

Satz 38.6(Konvergenz von Fourierreihen). Die Funktionf:R→RseiT-periodisch
und st ̈uckweise monoton und es seiω=^2 Tπ. Dann gilt:

- An allen Stetigkeitsstellentvonf konvergiert die Fourierreihe vonfgegenf(t).
- An allen Unstetigkeitsstellen existieren wegen der st ̈uckweisen Monotonie der links-
    und rechtsseitige Grenzwertf(t−) = limτ↗tf(τ)undf(t+) = limτ↘tf(τ)vonf,
    und die Fourierreihe konvergiert gegen den Mittelwertf(t−)+ 2 f(t+).

Man hat alsof ̈ur allet:

```
f(t−) +f(t+)
2
```
##### =

```
a 0
2
```
##### +

##### ∑∞

```
k=1
```
```
(akcos(kωt) +bksin(kωt)).
```
Einen Beweis des Satzes in dieser bequemen, aber nicht sehr verbreiteten Version
findet man etwa inMangoldt-Knopp, Einf ̈uhrung in die H ̈ohere Mathematik, Band III.
Wir gehen hier nicht weiter darauf ein.

Beispiel 38.7. 1) Wir betrachten die 2-periodische S ̈agezahn-Funktion

```
f:R→R, f(t) =
```
##### {

```
1 −t f ̈ur 0< t < 2
0 f ̈urt= 0,
```
```
aus Beispiel 37.7. An der Unstetigkeitsstelle 0 sind der links- und rechtsseitige
Grenzwertf(0−) =−1 undf(0+) = 1, so das (f(0−) +f(0+))/2 = 0 =f(0) ist.
Daher wird die Funktionfuberall durch ihre Fourierreihe dargestellt: ̈
```
```
f(t) =
```
##### ∑∞

```
k=1
```
##### 2

```
πk
```
```
sin(kπt) f ̈ur allet∈R.
```

```
2)Ahnlich gilt f ̈ ̈ur die 2π-periodische Rechteckspannung
```
```
g:R→R, g(t) =
```
##### 

##### 

##### 

```
−1 f ̈ur −π < t < 0
0 f ̈urt= 0,π
1 f ̈ur 0< t < π,
```
```
aus Beispiel 38.2, dass
```
```
g(t) =
```
##### 4

```
π
```
##### ∑∞

```
k=0
```
##### 1

```
2 k+ 1
```
```
sin((2k+ 1)t) f ̈ur allet∈R.
```
Als letztes notieren wir noch eine Folgerung aus Satz 38.3. Da der Fehler im quadra-
tischen Mittel gegen 0 konvergiert, erhalten wir die sogenannte Parsevalsche Gleichung.

Satz 38.8(Parsevalsche Gleichung und Besselsche Ungleichung).Die Funktionf:R→
RseiT =^2 ωπ-periodisch und st ̈uckweise monoton mit Fourierkoeffizientenak undbk.
Dann gilt dieParsevalsche Gleichung

```
2
T
```
##### ∫T

```
0
```
```
f(t)^2 dt=
a^20
2
```
##### +

##### ∑∞

```
k=1
```
```
(a^2 k+b^2 k):= limn→∞
```
##### (

```
a^20
2
```
##### +

```
∑n
```
```
k=1
```
```
(a^2 k+b^2 k)
```
##### )

##### .

Durch Abschneiden der Summe erh ̈alt man dieBesselsche Ungleichung

```
‖f‖^2 L 2 =
```
##### 2

##### T

##### ∫T

```
0
```
```
f(t)^2 dt≥
```
```
a^20
2
```
##### +

```
∑n
```
```
k=1
```
```
(a^2 k+b^2 k) =‖φn‖^2 L 2.
```
Mit der Parsevalschen Gleichung lassen sich unter anderem Grenzwerte von Folgen
von Summen (Reihen) berechnen.

Beispiel 38.9. 1) Fur die Funktion ̈ f aus Beispiel 38.7 gilt

```
∑∞
```
```
k=1
```
##### 4

```
π^2 k^2
```
##### =

##### 2

##### 2

##### ∫ 2

```
0
```
```
(1−t)^2 dt=−
```
##### 1

##### 3

```
(1−t)^3
```
##### ∣

##### ∣^2

##### 0 =

##### 2

##### 3

##### ,

```
also ist
lim
n→∞
```
```
∑n
```
```
k=1
```
##### 1

```
k^2
```
##### =

##### ∑∞

```
k=1
```
##### 1

```
k^2
```
##### =

```
π^2
6
```
##### .

```
2) F ̈ur die Funktiongaus Beispiel 38.7 gilt
```
```
16
π^2
```
##### ∑∞

```
k=0
```
##### 1

```
(2k+ 1)^2
```
##### =

##### 2

```
2 π
```
```
∫ 2 π
```
```
0
```
```
g(t)^2 dt= 2,
```
```
und damit ∞
∑
```
```
k=0
```
##### 1

```
(2k+ 1)^2
```
##### =

```
π^2
8
```
##### .


Integration und Differenziation von Fourierreihen. Konvergente Fourierreihen
darf man gliedweise integrieren, d.h. man kann wie bei einer endlichen Summe jeden
Summand einzeln integrieren:

```
∫b
```
```
a
```
##### (

```
a 0
2
```
##### +

##### ∑∞

```
k=1
```
```
(akcos(kωt) +bksin(kωt))
```
##### )

```
dt
```
##### =

```
∫b
```
```
a
```
```
a 0
2
dt+
```
##### ∑∞

```
k=1
```
```
∫b
```
```
a
```
```
(akcos(kωt) +bksin(kωt))dt.
```
Beim differenzieren ist das anders: Man darf Fourierreihen im Allgemeinen nicht gliedwei-
se differenzieren. Dabei machen nicht nur die Sprungstellen Probleme, wie man denken
k ̈onnte. Nach Beispiel 38.7 gilt

```
1 −t= 2
```
##### ∑∞

```
k=1
```
##### 1

```
kπ
```
```
sin(kπt) f ̈ur 0< t < 2.
```
Durch gliedweises differenzieren der Reihe erh ̈alt man die Reihe 2

##### ∑∞

k=1cos(kπt). An der
Stellet= 1, an derf keinen Sprung hat, ist die Reihe 2

##### ∑∞

```
k=1cos(kπ) = 2
```
##### ∑∞

```
k=1(−1)
k
```
divergent.
Wir fassen zusammen:Konvergente Fourierreihen darf man

- gliedweise integrieren,
- im Allgemeinen nicht gliedweise differenzieren.



Vorlesung 39

## 39 Komplexe Fourieranalysis

Wir haben schon gesehen, dass sich Sinus- und Cosinusfunktionen oft bequemer mittels
der Euler-Formel durch
eiφ= cos(φ) +isin(φ)

ersetzen lassen. Es ist also

```
cos(φ) = Re(eiφ) =
```
```
eiφ+e−iφ
2
, sin(φ) = Im(eiφ) =
```
```
eiφ−e−iφ
2 i
```
##### ,

Damit lassen sich trigonometrische Polynome sehr einfach komplex schreiben:

```
φ(t) =
a 0
2
```
##### +

```
∑n
```
```
k=1
```
```
(akcos(kωt) +bksin(kωt))
```
##### =

```
a 0
2
```
##### +

```
∑n
```
```
k=1
```
##### (

```
ak
2
eikωt+
ak
2
e−ikωt+
bk
2 i
eikωt−
bk
2 i
e−ikωt
```
##### )

##### =

```
a 0
︸︷︷︸^2
=:c 0
```
##### +

```
∑n
```
```
k=1
```
##### ((

```
ak
2
−i
```
```
bk
2
```
##### )

##### ︸ ︷︷ ︸

```
=:ck
```
```
eikωt+
```
##### (

```
ak
2
+i
```
```
bk
2
```
##### )

##### ︸ ︷︷ ︸

```
=:c−k
```
```
e−ikωt
```
##### )

##### =

```
∑n
```
```
k=−n
```
```
ckeikωt.
```
Aus Beispiel 30.8 wissen wir, dass die Basisfunktioneneikωtorthonormal sind bzgl.
des (komplexen) Skalarprodukts

```
〈f,g〉=
```
##### 1

##### T

##### ∫T

```
0
```
```
f(t)g(t)dt, (39.1)
```
d.h. f ̈ur allek,`∈Zgilt

```
〈eikωt,ei`ωt〉=
```
##### 1

##### T

##### ∫T

```
0
```
```
eikωte−i`ωtdt=
```
##### {

```
1 f ̈urk=`
0 sonst.
```

Definition 39.1(Komplexe trigonometrische Polynome). SeiT >0 undω=^2 Tπ. Die
komplexwertigen Funktionen der Form

```
∑n
```
```
k=−n
```
```
ckeikωt, ck∈C,
```
heißen komplexetrigonometrische Polynomevom Grad oder von der Ordnungn. Diese
sindT-periodisch.

Sind dieakundbkdie Fourierkoeffizienten einer Funktion mit PeriodeT =^2 ωπ>0,
so erhalten wir f ̈urk≥ 0

```
ck=
ak
2
```
```
−i
bk
2
```
##### =

##### 1

##### T

##### ∫T

```
0
```
```
f(t) cos(kωt)dt−i
```
##### 1

##### T

##### ∫T

```
0
```
```
sin(kωt)dt=
```
##### 1

##### T

##### ∫T

```
0
```
```
f(t)e−ikωtdt.
```
Genauso findet man, dass die Formel auch f ̈urk <0 gilt.

Definition 39.2(komplexe Fourierkoeffizienten und Fourierpolynome). Seif:R→R
(oderf :R→C) eine st ̈uckweise monotone Funktion der PeriodeT >0 und es sei
ω=^2 Tπ. Diekomplexen Fourierkoeffizientenvonfsind

```
ck=〈f,eikωt〉=
```
##### 1

##### T

##### ∫T

```
0
```
```
f(t)e−ikωtdt, k∈Z,
```
und

```
φn(t) =
```
```
∑n
```
```
k=−n
```
```
ckeikωt
```
ist dasn-te komplexe Fourierpolynomvonf.

Bemerkung 39.3. 1) Die Formeln sind ein Stuck weit einfacher als die f ̈ ̈ur das reelle
Fourierpolynom: Der Faktor vor dem Integral ist T^1 stattT^2 , und die Sonderrolle
vona 0 ist verschwunden.

```
2) Analog zu Satz 38.3 ist das komplexe Fourierpolynom vom Gradndasjenige kom-
plexe trigonometrische Polynom vom Gradn, dasfam Besten im quadratischen
Mittel approximiert (jetzt mit dem komplexen Skalarprodukt (39.1)), also in der
Norm
```
```
‖f‖L 2 =
```
##### √

```
〈f,f〉=
```
##### √

##### 1

##### T

##### ∫T

```
0
```
```
|f(t)|^2 dt
```
```
3) Istfreellwertig, so sind die Fourierkoeffizientenakundbkreell und es gilt
```
```
c−k=ck,
```
```
was die Berechnung der komplexen Fourierkoeffizienten vereinfachen kann.
```

Beispiel 39.4. Wir betrachten noch einmal die 2-periodische Funktionf:R→Raus
Beispiel 37.7:

```
f(t) =
```
##### {

```
1 −t f ̈ur 0< t < 2
0 f ̈urt= 0.
```
Es istT= 2, alsoω=^2 Tπ=π, und die komplexen Fourierkoeffizienten sind

```
c 0 =
```
##### 1

##### 2

##### ∫ 2

```
0
```
```
(1−t)ei^0 πtdt=
```
##### 1

##### 2

##### ∫ 2

```
0
```
```
(1−t)dt=
```
##### 1

##### 2

```
(t−
```
##### 1

##### 2

```
t^2 )
```
##### ∣

##### ∣^2

##### 0 = 0

und f ̈urk 6 = 0:

```
ck=
```
##### 1

##### 2

##### ∫ 2

```
0
```
```
(1−t)e−ikπtdt=
```
##### 1

##### 2

##### ∫ 2

```
0
```
```
1 e−ikπtdt
︸ ︷︷ ︸
=0,da orthogonal
```
##### +

##### 1

##### 2

##### ∫ 2

```
0
```
```
teikπtdt.
```
Das erste Integral verschwindet, daei^0 πtundeikπtorthogonal sind (oder explizit aus-
rechnen), beim zweiten Integral integrieren wir partiell: u′(t) = eikπt,v(t) = t, also
u(t) =ikπ^1 eikπtundv′(t) = 1, also

```
ck=
```
##### 1

##### 2

##### 1

```
ikπ
teikπt
```
##### ∣

##### ∣^2

##### 0 −

##### 1

##### 2

##### ∫ 2

```
0
```
##### 1

```
ikπ
eikπtdt
︸ ︷︷ ︸
=0
```
##### =

##### 1

```
ikπ
```
##### .

Damit ist dasn-te Fourierpolynom vonf

```
φn(t) =
```
```
∑n
```
```
k=−n
```
```
ckeikωt=
```
##### 1

```
π
```
```
∑n
```
```
k=−n
k 6 =0
```
##### 1

```
ik
eikωt.
```
Durch Zusammenfassen der Terme mit negativem und positiven Index erhalten wir das
reelle Fourierpolynom (siehe Beispiel 37.7):

```
φn(t) =
```
##### 1

```
π
```
```
∑n
```
```
k=1
```
##### (

##### 1

```
ik
```
```
eikωt+
```
##### 1

```
−ik
```
```
e−ikωt
```
##### )

##### =

##### 2

```
π
```
```
∑n
```
```
k=1
```
```
sin(kωt)
k
```
##### .

Umrechnungsformeln zwischen reellen und komplexen Fourierkoeffizienten.
Seif:R→R(oderf:R→C) eineT-periodische Funktion undω=^2 Tπ.

```
1) Sind die reellen Fourierkoeffizientenak,bkgegeben, so sind
```
```
ck=
```
##### 1

##### 2

```
(ak−ibk), k≥ 1 ,
```
```
c 0 =
```
```
a 0
2
```
##### ,

```
c−k=
```
##### 1

##### 2

```
(ak+ibk), k≥ 1.
```

2) Sind die komplexen Fourierkoeffizientenck,k∈Z, gegeben, so gilt

```
a 0
2
=c 0 ,
ak=ck+c−k, k≥ 1 ,
bk=i(ck−c−k), k≥ 1.
```

Vorlesung 40

## 40 Reihen

Unendliche Reihen sind grob gesprochen Grenzwerte n→ ∞von Summen

∑n
k=0ak.
Reihen sind uns schon ̈ofter begegnet: In Abschnitt 18.4 sowie als Taylorreihen und
Fourierreihen. Wie beginnen mit Reihen, deren Glieder Konstanten sind. Anschließend
betrachten wir noch einmal die Taylorreihen (=Potenzreihen)

##### ∑∞

```
k=0ak(x−x^0 )
k, die
```
Grenzwerte von Polynomen sind.

### 40.1 Konvergenz von Reihen

Wir beginnen mit der Definition einer Reihe und ihrer Konvergenz.

Definition 40.1(Reihe). 1) Aus einer gegebenen Folge (an)n∈Nbilden wir eine neue
Folge von Summen (sn)n∈N:

```
sn=
```
```
∑n
```
```
k=0
```
```
ak=a 0 +a 1 +...+an.
```
```
Die Folge (sn)n∈Nnennen wir eineunendliche Reiheund schreiben auch
```
##### ∑∞

```
k=0ak.
DieakheißenGlieder der Reihe, die Summesnheißt dien-tePartialsummeder
Reihe.
```
```
2) Die Reihekonvergiert(bzw.divergiert), falls die Folge (sn)n∈Nkonvergiert (bzw.
divergiert).
```
```
3) Existiert der Grenzwerts= limn→∞sn, so heißtsderWert(oder dieSumme) der
Reihe und wir schreiben daf ̈urs=
```
##### ∑∞

```
k=0ak.
Beobachtung: Wenn
```
##### ∑∞

```
k=0akkonvergiert, dann konvergiert auch
```
##### ∑∞

k=mak, und es
gilt
∑∞

```
k=0
```
```
ak=a 0 +a 1 +...+am− 1 +
```
##### ∑∞

```
k=m
```
```
ak.
```
D.h. die ersten Summanden spiele keine Rolle f ̈ur die Konvergenz, wohl aber f ̈ur den
Wert der Reihe.


Bemerkung 40.2.Die Bezeichnung

##### ∑∞

```
k=0akhat zwei Bedeutungen:
1) Alleinstehend bezeichnet
```
##### ∑∞

```
k=0akdie Folge (sn)n∈N.
2) In einer Gleichung
```
##### ∑∞

```
k=0ak=sbezeichnet
```
##### ∑∞

```
k=0akden Grenzwerts= limk→∞sk.
```
∑StattNbetrachtet man auch andere Summationsbereiche, zum BeispielN\{^0 }in
∞
k=1

```
1
k.
```
Beispiel 40.3(Die geometrische Reihe). Seiq∈Rundak=qk. Die zugeh ̈orige Reihe

```
∑∞
```
```
k=0
```
```
qk= 1 +q+q^2 +q^3 +...
```
heißt diegeometrische Reihe. Sie ist die wichtigste Reihe ̈uberhaupt. In Beispiel 18.12
hatten wir bereits die Konvergenz der geometrischen Reihe untersucht: F ̈ur|q|<1 ist
die geometrische Reihe konvergent und hat den Wert

```
∑∞
```
```
k=0
```
```
qk=
```
##### 1

```
1 −q
f ̈ur|q|< 1.
```
Fur ̈ |q|≥1 ist die geometrische Reihe divergent.

Beispiel 40.4. 1) Achilles und die Schildkr ̈ote: Achilles ist 100 m von der Schildkr ̈ote
entfernt und ist 10 mal so schnell wie diese. Wann hat Achilles die Schildkr ̈ote
eingeholt?
W ̈ahrend Achilles die 100 m zur ̈ucklegt, l ̈auft die Schildkr ̈ote 10 m, so dass der
Abstand nun 10 m betr ̈agt. L ̈auft Achilles diese weiteren 10 m, l ̈auft die Schildkr ̈ote
1 m, usw. Daher holt Achilles die Schildkr ̈ote ein nach

##### 100 + 10 + 1 +

##### 1

##### 10

##### +

##### 1

##### 100

##### +...= 110 +

##### ∑∞

```
k=0
```
##### (

##### 1

##### 10

```
)k
= 110 +
```
##### 1

##### 1 − 101

##### = 110 +

##### 10

##### 9

```
= 111,1 m.
```
```
2) Wir zeigen 0,9 = 1. Mit der geometrischen Reihe ist
```
##### 0 ,9 = 0, 999 ...=

##### ∑∞

```
k=1
```
##### 9 ·

##### (

##### 1

##### 10

```
)k
=
```
##### 9

##### 10

##### ∑∞

```
k=0
```
##### (

##### 1

##### 10

```
)k
=
```
##### 9

##### 10

##### 1

##### 1 − 101

##### = 1.

Aus den Grenzwerts ̈atzen f ̈ur Folgen (Satz 18.1), erhalten wir sofort Rechenregeln
fur konvergente Reihen. ̈

Satz 40.5(Rechenregeln fur konvergente Reihen) ̈ .Sind

##### ∑∞

```
k=0akund
```
##### ∑∞

k=0bkkonver-
gente Reihen undc∈R, so konvergieren auch

##### ∑∞

```
k=0(ak+bk)und
```
##### ∑∞

k=0cak und es
gilt
∑∞

```
k=0
```
```
(ak+bk) =
```
##### ∑∞

```
k=0
```
```
ak+
```
##### ∑∞

```
k=0
```
```
bk und
```
##### ∑∞

```
k=0
```
```
cak=c
```
##### ∑∞

```
k=0
```
```
ak.
```

Mit Produkten ist es nicht ganz so einfach, weil man jedes Glied der einen Reihe mit
jedem Glied der anderen Reihe multiplizieren muss, was bei unendlich vielen Gliedern
Probleme machen kann. Eine sinnvolle Anordnung der Produkte gibt die sogenannte
Produktformel von Cauchy:
(∞
∑

```
k=0
```
```
ak
```
##### )

##### ·

##### (∞

##### ∑

```
k=0
```
```
bk
```
##### )

##### =

##### ∑∞

```
k=0
```
```
ck, wobeick=a 0 bk+a 1 bk− 1 +a 2 bk− 2 +...+akb 0.
```
Diese Anordnung kann man sich wie folgt∑ uberlegen und merken: Multipliziert man ̈
∞
k=0akx

```
k und∑∞
k=0bkx
```
```
k und sortiert anschließend nach Potenzen vonx, so erh ̈alt
```
man (∞
∑

```
k=0
```
```
akxk
```
##### )

##### ·

##### (∞

##### ∑

```
k=0
```
```
bkxk
```
##### )

##### =

##### ∑∞

```
k=0
```
```
ckxk
```
mit obigen Koeffiientenck. Fur ̈ x= 1 hat man dann die Formel des Cauchy-Produkts.
Die Gleichung im Cauchy-Produkt stimmt, wenn alle drei Reihen konvergieren. Es
kann aber sein, dass

##### ∑∞

```
k=0akund
```
##### ∑∞

```
k=0bkkonvergieren, ohne dass
```
##### ∑∞

k=0ckkonvergiert.
Um aus der Konvergenz der Reihen

##### ∑∞

```
k=0akund
```
##### ∑∞

∑∞ k=0bkauch die Konvergenz der Reihe
k=0ckzu erhalten, brauchen wir einen besseren Konvergenzbegriff, den der absoluten
Konvergenz (siehe Definition 41.1).

### 40.2 Konvergenzkriterien

Konvergenzkriterien f ̈ur Reihen sind Tests, ob eine Reihe konvergiert. Dabei sagen sie
oft nur aus, ob eine Reihe konvergiert oder nicht, ohne etwasuber den Wert der Reihe ̈
zu sagen. In vielen F ̈allen reicht das aber bereits.

Satz 40.6 (Notwendiges Kriterium). Wenn die Reihe

##### ∑∞

k=0ak konvergiert, dann gilt
limk→∞ak= 0.

Beweis. Wenn die Reihe konvergiert, so konvergieren die Partialsummensn=

∑n
k=0ak
gegen eine reelle Zahls, d.h. es ists= limn→∞sn. Dann gilt aber auchs= limn→∞sn− 1 ,
und daher

```
0 =s−s= lim
n→∞
sn− lim
n→∞
sn− 1 = lim
n→∞
(sn−sn− 1 ) = lim
n→∞
an.
```
D.h., wenn limk→∞ak 6 = 0 ist (oder die Folge (ak)k gar nicht konvergiert), dann
konvergiert auch die Reihe nicht. Das notwendige Kriterium ist also ein Test, ob die
Reihe divergiert.
Ein weiteres sehr nutzliches Kriterium ist das folgende Integralvergleichskriterium. ̈

Satz 40.7 (Integralvergleichskriterium). Seif : [m,∞[→R, m∈N, eine monoton
fallende Funktion mitf(x)≥ 0 f ̈ur allex∈[m,∞[. Dann gilt:

```
∑∞
```
```
k=m
```
```
f(k)konvergiert ⇔
```
##### ∫∞

```
m
```
```
f(x)dxexistiert.
```

Beweis. Die folgende Skizze zeigt, warum es funktioniert:

```
t
```
```
f(m)
```
```
m m+ 1 m+ 2 m+ 3...
```
##### ∑∞

```
k=m
```
```
f(k)
```
##### ∑∞

```
k=m+1
```
```
f(k)
```
##### ∫∞

```
m
```
```
f(x)dx
```
Eine ausfuhrlichere Begr ̈ undung ist wie folgt: Da ̈ fmonoton fallend ist, gilt

```
f(k+ 1)≤f(x)≤f(k) f ̈ur allex∈[k,k+ 1].
```
Integrieren wir vonkbisk+ 1, so folgt mit der Monotonie des Integrals (Satz 28.8)

```
f(k+ 1) =
```
```
∫k+1
```
```
k
```
```
f(k+ 1)dx≤
```
```
∫k+1
```
```
k
```
```
f(x)dx≤
```
```
∫k+1
```
```
k
```
```
f(k)dx=f(k).
```
Summieren wir dieses auf, folgt

```
∑n
```
```
k=m
```
```
f(k+ 1)≤
```
```
∑n
```
```
k=m
```
```
∫k+1
```
```
k
```
```
f(x)≤
```
```
∑n
```
```
k=m
```
```
f(k),
```
also
n∑+1

```
k=m+1
```
```
f(k)≤
```
```
∫n+1
```
```
m
```
```
f(x)dx≤
```
```
∑n
```
```
k=m
```
```
f(k).
```
```
Wir nehmen nun an, dass die Reihe konvergiert mit Summes= limn→∞
```
∑n
k=mf(k).
Dafpositiv ist, ist die Folge der Partialsummen monoton wachsend und daher

```
∫n+1
```
```
m
```
```
f(x)dx≤
```
```
∑n
```
```
k=m
```
```
f(k)≤s.
```
Daf(x)≥0 f ̈ur allexist, ist auch die Folge (

∫n+1
m f(x)dx)nmonoton wachsend und nach
oben durchsbeschr ̈ankt, also konvergent. Daher existiert das uneigentliche Integral.
Nun zur Ruckrichtung. Wir nehmen an, dass das uneigentliche Integral existiert. Da ̈
f(x)≥0 ist, ist die Folge (

```
∫n+1
m f(x)dx)nmonoton wachsend, also gilt insbesondere
∫n+1
```
```
m
```
```
f(x)dx≤
```
##### ∫∞

```
m
```
```
f(x)dx=s.
```
Dann ist

∑n+1
k=m+1f(k)≤sfur alle ̈ n, und auch diese Folge ist monoton wachsend und
beschr ̈ankt, also konvergent. Daher konvergiert auch die Reihe.


Beispiel 40.8. 1) Dieharmonische Reihe

##### ∑∞

```
k=1
1
kist divergent, denn: Die Funktion
f: [1,∞[→R,f(x) =^1 x, ist monoton fallend mitf(x)≥0 f ̈ur allex∈[1,∞[, und
das Integral ∫
∞
1
```
```
f(x)dx=
```
##### ∫∞

```
1
```
##### 1

```
x
```
```
dx
```
```
ist divergent (Beispiel 31.2).
```
```
2) Die Reihe
```
##### ∑∞

```
k=1
```
```
1
k^2 ist konvergent, da
```
##### ∫∞

```
1
```
```
1
x^2 dxkonvergiert (Beispiel 31.2).
3) Allgemein gilt: Die Reihe
```
##### ∑∞

```
k=1
1
kα ist konvergent f ̈urα >1 und divergent f ̈ur
0 < α≤1, da die entsprechende Aussage f ̈ur die uneigentlichen Integrale
```
##### ∫∞

```
1
```
```
1
xαdx
gilt (siehe Beispiel 31.2).
Bei den Integralen konnten wir den Wert einfach ausrechnen, bei den Reihen ist
das leider nicht so.
```
Ist die aufsummierte Folge nicht monoton fallend (wie beim Integralvergleichskriteri-
um), sondern wechselt immer ihr Vorzeichen, so kann das Leibnizkriterium weiterhelfen,
das wir ohne Beweis angeben.

Satz 40.9(Leibnizkriterium).Ist die Folge(ak)kstreng monoton fallend und gegen Null
konvergent,limk→∞ak= 0, so konvergiert

##### ∑∞

```
k=0(−1)
```
```
kak.
```
Die Idee ist, dass die Partialsummen mit abnehmender Amplitude”oszillieren“, da
aufeinander folgende Glieder der Reihe verschiedene Vorzeichen haben und das n ̈achste
Gliedak+1kleineren Betrag als das vorangehende Gliedakhat:

```
n
```
```
sn
```
##### 1 2 3 4 5 6 7 8 9 10

Beispiel 40.10. Diealternierende harmonische Reihe

##### ∑∞

```
k=1
```
(−1)k
k ist nach dem Leib-
nizkriterium konvergent, denn: ak =^1 k ist eine streng monoton fallende Folge mit
limk→∞^1 k= 0.



Vorlesung 41

## 41 Absolut konvergente Reihen

Wir lernen einen st ̈arkeren Konvergenzbegriff f ̈ur Reihen kennen, die absolute Konver-
genz.

### 41.1 Absolute Konvergenz

Oft ist der folgende st ̈arkere Konvergenzbegriff hilfreich und wichtig.

Definition 41.1(absolute Konvergenz).Die Reihe

##### ∑∞

k=0akheißtabsolut konvergent,
falls die Reihe

##### ∑∞

k=0|ak|konvergent ist.
Absolute Konvergenz ist besser als gew ̈ohnliche Konvergenz, denn es gilt: Konvergiert
eine Reihe absolut, so konvergiert sie auch im gew ̈ohnlichen Sinne. Kurz:

```
∑∞
```
```
k=0
```
```
|ak|konvergent⇒
```
##### ∑∞

```
k=0
```
```
akkonvergent.
```
Andersherum ist nicht richtig: Es gibt konvergente Reihen, die nicht absolut konvergie-
ren. Ein Beispiel ist die alternierende harmonische Reihe, die nach dem Leibniz-Kriterium
konvergiert, aber

##### ∑∞

```
k=1
```
1
k ist die harmonische Reihe, die divergiert.
Mit absolut konvergenten Reihen kann man meist so rechnen wie mit endlichen Sum-
men. Insbesondere k ̈onnen wir sie wie folgt multiplizieren.

Satz 41.2(Cauchy-Produkt).Sind die Reihen

##### ∑∞

```
k=0akund
```
##### ∑∞

k=0bkbeide absolut kon-
vergent, so ist auch ihr Produkt (dasCauchy-Produkt) absolut konvergent:
(∞
∑

```
k=0
```
```
ak
```
##### )

##### ·

##### (∞

##### ∑

```
k=0
```
```
bk
```
##### )

##### =

##### ∑∞

```
k=0
```
```
ck, wobeick=a 0 bk+a 1 bk− 1 +...+akb 0.
```
### 41.2 Konvergenzkriterien f ̈ur absolute Konvergenz

Wir sammeln die wichtigsten Konvergenzkriterien f ̈ur absolut konvergente Reihen. In
der Literatur finden Sie zahlreiche weitere.


Satz 41.3(Majorantenkriterium).Die Reihe

##### ∑∞

```
k=0bksei konvergent, und es gelte
```
```
|ak|≤bk f ̈ur allek.
```
(Es gen ̈ugt auch, wenn|ak|≤bkf ̈ur allek≥k 0 ab einem Startwertk 0 ∈Ngilt.) Dann
ist die Reihe ∞
∑

```
k=0
```
```
ak
```
absolut konvergent.

Beweis. Wegenbk≥|ak|≥0 sind allebk≥0 und die Folge der Partialsummen

∑n
k=0bk
ist monoton wachsend. Insbesondere gilt

```
∑n
```
```
k=0
```
```
|ak|≤
```
```
∑n
```
```
k=0
```
```
bk≤
```
##### ∑∞

```
k=0
```
```
bk=:M.
```
Daher ist die Folge der Partialsummen

∑n
k=0|ak|durchM beschr ̈ankt und monoton
wachsend (da|ak| ≥0), also konvergent. Damit konvergiert

##### ∑∞

```
k=0|ak|, d.h.
```
##### ∑∞

k=0akist
absolut konvergent.

Beispiel 41.4.Wir vergleichen die Reihen

```
∑∞
```
```
k=1
```
##### 1

```
k^2
```
```
und
```
##### ∑∞

```
k=2
```
##### (

##### 1

```
k− 1
```
##### −

##### 1

```
k
```
##### )

##### .

Fur ̈ k≥2 ist ∣
∣
∣∣^1
k^2

##### ∣∣

##### ∣∣=^1

```
k^2
```
##### <

##### 1

```
k(k−1)
```
##### =

##### 1

```
k− 1
```
##### −

##### 1

```
k
```
##### .

Man kann hier direkt nachrechnen, dass die zweite Reihe konvergiert:

∑n

k=2

##### (

##### 1

```
k− 1
```
##### −

##### 1

```
k
```
##### )

##### =

##### (

##### 1 −

##### 1

##### 2

##### )

##### +

##### (

##### 1

##### 2

##### −

##### 1

##### 3

##### )

##### +

##### (

##### 1

##### 3

##### −

##### 1

##### 4

##### )

##### +...+

##### (

##### 1

```
n− 1
```
##### −

##### 1

```
n
```
##### )

##### = 1−

##### 1

```
n
```
##### → 1.

Dies zeigt noch einmal die Konvergenz von

##### ∑∞

k=1
1
k^2 , siehe Beispiel 40.8. Den Wert der
Reihe direkt zu bestimmen ist schwieriger, aber mit Hilfe der Fourier-Analysis haben
wir schon in Beispiel 38.9 nachgerechnet, dass

##### ∑∞

```
k=1
1
k^2 =
```
```
π^2
6.
```
Satz 41.5(Minorantenkriterium).Gilt 0 ≤bk≤akund ist die Reihe

##### ∑∞

k=0bkbestimmt
divergent, so ist auch die Reihe

##### ∑∞

```
k=0akbestimmt divergent.
```
Beweis. Aus 0≤bk≤akfolgt
∑n

```
k=0
```
```
bk≤
```
```
∑n
```
```
k=0
```
```
ak.
```
Da die erste Summe ̈uber alle Schranken w ̈achst, tut das auch die zweite.


Der Nachteil beim Majorantenkriterium ist, dass man schon eine konvergente Majo-
rante

##### ∑

bkhaben muss. Das folgende Kriterium benutzt nur die zu untersuchende Reihe
und ist deshalb meistens dieerste Wahl, wenn man eine Reihe auf Konvergenz un-
tersuchen will. Erst wenn das Quotientenkriterium keine Auskunftuber die Konvergenz ̈
gibt, versucht man andere Kriterien.

Satz 41.6(Quotientenkriterium).Gegeben sei die Reihe

##### ∑∞

k=0ak. Falls der Grenzwert
r= limk→∞

##### ∣

##### ∣

##### ∣

```
ak+1
ak
```
##### ∣

##### ∣

```
∣existiert, so gilt:
```
- Istr < 1 , so konvergiert die Reihe absolut.
- Istr > 1 , so divergiert die Reihe.
- Istr= 1, so ist alles m ̈oglich und das Kriterium trifft keine Aussage.

Beweis. Sei zuerstr= limk→∞

##### ∣

```
∣∣ak+1
ak
```
##### ∣

```
∣∣< 1. Wir w ̈ahlen einq mitr < q <1. Nach
```
Definition der Konvergenz ist dann
∣
∣
∣∣ak+1
ak

##### ∣

##### ∣

```
∣∣< q
```
f ̈ur allekab einem gewissen Wertk 0. Daher ist f ̈urk≥k 0 :

```
|ak|< q|ak− 1 |< q^2 |ak− 2 |< ... < qk−k^0 |ak 0 |=qk
|ak 0 |
qk^0
```
##### .

Mit der geometrischen Reihe rechnen wir

```
∑∞
```
```
k=0
```
```
qk
|ak 0 |
qk^0
```
##### =

```
|ak 0 |
qk^0
```
##### ∑∞

```
k=0
```
```
qk=
|ak 0 |
qk^0
```
##### 1

```
1 −q
```
##### ,

d.h. wir haben eine konvergente Majorante fur ̈

##### ∑∞

k=0|ak|, und das Majorantenkriterium
(mitbk=qk
|ak 0 |
qk^0 ) zeigt die absolute Konvergenz von

##### ∑∞

```
k=0ak.
Istr= limk→∞
```
##### ∣∣

```
∣aka+1k
```
##### ∣∣

```
∣>1, so gilt f ̈ur großek, dass
```
##### ∣∣

```
∣aka+1k
```
##### ∣∣

```
∣>1, also dass|ak+1|>
```
|ak|ist. Dann ist aber (ak)k∈Nkeine Nullfolge, so dass die Reihe

##### ∑∞

k=0ak divergiert
(Satz 40.6).

Bemerkung 41.7. Im Beweis haben wir nicht wirklich den Grenzwert limk→∞

##### ∣∣

```
∣aka+1k
```
##### ∣∣

##### ∣

verwendet, es reicht, dass

##### ∣

##### ∣

##### ∣

```
ak+1
ak
```
##### ∣

##### ∣

∣≤q <1 f ̈ur allek≥ k 0 gilt. Wir haben damit die
folgende bessere Aussage:

```
Gibt es einq < 1 und eink 0 , so dass
```
##### ∣

```
∣∣ak+1
ak
```
##### ∣

∣∣≤qf ̈ur allek≥k
∑^0 , so ist die Reihe
∞
k=0akabsolut konvergent.


Beispiel 41.8. 1) Die Reihe
∑∞

```
k=0
```
```
k
2 k
```
```
ist absolut konvergent nach dem Quotientenkriterium, denn:
∣
∣∣
∣
```
```
ak+1
ak
```
##### ∣

##### ∣∣

##### ∣=

```
k+1
2 k+1
k
2 k
```
##### =

```
k+ 1
2 k+1
```
```
2 k
k
```
##### =

##### 1

##### 2

##### (

##### 1 +

##### 1

```
k
```
##### )

##### →

##### 1

##### 2

##### < 1.

```
2) Bei den Reihen
∑∞
```
```
k=1
```
##### 1

```
k
```
```
und
```
##### ∑∞

```
k=1
```
##### 1

```
k^2
```
```
liefert das Quotientenkriterium beide Male keine Aussage, denn:
```
```
lim
k→∞
```
##### ∣∣

##### ∣∣

##### ∣

```
1
k+1
1
k
```
##### ∣∣

##### ∣∣

##### ∣

```
= lim
k→∞
```
##### ∣∣

##### ∣

##### ∣

```
k
k+ 1
```
##### ∣∣

##### ∣

```
∣= 1 und klim→∞
```
##### ∣∣

##### ∣∣

##### ∣

```
1
(k+1)^2
1
k^2
```
##### ∣∣

##### ∣∣

##### ∣

```
= lim
k→∞
```
##### ∣∣

##### ∣

##### ∣

```
k
k+ 1
```
##### ∣∣

##### ∣

##### ∣

```
2
= 1.
```
```
Wir wissen aber bereits, dass die erste Reihe divergiert (die harmonische Reihe),
und die zweite Reihe konvergiert. Das zeigt noch einmal: Ist der Grenzwert im
Quotientenkriterium 1, so kann alles passieren.
```
### 41.3 Komplexe Reihen

Fur Reihen mit komplexen Gliedern definiert man die Konvergenz und absolute Kon- ̈
vergenz genauso wie f ̈ur Reihen mit reellen Gliedern. F ̈ur komplexe Reihen gilt eben-
falls: Aus absoluter Konvergenz folgt Konvergenz der Reihe. Das notwendige Kriterium
(Satz 40.6), sowie das Majoranten-, Minoranten-, und Quotientenkriterium gelten genau-
so fur komplexe Reihen. Das wichtigste Beispiel ist auch hier die geometrische Reihe: Es ̈
gilt
∑∞

```
k=0
```
```
zk=
```
##### 1

```
1 −z
```
```
f ̈ur allez∈Cmit|z|< 1.
```

Vorlesung 42

## 42 Potenzreihen

Die wichtigsten Funktionen lassen sich als Potenzreihen (=Taylorreihen) darstellen, d.h.
als Grenzwerten→ ∞von Polynomen

```
∑n
k=0ak(z−z^0 )
k. Potenzreihen k ̈onnen damit
```
als Verallgemeinerung von Polynomen angesehen werden. Solche Grenzwerte von Poly-
nomen haben wir bereits bei Taylorreihen in Abschnitt 25.4 gesehen. Nun wollen wir die
Konvergenz und weitere Eigenschaften von Potenzreihen n ̈aher untersuchen.

### 42.1 Konvergenz von Potenzreihen

Definition 42.1(Potenzreihe).EinePotenzreiheist eine unendliche Reihe der Form

```
∑∞
```
```
k=0
```
```
ak(z−z 0 )k.
```
Dieakheißen dieKoeffizientender Potenzreihe, undz 0 heißt derEntwicklungspunktder
Potenzreihe.

Wir wollen Potenzreihen, wie schon Polynome, gleich im Komplexen betrachten.
Das ist nicht komplizierter, aber die Begriffe”Konvergenzradius“ und”Konvergenzkreis“
werden viel anschaulicher. Wir nehmen deshalb an, dass die Koeffizientenakund der
Entwicklungspunkt z 0 reelle oder komplexe Zahlen sind. Weiter ist zeine komplexe
Variable.
Auf der MengeD⊆Callerz, fur die die Reihe konvergiert, liefert ̈

```
f(z):=
```
##### ∑∞

```
k=0
```
```
ak(z−z 0 )k
```
eine Funktionen
f:D→C.

Wir untersuchen die Frage, f ̈ur welche Werte vonzdie Reihe konvergiert. Dazu versuchen
wir das Quotientenkriterium: Nun sind die Glieder der Reiheak(z−z 0 )k, also betrachten


wir den Quotienten ∣
∣∣
∣

```
ak+1(z−z 0 )k+1
ak(z−z 0 )k
```
##### ∣

##### ∣∣

##### ∣=

##### ∣

##### ∣∣

##### ∣

```
ak+1
ak
```
##### ∣

##### ∣∣

```
∣·|z−z^0 |.
```
Wir nehmen an, dass der GrenzwertA= limk→∞

##### ∣∣

```
∣aka+1k
```
##### ∣∣

∣existiert. (Das muss nicht sein,
und in diesem Fall kommen wir mit dem Quotientenkriterium nicht weiter.) Dann ist
die Potenzreihe fur ̈ A|z−z 0 |<1 absolut konvergent und f ̈urA|z−z 0 |>1 divergent.
BeiA= 0 ist die Reihe also f ̈ur allez∈Cabsolut konvergent, bei 0< A <∞ist die
Reihe f ̈ur allezmit

```
|z−z 0 |< R=
```
##### 1

##### A

```
absolut konvergent,
```
```
|z−z 0 |> R=
```
##### 1

##### A

```
divergent.
```
IstA=∞, so ist die Reihe nur fur ̈ z=z 0 konvergent. Wir fassen zusammen.

Satz 42.2(Konvergenz von Potenzreihen). Die Potenzreihe

```
∑∞
```
```
k=0
```
```
ak(z−z 0 )k
```
ist

- f ̈ur allezin einem offenen Kreis (demKonvergenzkreis) vom RadiusRum den
    Mittelpunktz 0 absolut konvergent (also f ̈ur allezmit|z−z 0 |< R) und
- f ̈ur allezaußerhalb des abgeschlossenen Kreises divergent (also divergent f ̈ur alle
    zmit|z−z 0 |> R).
- Das Konvergenzverhalten auf dem Rand des Kreises muss man bei jeder speziellen
    Reihe in jedem Punkt einzeln untersuchen (also f ̈ur allezmit|z−z 0 |=R).
F ̈ur den sogenanntenKonvergenzradiusRgilt

```
R= lim
k→∞
```
##### ∣

##### ∣∣

##### ∣

```
ak
ak+1
```
##### ∣

##### ∣∣

##### ∣, (42.1)

falls der Grenzwert existiert. Dabei ist auch der WertR=∞zugelassen, ein Kreis mit
unendlichem Radius ist die ganze EbeneC.

Beachten Sie, dass der Quotient derakhier genau das Reziproke von der Formel im
Quotientenkriterium ist. Beachten Sie weiter, dass wir die obige Aussage nur unter der
Annahme nachgerechnet haben, dass der Grenzwert existiert. Sie bleibt aber auch dann
richtig, wenn der Grenzwert nicht existiert, nur hat man dann keine so einfache Formel
fur den Konvergenzradius ̈ R.

```
Im Reellen
```
```
x 0 −R x^0 x 0 +R
```
```
divergent? konvergent? divergent
```

```
Im Komplexen
```
```
z 0
```
```
absolut konvergent divergent
```
##### ?

Ein Wort zur Sprache: Sagen Sie nicht, die Reihe sei innerhalb des Konvergenzradi-
us konvergent. Der Konvergenzradius ist eine Zahl, z.B. 7. Was soll es bedeuten, dass
die Reihe innerhalb von 7 konvergiert? Sagen Sie besser, dass die Reihe innerhalb des
Konvergenzkreiseskonvergiert.

Beispiel 42.3. Die geometrische Reihe

##### ∑∞

```
k=0z
kist eine Potenzreihe (alleak= 1) mit
```
KonvergenzradiusR= limk→∞|ak+1/ak|= 1. Es gilt

```
f(z) =
```
##### ∑∞

```
k=0
```
```
zk=
```
##### 1

```
1 −z
f ̈ur|z|< 1.
```
F ̈ur|z|>1 ist die Reihe divergent. Hier k ̈onnen wir auch die Punkte auf dem Rand des
Konvergenzkreises untersuchen: F ̈ur|z|= 1 ist n ̈amlich|zk|= 1 keine Nullfolge, also ist
die geometrische Reihe fur ̈ |z|= 1 divergent nach Satz 40.6.

Beispiel 42.4.Die Potenzreihe

```
∑∞
```
```
k=0
```
```
5 k
k+ 1
z^2 k+1
```
enth ̈alt nur ungerade Potenzen von z, es ist ak = 0 fur alle geraden ̈ k. Darum ist
a 2 k/a 2 k− 1 = 0 unda 2 k+1/a 2 knicht definiert. Der Grenzwert (42.1) zur Berechnung des
Konvergenzradius existiert daher nicht, und wir mussen uns anders helfen. ̈
Eine M ̈oglichkeit ist, das das Quotientenkriterium direkt zu versuchen:

```
lim
k→∞
```
##### ∣

##### ∣∣

##### ∣

##### ∣

```
5 k+1
k+2z
2 k+3
5 k
k+1z
2 k+1
```
##### ∣

##### ∣∣

##### ∣

##### ∣

```
= lim
k→∞
```
##### ∣∣

```
∣∣k+ 1
k+ 2
```
```
5 z^2
```
##### ∣∣

```
∣∣= 5|z|^2.
```
Also hat man absolute Konvergenz f ̈ur |z| < √^15 und Divergenz f ̈ur|z| > √^15. Der

Konvergenzradius istR=√^15.


```
Eine andere M ̈oglichkeit ist, wie folgt umzuformen:
∑∞
```
```
k=0
```
```
5 k
k+ 1
```
```
z^2 k+1=z
```
##### ∑∞

```
k=0
```
```
5 k
k+ 1
```
```
(z^2 )k.
```
Die Potenzreihe

##### ∑∞

```
k=0
5 k
k+1w
```
```
khat den Konvergenzradius
```
```
R= lim
k→∞
```
##### ∣

##### ∣∣

##### ∣∣

```
5 k
k+1
5 k+1
k+2
```
##### ∣

##### ∣∣

##### ∣∣=

##### 1

##### 5

und konvergiert somit f ̈ur|w|<^15. Die Originalreihe konvergiert daher f ̈ur|z^2 |<^15 , also
fur ̈

```
|z|<
```
##### 1

##### √

##### 5

### 42.2 Ableitung von Potenzreihen

Als n ̈achsten betrachten wir die Ableitung von Potenzreihen. Da wir die Ableitung nur
fur Funktionen einer reellen Variablen erkl ̈ ̈art haben, beschr ̈anken wir uns auf reelle
Reihen. Im Komplexen geht es ganz genau so, mehr dazu in der”Analysis III f ̈ur Inge-
nieurwissenschaften“.

Satz 42.5(Differentiation von Potenzreihen).Die reelle Potenzreihe

```
∑∞
```
```
k=0
```
```
ak(x−x 0 )k
```
habe den KonverganzradiusR > 0. Sie konvergiert also auf dem Intervall]x 0 −R,x 0 +R[
und definiert dort eine Funktionf(x). Diese Funktion ist differenzierbar, und es gilt

```
f′(x) =
```
##### ∑∞

```
k=1
```
```
kak(x−x 0 )k−^1.
```
Der Konvergenzradius der abgeleiteten Reihe ist wiederR.
D.h.: Potenzreihen darf man gliedweise differenzieren. Der Konvergenzradius bleibt
gleich. Ebenso darf man Potenzreihen gliedweise integrieren.

Bei endlichen Summen (also bei Polynomen) ist das ganz klar. Das man bei Reihen
die Grenzwerte der Reihe und des Ableitens (bzw. Integrierens) vertauschen darf ist
hingegen nicht so einfach zu sehen, und wir verzichten auf den Beweis.

Beispiel 42.6. Die Reihe

##### ∑∞

```
k=0
1
k+1x
```
```
k+1hat den Konvergenzradius R= 1. An den
```
Randpunkten des Konvergenzintervalls gilt: Fur ̈ x= 1 ist die Reihe divergent (harmoni-
sche Reihe), und f ̈urx=−1 ist sie konvergent (alternierende harmonische Reihe). Wir
zeigen nun: F ̈ur reellesx∈[− 1 ,1[ gilt

```
∑∞
```
```
k=0
```
##### 1

```
k+ 1
```
```
xk+1=−ln(1−x). (42.2)
```

Das sieht man durch eine Taylorentwicklung von−ln(1−x) oder wie folgt. Die Ablei-
tungen der beiden Seiten sind

##### ∑∞

k=0x
k und^1
1 −xund sind damit gleich (geometrische
Reihe). Deshalb sind die beiden Funktionen gleich bis auf eine Konstante (nach dem
Konstanzkriterium, Satz 23.7), d.h. es ist

```
∑∞
```
```
k=0
```
##### 1

```
k+ 1
xk+1=−ln(1−x) +c
```
mit einer Konstantenc∈R. Einsetzen vonx= 0 liefertc= 0. Das liefert die Gleich-
heit (42.2) in ]− 1 ,1[. Man kann zeigen, dass Gleichheit auch noch in−1 wahr ist. Daraus
erhalten wir nach Multiplikation mit−1 den Wert der alternierenden harmonischen Rei-
he: ∞
∑

```
k=0
```
```
(−1)k
k+ 1
```
##### = 1−

##### 1

##### 2

##### +

##### 1

##### 3

##### −

##### 1

##### 4

```
±...= ln(2).
```
### 42.3 Taylor- und Potenzreihen

Ist die Funktionfauf einem Intervall umx 0 unendlich oft differenzierbar, so kann man
die Potenzreihe ∞
∑

```
k=0
```
```
f(k)(x 0 )
k!
(x−x 0 )k
```
bilden, die Taylorreihe vonfim Punktx 0. Die Funktion stimmt genau dann mit ihrer
Taylorreihe ̈uberein, wenn das Restglied gegen 0 konvergiert; vergleiche Abschnitt 25.4.
Das ist nicht f ̈ur alle Funktionen der Fall, aber f ̈ur die meisten Funktionen, die in der
Praxis auftreten. Insbesondere lassen sich fast alle elementaren Funktionen als Potenz-
reihen darstellen, die auf ganzCkonvergieren: exp, sin, cos, sowie sinh und cosh; siehe
Vorlesungen 26 und 27. Auch Polynome sind Potenzreihen, bei denen die Koeffizienten
akirgendwann alle Null sind. Die anderen elementaren Funktionen lassen sich ebenfalls
in eine Taylorreihe entwicklen, die aber nicht in der ganzen Ebene konvergiert, sondern
nur in einem Kreis, etwa die Wurzelfunktionen, der Logarithmus und die Arcus- und
Area-Funktionen.
Ist die Funktionfdurch eine Potenzreihe umx 0 gegeben, so stimmt die Taylorreihe
mit Entwicklunsgpunktx 0 mit der Potenzreihe ̈uberein: Ist

```
f(x) =
```
##### ∑∞

```
k=0
```
```
ak(x−x 0 )k,
```
so giltf(x 0 ) =a 0 und

```
f′(x) =
```
##### ∑∞

```
k=1
```
```
kak(x−x 0 )k−^1 , f′(x 0 ) = 1a 1 ,
```
```
f′′(x) =
```
##### ∑∞

```
k=2
```
```
k(k−1)ak(x−x 0 )k−^2 , f′′(x 0 ) = 2!a 2 ,
```

usw. Per Induktion rechnet man nach, dass

```
f(k)(x 0 ) =k!ak f ̈ur allek= 0, 1 , 2 ,...
```
so dassf

```
(k)(x 0 )
k! =akf ̈ur allekgilt. Damit haben wir folgenden Satz nachgerechnet.
```
Satz 42.7. Istfdurch eine Potenzreihe mit Entwicklungspunktx 0 gegeben,

```
f(x) =
```
##### ∑∞

```
k=0
```
```
ak(x−x 0 )k,
```
so ist die Taylorreihe vonfinx 0 genau wieder die Potenzreihe vonf.


Vorlesung 43

## 43 Lineare Differentialgleichungen

Differentialgleichungen sind Gleichungen, deren L ̈osung eine Funktion ist. In der Glei-
chung k ̈onnen sowohl die gesuchte Funktion als auch die Ableitungen der Funktion auf-
treten. Wir behandeln zwei einfache Beispiele:

```
y′=ay und y′′=−ay.
```
DieOrdnung einer Differentialgleichung (DGL) ist die h ̈ochste auftretende Ableitung.
So hat die DGLy′=aydie Ordnung 1 (day′die h ̈ochste auftretende Ableitung ist)
und y′′ =−ayhat die Ordnung 2 (wegeny′′). Wir sprechen von einerlinearen Dif-
ferentialgleichung, wenn nur Linearkombinationen der Funktionyund ihrer Ableitung
vorkommen. So sind beide DGL oben lineare DGL, aber zum Beispiely′=y^2 ist keine
lineare DGL.

### 43.1 Lineare Differentialgleichungen 1. Ordnung

Wir betrachten die lineare Differentialgleichung erster Ordnung

```
y′=ay
```
mit gegebenema∈ K(wobeiK= RoderK= C) und gesuchter Funktiony. Eine
L ̈osung ist
y:R→K, y(t) =eat,

denn es gilt

```
y′(t) =
d
dt
```
```
eat=aeat=ay(t).
```
Aber auch jedes Vielfache vonyist eine L ̈osung, also jede Funktiony(t) =ceat, mit be-
liebiger Konstantenc, denn f ̈ur diese isty′(t) =caeat=ay(t). Gibt es weitere L ̈osungen?
Der n ̈achste Satz zeigt, dass dies alle L ̈osungen sind.

Satz 43.1. Jede L ̈osung der DGL
y′=ay


ist von der Form
y(t) =ceat

mit einer Konstantenc∈K.

Beweis. Seiy(t) eine L ̈osung der DGL, d.h. es isty′(t) =ay(t). Betrachte die Funktion
f(t) =e−aty(t). Fur diese gilt ̈

```
f′(t) = (e−aty(t))′=−ae−aty(t) +e−aty′(t) =−ae−aty(t) +e−atay(t) = 0.
```
Daher istfauf ganzRkonstant (Satz 23.7), d.h. es gibt eine Konstantecmit

```
c=f(t) =e−aty(t),
```
und daher isty(t) =ceat.

Insbesondere ist die L ̈osungsmenge der DGLy′=ayein eindimensionaler Teilraum
der differenzierbaren Funktionen, und{eat}ist eine Basis dieses Teilraums.

Anfangswertprobleme Beschreibt man ein gegebenes Problem mit einer Differenti-
algleichungy′=ay, etwa den radioaktiven Zerfall einer Substanz, so wissen wir, dass
y(t) =ceatmit einer Konstantencist. Aber wie viel der Substanz haben wir nun zum
Zeitpunktt? Dazu mussen wir die Konstante ̈ cbestimmen. Dies geht, wenn wir die
Funktion zu einem Zeitpunkt, etwat 0 , kennen.
Erf ̈ullt die Funktiony die Differentialgleichungy′ =ay, und hat die Funktion zu
einem Zeitpunktt 0 einen bekannten Werty 0 , d.h. gilt auchy(t 0 ) =y 0 , so spricht man
von einem Anfangswertproblem. Man denkt sicht 0 als Anfangszeitpunkt undy 0 als
Anfangswert der Funktion. Mit dem Anfangswert k ̈onnen wir die Konstantecbestimmen
und die L ̈osung wird eindeutig.

Satz 43.2. Das Anfangswertproblem (AWP)

```
{
y′=ay
y(t 0 ) =y 0
```
hat die eindeutige L ̈osung
y(t) =ea(t−t^0 )y 0 , t∈R.

Beweis. Jede L ̈osung der Differentialgleichung hat die Formy(t) =ceat. Gilt nuny 0 =
y(t 0 ) =ceat^0 , so istc=e−at^0 y 0 , und damity(t) =e−at^0 y 0 eat=ea(t−t^0 )y 0.

Beispiel 43.3.Radioaktives Kalium-42 zerf ̈allt gem ̈aß der DGLy′=− 0. 055 y, wobei
y(t) die Masse (in Gramm) an vorhandenem Kalium zum Zeitpunktt(in Stunden) ist.
Daher isty(t) =e−^0.^055 tc. Betr ̈agt die Masse zum Zeitpunktt 0 = 0 dann 12 g, so ist
y(t) = 12e−^0.^055 tdie Masse zum Zeitpunktt.


### 43.2 Systeme linearer Differentialgleichungen 1. Ordnung

Wir betrachten nun ein System von Differentialgleichungen erster Ordnung der Form

```
~y′=A~y,
```
wobei die MatrixA∈Kn,ngegeben ist, und die Funktion

```
~y:R→Kn, t7→~y(t) =
```
##### 

##### 

##### 

```
y 1 (t)
..
.
yn(t)
```
##### 

##### 

##### ,

gesucht ist. Dabei ist die Ableitung von~yeintragsweise definiert:

```
~y′(t) =
```
##### 

##### 

##### 

```
y 1 ′(t)
..
.
y′n(t)
```
##### 

##### 

##### .

Wir wollen zun ̈achst annehmen, dassAdiagonalisierbar ist, das heißtA=SDS−^1
mit einem invertierbarenSund einer DiagonalmatrixD= diag(λ 1 ,...,λn). Dann ist

```
~y′=A~y=SDS−^1 ~y,
```
also
(S−^1 ~y)′=S−^1 ~y′=D(S−^1 ~y).

Setzen wir~z=S−^1 ~y, so ist also
~z′=D~z,

was ausgeschrieben die Gleichungen

```
z 1 ′=λ 1 z 1 ,
z 2 ′=λ 2 z 2 ,
..
.
zn′ =λnzn,
```
ergibt. Wir konnten das DGL-Systementkoppeln, jede Gleichung beinhaltet nur noch
eine gesuchte Funktion und wir sind zur ̈uck im skalaren Fall. Die skalaren DGL k ̈onnen
wir l ̈osen: Es sind

```
z 1 (t) =c 1 eλ^1 t, z 2 (t) =c 2 eλ^2 t, ..., zn(t) =cneλnt
```
mit Konstantenc 1 ,...,cn∈K. Dann ist also

```
~z(t) =
```
##### 

##### 

##### 

```
z 1 (t)
..
.
zn(t)
```
##### 

##### 

##### =

##### 

##### 

##### 

```
c 1 eλ^1 t
..
.
cneλnt
```
##### 

##### 

##### =

##### 

##### 

##### 

```
eλ^1 t
...
eλnt
```
##### 

##### 

##### 

##### ︸ ︷︷ ︸

```
=:etD
```
##### 

##### 

##### 

```
c 1
..
.
cn
```
##### 

##### 

##### 

##### ︸︷︷︸

```
=: ̃~c
```
##### ,


vergleiche Abschnitt 34.2 f ̈ur die Definition vonetD. Es ist also~z(t) =etD ̃~c. Nun sind
wir an der L ̈osung~yder DGL~y′=A~yinteressiert und rechnen daher zur ̈uck:

```
~y(t) =S~z(t) =SetD ̃~c=SetDS−^1 ︸︷︷︸S ̃~c
=~c
```
```
=etA~c,
```
wobei

```
etA=SetDS−^1 , etD=
```
##### 

##### 

##### 

```
etλ^1
...
etλn
```
##### 

##### 

##### . (43.1)

Die L ̈osung~y(t) =etA~csieht aus wie im skalaren Fall, nur dassetAeine matrixwertige
Funktion ist und~cein Vektor von Konstanten.

Bemerkung 43.4.Auf die Definition vonetDkommt man auch mit der Exponentialreihe:



```
eλ^1 t
...
eλnt
```
```

=
```
```



```
```
∑∞
k=0
λk 1 tk
k!
...
∑∞
k=0
```
```
λkntk
k!
```
```


=
```
```
∑∞
k=0
```
```
tk
k!
```
```


```
```
λk 1
...
λkn
```
```

=∑∞
k=0
```
```
tk
k!D
```
```
k=etD.
```
Auf (43.1) kommt man dann, da

```
Ak= (SDS−^1 )k=SDS−^1 SDS−^1 ...SDS−^1 =SDkS−^1
```
ist, so dass manSundS−^1 aus der Reihe herausziehen kann.

Bemerkung 43.5.Auch wennAnicht diagonalisierbar ist, ist jede L ̈osung von~y′=A~y
von der Form~y(t) =etA~c. In diesem Fall l ̈asst sichetAnicht durch Diagonalisieren von
Aberechnen, man kann aber die ReiheetA=

##### ∑∞

k=0
tkAk
k! verwenden.
Ist ein Anfangswert gegeben,~y(t 0 ) =~y 0 , so l ̈asst sich der Konstantenvektor wie im
skalaren Fall eindeutig bestimmen, denn

```
~y 0 =~y(t 0 ) =et^0 A~c,
```
also ist~c=e−t^0 A~y 0 und daher
~y(t) =e(t−t^0 )A~y 0.

Wir fassen zusammen und erhalten die Verallgemeinerung der S ̈atze 43.1 und 43.2 f ̈ur
allgemeinesn≥1.

Satz 43.6. SeiA∈Kn,n.

```
1) Das lineare Differentialgleichungssystem 1. Ordnung
```
```
~y′=A~y
```
```
hat die allgemeine L ̈osung
~y(t) =etA~c
mit~c∈Kn. Insbesondere ist die L ̈osungsmenge einn-dimensionaler Vektorraum.
```

```
2) Das Anfangswertproblem {
~y′=A~y
~y(t 0 ) =~y 0 ∈Kn
hat die eindeutige L ̈osung
~y(t) =e(t−t^0 )A~y 0.
```
Fur diagonalisierbares ̈ AkannetA ̈uber (43.1) berechnet werden, wobei die Gleichun-
gen beim Diagonalisieren entkoppelt werden.

Beispiel 43.7. 1) Die DGL~y′= [2 00 3]~yhat die L ̈osungen

```
~y(t) =
```
##### [

```
e^2 t 0
0 e^3 t
```
##### ][

```
c 1
c 2
```
##### ]

##### =

##### [

```
c 1 e^2 t
c 2 e^3 t
```
##### ]

```
Geben wir den Anfangswert~y(1) =
```
##### [

##### 5

##### 7

##### ]

```
vor, so hat das Anfangswertproblem die
eindeutige L ̈osung
~y(t) =
```
##### [

```
5 e2(t−1)
7 e3(t−1)
```
##### ]

```
2)~y′=
```
##### [

##### 2 1

##### 0 3

##### ]

```
~y: Eine Diagonalisierung vonAist
```
##### A=SDS−^1 =

##### [

##### 1 1

##### 0 1

##### ][

##### 2 0

##### 0 3

##### ][

##### 1 − 1

##### 0 1

##### ]

##### ,

```
also ist
```
```
~y(t) =etA~c=
```
##### [

##### 1 1

##### 0 1

##### ][

```
e^2 t 0
0 e^3 t
```
##### ][

##### 1 − 1

##### 0 1

##### ]

```
~c=
```
##### [

```
e^2 t e^3 t−e^2 t
0 e^3 t
```
##### ]

```
~c
```
##### =

##### [

```
(c 1 −c 2 )e^2 t+c 2 e^3 t
c 2 e^3 t
```
##### ]

```
Geben wir wieder den Anfangswert~y(1) =
```
##### [

##### 5

##### 7

##### ]

```
vor, so hat das Anfangswertproblem
die eindeutige L ̈osung
```
```
~y(t) =
```
##### [

```
− 2 e2(t−1)+ 5e3(t−1)
5 e3(t−1)
```
##### ]

### 43.3 Lineare skalare Differentialgleichungen 2. Ordnung

Wir betrachten die lineare Differentialgleichung zweiter Ordnung

```
x′′+ω^2 x= 0
```

wobeiω >0 gegeben undx=x(t) die gesuchte Funktion ist. Sicher sind cos(ωt) und
sin(ωt) L ̈osungen, und, da die DGL linear ist, auch alle Linearkombinationen dieser
beiden Funktionen (also alle Funktionenx(t) =c 1 cos(ωt) +c 2 sin(ωt)). Gibt es weitere
L ̈osungen?
Um diese Frage zu beantworten benutzen wir folgenden Trick: Wir betrachten die
Hilfsfunktion

```
~y(t) =
```
##### [

```
x′(t)
x(t)
```
##### ]

##### .

Dann ist

```
~y′(t) =
```
##### [

```
x′′(t)
x′(t)
```
##### ]

##### =

##### [

```
−ω^2 x(t)
x′(t)
```
##### ]

##### =

##### [

```
0 −ω^2
1 0
```
##### ][

```
x′(t)
x(t)
```
##### ]

```
=A~y(t),
```
d.h.~yerf ̈ullt ein lineares DGL-System erster Ordnung, dessen L ̈osung wir kennen und
ausrechnen k ̈onnen (Satz 43.6):
~y(t) =etA~c.

Insbesondere ist der L ̈osungsraum zweidimensional, d.h. es gibt nur zwei linear unabh ̈an-
gige L ̈osungen. Da die L ̈osungen sin(ωt) und cos(ωt) f ̈urω >0 linear unabh ̈angig sind,
bilden sie eine Basis des L ̈osungsraum, und alle L ̈osungen vonx′′+ω^2 x= 0 haben die
Gestalt
x(t) =c 1 cos(ωt) +c 2 sin(ωt)

mit Konstantenc 1 ,c 2. Die L ̈osungen sind also harmonische Schwingungen.
Um eine eindeutige L ̈osung zu bekommen, brauchen wir zwei Bedingungen, um die
beiden Konstanten zu bestimmen. Oft stellt man diese als Anfangswertbedingungen an
x(t 0 ) undx′(t 0 ) (
”
Startort“ und
”
Startgeschwindigkeit“).
Die L ̈osungen k ̈onnen wir auch direkt alsy(t) =etA~cberechnen. Das charakteristische Polynom
vonAistpA(z) =z^2 +ω^2 = (z−iω)(z+iω), also hatAdie Eigenwerteλ 1 =iωundλ 2 =−iω. Die
zugeh ̈origen Eigenr ̈aume sind

```
Kern(A−iωI) = Kern
```
```
([
− 1 iω−−ωiω^2
])
= span{[iω 1 ]},
Kern(A+iωI) = Kern
([iω−ω 2
1 iω
```
```
])
= span
{[−iω
1
```
```
]}
.
```
Daher istA=SDS−^1 mitD=
[iω 0
0 −iω

```
]
undS=
[iω−iω
1 1
```
```
]
, und damitS−^1 = 21 iω
[ 1 iω
− 1 iω
```
```
]
```
. Dann ist

```
etA=SetDS−^1 =
```
```
[
iω −iω
1 1
```
```
][
eiωt 0
0 e−iωt
```
```
]
1
2 iω
```
```
[
1 iω
− 1 iω
```
```
]
= 21 iω
```
```
[
iω −iω
1 1
```
```
][
eiωt iωeiωt
−e−iωt iωe−iωt
```
```
]
```
```
= 21 iω
```
```
[
iω(eiωt+e−iωt) i^2 ω^2 (eiωt−e−iωt)
eiωt−e−iωt iω(eiωt+e−iωt)
```
```
]
=
```
```
[
cos(ωt) −ωsin(ωt)
ω^1 sin(ωt) cos(ωt)
```
```
]
.
```
Dabei haben wir die Euler-Formeleiωt= cos(ωt) +isin(ωt) verwendet, aus der folgt

```
cos(ωt) = Re(eiωt) =^12 (eiωt+e−iωt), sin(ωt) = Im(eiωt) = 21 i(eiωt−e−iωt). (43.2)
```
Dax(t) der zweite Eintrag von~y(t) =etA~cist, finden wir alsox(t) =c (^11) ωsin(ωt) +c 2 cos(ωt) mit
beliebigen Konstantenc 1 ,c 2.
Eine allgemeine lineare Differentialgleichung 2. Ordnung
x′′(t) +a 1 x′(t) +a 0 = 0


kann man genauso l ̈osen. Wir verwenden den gleichen Trick: Mity(t) =

##### [

```
x′(t)
x(t)
```
##### ]

```
ist
```
```
~y′(t) =
```
##### [

```
−a 1 −a 0
1 0
```
##### ]

```
~y(t),
```
also~y(t) =etA~cgilt.
Fur allgemeine skalare lineare Differentialgleichungen ̈ n-ter Ordnung

```
x(n)+an− 1 x(n−1)+...+a 1 x′+a 0 x= 0
```
funktioniert der gleiche Trick, nun mit~y=

##### 

##### 

##### 

##### 

```
x(n−1)
..
.
x′
x
```
##### 

##### 

##### 

##### 

. Dann sind wieder~y′=A~yund

daher~y=etA~c.

Beispiel 43.8 (Ged ̈ampfte harmonische Schwingung). Als letztes Beispiel betrachten
wir die linear ged ̈ampfte Schwingung

```
x′′+ 2βx′+ω^20 x= 0
```
wobeiβ= 2 bm >0 die Abklingkonstante undω 0 =

##### √

k
m die unged ̈ampfte Eigenkreis-
frequenz sind. Die Bedeutung der Konstanten wird erst beim Betrachten der L ̈osungen
klar.

```
Wir L ̈osen wir eben: Mit~y(t) =
```
##### [

```
x′(t)
x(t)
```
##### ]

```
ist
```
```
~y′(t) =
```
##### [

```
x′′(t)
x′(t)
```
##### ]

##### =

##### [

```
− 2 βx′(t)−ω^20 x(t)
x′(t)
```
##### ]

##### =

##### [

```
− 2 β −ω 02
1 0
```
##### ][

```
x′(t)
x(t)
```
##### ]

```
=A~y(t),
```
und~y(t) =etA~c. Das charakteristische Polynom der Matrix istpA(z) = det(A−zI 2 ) =
z^2 + 2βz+ω^20 , somit sind die Eigenwerte

```
λ±=−β±
```
##### √

```
β^2 −ω 02.
```
Das verhalten der L ̈osungen h ̈angt davon ab, obβ^2 −ω 02 positiv, null oder negativ ist.

- Fallβ^2 > ω^20 : Dann ist der Term unter der Wurzel positiv, also hatAzwei ver-
    schiedene reelle Eigenwerte. Da~y(t) =etAd~=SetDS−^1 d~, ist der zweite Eintrag
    (egal wieSundS−^1 aussehen) eine Linearkombination voneλ+tundeλ−t:

```
x(t) =c 1 eλ+t+c 2 eλ−t,
```
```
mit Konstantenc 1 ,c 2. Da beide Eigenwerte negativ sind, klingt die L ̈osung expo-
nentiell ab, und der Fall wird auchKriechfallgenannt.
```

- Fallβ^2 =ω^20 : In diesem Fall hatAzwei gleiche Eigenwerteλ+=λ−=−β. Man
    kann nachrechnen, dassAnicht diagonalisierbar ist. Die L ̈osung erh ̈alt man als

```
x(t) =c 1 e−βt+c 2 te−βt.
```
```
Dieser Fall wirdaperiodischer Grenzfallgenannt.
```
- Fallβ^2 < ω^20 : Dann ist der Term unter der Wurzel negativ, undAhat zwei ver-
    schiedene komplexe Eigenwerte:

```
λ±=−β±
```
##### √

```
β^2 −ω^20 =−β∓i
```
##### √

```
ω^20 −β^2 =−β∓iωd.
```
```
(ωd>0 ist die ged ̈ampfte Eigenkreisfrequenz.) Die L ̈osung hat dann die Gestalt
```
```
x(t) =c 1 eλ+t+c 2 eλ−t=e−βt(c 1 eiωdt+c 2 e−iωdt).
```
```
Die komplexen Exponentialfunktionen k ̈onnen wie in (43.2) mit Sinus und Cosinus
geschrieben werden, so dass
```
```
x(t) =e−βt(d 1 sin(ωdt) +d 1 cos(ωdt)).
```
```
Der zweite Faktor beschreibt eine Schwingung (mit Frequenzωd), der erste Faktor
e−βt klingt exponentiell ab und d ̈ampft die Schwingung. Zusammen haben wir
eine ged ̈ampfte Schwingung (fallsβ >0). Dieser Fall wird daher auchSchwingfall
genannt.
Ist β= 0 (keine D ̈ampfung), so ist die DGLx′′+ω^2 x= 0, in der L ̈osung ver-
schwindet der D ̈ampfungsterm und wir haben wieder eine freie Sinus- und Cosi-
nusschwingung.
```
##### 0 5 10 15 20 25 30

##### − 1

##### 0

##### 1

##### 2

```
Kriechfall
aperiodischer Grenzfall
Schwingfall
```
Mehr zu Differentialgleichungen k ̈onnen Sie in den Vorlesungen
”
Differentialgleichun-
gen f ̈ur Ingenieurwissenschaften“ und”Integraltransformationen und partielle Differen-
tialgleichungen“ lernen.


## Index

Abbildung, 39
Bild, 40
Inverse, 43
Komposition, 42
linear, 115
Urbild, 40
Verkettung, 42
ableiten, 160
Ableitung, 160
der Umkehrfunktion, 165
der Wurzel, 166
Diskretisierung, 189
h ̈ohere, 167
zweite, 167
absolut konvergent, 309
Absolutbetrag
einer komplexen Zahl, 29, 30
einer reellen Zahl, 20
Absolute Konvergenz, 309
Addition
von Matrizen, 93
von Vektoren, 75
Additionstheorem
f ̈ur den Tangens, 54
f ̈ur Sinus und Cosinus, 50, 199
adjungiert, 98
Adjungierte, 98
̈aquivalente Aussagen, 15
algebraische Vielfachheit, 252
allgemeine Potenz, 196
alternierende harmonische Reihe, 307
Amplitude, 51, 282
Anfangswertproblem, 320
Approximation
lineare, 162

```
Arcus Tangens, 56
Arcus-Funktionen, 201
Arcuscosinus, 202
Arcussinus, 202
Hauptwert, 202
Arcustangens, 203
Area Cosinus hyperbolicus, 205
Area Sinus hyperbolicus, 205
Argument, 55
arithmetisches Mittel, 19
Aussage, 15
̈aquivalent, 15
Negation, 15
AWP,sieheAnfangswertproblem
```
```
Basis, 85, 196
des Bildes, 119
des Kerns, 119
eines Vektorraums, 85
naturliche, 196 ̈
Basis ̈ubergangsmatrix, 127
Basiswechsel
darstellende Matrix, 127
Koordinatenvektor, 127
Basiswechselmatrix, 127
Bernoulli-Ungleichung, 35
Beschleunigung, 168
Beschr ̈anktheit, 46
einer Folge, 132
einer Funktion, 46
Besselsche Ungleichung, 296
bestimmt divergent, 135
bestimmtes Integral, 210
Betrag
einer komplexen Zahl, 29, 30, 55
einer reellen Zahl, 20
```

Beweis durch Widerspruch, 16
bijektiv, 43
Bild, 40, 118
Basis, 119
Binomialkoeffizient, 36
binomische Formel, 37
Binomischer Lehrsatz, 37
Bisektionsverfahren, 152
Bogenmaß, 49, 55

Cauchy-Produkt, 305, 309
Cauchy-Schwarz-Ungleichung, 266
charakteristisches Polynom, 252
Cosinus, 50, 164, 192, 199
Cosinus hyperbolicus, 204
Cosinusschwingung, 53
Cotangens, 200
Cotanges hyperbolicus, 206

darstellende Matrix, 125
Definitionsbereich, 39
Determinante, 241
Determinantenmultiplikationssatz, 245
Dezimalzahl, 18
DGL,sieheDifferentialgleichung
diagonalisierbare Matrix, 257
Diagonalisierbarkeit, 257
Diagonalisierung, 257
Diagonalmatrix, 257
Differentialgleichung, 319
lineare, 319
Ordnung, 319
Differentialquotient, 161
Differenzenquotient, 161
differenzierbar
auf einer Menge, 160
in einem Punkt, 160
differenzieren, 160
Differenzmenge, 13
Dimension
eines Vektorraums, 86
Dimensionsformel fur lineare Abbildungen, ̈
120
direkter Beweis, 16

```
Diskretisierung der Ableitung, 189
Diskriminante, 31, 60
divergent, 133
divergente Folge, 133
Division mit Rest, 62
Divisionsrest, 62
Drei-Folgen-Satz, 139
Dreiecksform, 100
Dreiecksmatrix, 242
obere, 242
Dreiecksungleichung, 20, 31
Norm, 264
```
```
Eigenraum, 251
Eigenvektor, 250
Eigenwert, 250
Eigenwertgleichung, 250
einfacher Pol, 67
Einheitskreis, 49
Einheitsmatrix, 92
Element, 11
elementare Zeilenoperationen, 100
endlichdimensionaler Vektorraum, 86
Entwicklungspunkt, 180
Entwicklunspunkt, 313
erweiterte Koeffizientenmatrix, 103
Erzeugendensystem, 80
Erzeugnis,sieheSpan
euklidische Norm, 264
euklidischer Vektorraum, 265
Euler-Formel, 58, 192, 198, 299
Eulerdarstellung, 58
Eulersche Zahl, 186, 193, 196
Exponent, 196
Exponentialfunktion, 47, 193
komplexe, 198
Extremalstelle, 154
Extremum, 154, 171
strenges, 172
striktes, 172
Extremwert, 171
Extremwert-Test, 176
Extremwerte
hinreichendes Kriterium, 176, 182
```

```
notwendiges Kriterium, 173
```
Fakult ̈at, 25
Fehlerabsch ̈atzung, 187
Fehlerapproximation
qualitativ, 187
Fibonacci-Folge, 132, 261
Fl ̈achenberechnung, 211
Fl ̈achenbilanz, 211
Folge, 131
beschr ̈ankt, 132
nach oben, 132
nach unten, 132
bestimmt divergent, 135
divergent, 133
explizit definiert, 132
Fibonacci, 261
Glied, 131
Grenzwert, 133
komplexer Zahlen, 131, 136
konvergent, 133
Konvergenz, 133
monoton, 132
monoton fallend, 132
monoton wachsend, 132
nach oben beschr ̈ankt, 132
nach unten beschr ̈ankt, 132
Nullfolge, 133
reeller Zahlen, 131
rekursiv definiert, 132
streng monoton, 132
streng monoton fallend, 132
streng monoton wachsend, 132
Folgenglied, 131
Fourierkoeffizienten
komplex, 300
reell, 285
Fourierpolynom
komplex, 300
reell, 285
Fourierreihe, 295
Frequenz, 51–53, 282
Fundamentalsatz der Algebra, 64
Funktion, 39

```
Ableitung, 160
beschr ̈ankt, 46
differenzierbar, 160
gerade, 44
Grenzwert, 143
integrierbar, 213
monoton fallend, 45, 175
monoton wachsend, 45, 175
Monotonie, 45
nach oben beschr ̈ankt, 46
nach unten beschr ̈ankt, 46
Nullfunktion, 40
Periode, 281
periodisch, 50, 281
punktsymmetrisch, 45
quadratisches Mittel, 293
spiegelsymmetrisch, 45
stetig, 146, 147
streng monoton fallend, 45, 175
streng monoton wachsend, 45, 175
strenge Monotonie, 45
stuckweise monoton, 213 ̈
stuckweise stetig, 213 ̈
T-periodisch, 281
uneigentlich integrierbar, 231, 232, 234
ungerade, 45
Funktionalgleichung
der Exponentialfunktion, 47
des Logarithmus, 48
```
```
ganze Zahlen, 17
Gauß-Algorithmus, 101
Gaußsche Zahlenebene, 28
geometrische Folge, 141
geometrische Reihe, 142, 304
geometrische Summe, 23, 24, 34
geometrische Vielfachheit, 251
geometrisches Mittel, 19
gerade Funktion, 44
Geschwindigkeit, 162
Gleichung
quadratische, 31, 60
Glied
einer Folge, 131
```

einer Reihe, 303
globales Maximum, 172
globales Minimum, 172
Grad eines Polynoms, 61
Gradformel, 62
Gram-Schmidt-Verfahren, 275
Grenzwert
einer Folge, 133
einer Funktion, 143
linksseitig, 145
rechtsseitig, 146
Grenzwerts ̈atze
fur Folgen, 137 ̈
fur Funktionen, 145 ̈
Grundintegrale, 220
Grundschwingung, 282

harmonische Reihe, 307
harmonische Schwingung, 53, 281
Hauptsatz der Differential- und Integral-
rechnung, 218
Hauptwert
des Arcussinus, 202
Hermitesch, 98
Homogenit ̈at, 264
Homomorphismus, 115
Householder-Matrix, 272
Hutfunktion, 90
Hyperbelfunktionen, 204
Cosinus hyperbolicus, 204
Cotangens hyperbolicus, 206
Sinus hyperbolicus, 204
Tangens hyperbolicus, 206

Identit ̈at, 40
Imagin ̈arteil, 29
Implikation, 15
Index, 131
Indexverschiebung, 23
Induktion, 33
Induktionsanfang, 33
Induktionsbehauptung, 34
Induktionsschritt, 33, 34
Induktionsvoraussetzung, 34

```
Infimum, 156
Inhomogenit ̈at, 99
injektiv, 43
Integral, 210
bestimmtes, 210
einer komplexwertigen Funktion, 229
uneigentliches, 231, 232, 234
Integralfunktion, 218
Integralvergleichskriterium, 305
Integrand, 210
Integration
partielle, 222
Integrationsgrenze
obere, 210
untere, 210
Integrationsvariable, 210
integrierbar, 213
uneigentlich, 231, 232, 234
Intervall, 13
kompakt, 13
Intervallhalbierungsverfahren, 151
Inverse, 43, 96, 113, 117
Koordinatenabbildung, 124
lineare Abbildung, 117
Matrix, 96
Isomorphismus, 117
```
```
kartesisches Produkt, 13
Kern, 118
Basis, 119
Kettenregel, 163
kleinste-Quadrate-Approximation, 279
Koeffizienten
einer Linearkombination, 78
einer Potenzreihe, 313
eines linearen Gleichungssystems, 99
eines Polynoms, 61
Koeffizientenmatrix, 99
erweiterte, 103
Koeffizientenvergleich, 64
kompaktes Intervall, 13
komplex konjugiert, 29
komplexe Exponentialfunktion, 198
komplexe Folge, 131, 136
```

komplexe Partialbruchzerlegung, 68
komplexe Wurzel, 59
komplexe Zahlen, 27
komplexe Zahlenfolge, 131, 136
komplexes trigonometrisches Polynom, 300
Komposition, 42
Konjugation, 29, 30
Kontraposition, 16
konvergente Folge, 133
Konvergenz
bei Funktionen, 143
einer Reihe, 303
einer Zahlenfolge, 133
Konvergenzkreis, 314
Konvergenzkriterium
Integralvergleichskriterium, 305
Leibnizkriterium, 307
Majorantenkriterium, 310
Minorantenkriterium, 310
Notwendiges Kriterium, 305
Quotientenkriterium, 311
Konvergenzradius, 314
Koordinaten, 88
eines Vektors, 88
Koordinatenabbildung, 124
Koordinatenvektor, 88, 123
bei Basiswechsel, 127
Kr ̈ummung, 168

l’Hospital
Regel von, 168
Lagrange-Restglied, 180
Laplace-Entwicklung, 244
least squares approximation, 279
leere Menge, 12
leere Summe, 23
leeres Produkt, 25
Leibnizkriterium, 307
LGS,sieheLineares Gleichungssystem
Limes,sieheGrenzwert
linear abh ̈angig, 83
linear unabh ̈angig, 83
Lineare Abbildung, 115
lineare Abbildung

```
Bild, 118
Dimensionsformel, 120
injektiv, 120
Kern, 118
Matrixdarstellung, 125
Potenz, 117
surjektiv, 121
Umkehrabbildung, 117
lineare Abh ̈angigkeit, 83
lineare Approximation, 162
lineare Differentialgleichung, 319
```
1. Ordnung, 319
2. Ordnung, 323, 324
n-ter Ordnung, 325
lineare H ̈ulle,sieheSpan
Lineare Regression, 279
lineare Unabh ̈angigkeit, 83
Lineares Gleichungssystem, 99
homogen, 99
inhomogen, 99
Inhomogenit ̈at, 99
Koeffizienten, 99
Koeffizientenmatrix, 99
komplexes, 99
L ̈osung, 100
L ̈osungsmenge, 100, 106
partikul ̈are L ̈osung, 106
reelles, 99
spezielle L ̈osung, 106
Linearfaktor, 63
Linearfaktorzerlegung, 64
Linearkombination, 78
Linears Gleichungssystem
L ̈osbarkeitskriterium, 111
linksseitiger Grenzwert, 145
L ̈osbarkeitskriterium f ̈ur LGS, 111
L ̈osung
eines linearen Gleichungssystems, 100
L ̈osungsmenge
eines linearen Gleichungssystems, 100,
106
Logarithmus, 48, 195
zur Basisa, 197


lokales Maximum, 172
lokales Minimum, 172
Lot, 274

Majorantenkriterium, 310
Matrix, 91
Addition, 93
Adjungierte, 98
darstellende, 125
diagonalisierbar, 257
Dreiecksform, 100
Hermitesch, 98
Householder-Matrix, 272
Inverse, 96, 113
Multiplikation, 94
orthogonal, 270
quadratisch, 91
Rang, 109
Skalarmultiplikation, 93
Transponierte, 97
unit ̈ar, 272
Zeilenstufenform, 101
Matrixdarstellung, 125
bei Basiswechsel, 127
Matrizenmultiplikation, 94
Maximalstelle, 154
Maximum, 154
globales, 172
lokales, 172
strenges, 172
striktes, 172
Maximumsnorm, 264
mehrfacher Pol, 67
Menge, 11
Differenz, 13
Element, 11
leere Menge, 12
Schnitt, 12
Teilmenge, 12
Vereinigung, 12
Minimalstelle, 154
Minimum, 154
globales, 172
lokales, 172

```
strenges, 172
striktes , 172
Minorantenkriterium, 310
Mittelwert
einer Funktion, 215
Mittelwertsatz, 174
der Integralrechnung, 215
monoton
st ̈uckweise, 213
Monotonie, 45
einer Folge, 132
einer Funktion, 45
Monotoniekriterium
f ̈ur Folgen, 140
f ̈ur Funktionen, 175
```
```
naturliche Zahlen, 17 ̈
naturliche Basis, 196 ̈
naturlicher Logarithmus, 48 ̈
Negation, 15
Newton-Verfahren, 166
Norm, 263
Dreiecksungleichung, 264
euklidische, 264
homogen, 264
Maximumsnorm, 264
p-Norm, 264
positiv definit, 264
Standardnorm, 264
normierte Zeilenstufenform, 102
Notwendiges Extremwertkriterium, 173
Nullabbildung, 40
Nullfolge, 133
Nullmatrix, 92
Nullstelle, 63
eines Polynoms, 63
einfache, 64
mehrfache, 64
Vielfachheit, 64
Nullvektor, 75
NZSF,siehenormierte Zeilenstufenform
```
```
obere Dreiecksmatrix, 242
obere Integrationsgrenze, 210
```

Oberschwingung, 282
ONB,sieheOrthonormalbasis
Ordnung
einer Differentialgleichung, 319
eines Pols, 67
orthogonal, 267
Matrix, 270
Vektor, 267
orthogonale Projektion, 274
Orthogonalit ̈atsrelationen, 283
orthonormal, 267
Orthonormalbasis, 268

p-Norm, 264
Parallelepiped, 240
Parallelotop, 240
Parsevalsche Gleichung, 296
Partialbruchzerlegung, 68
komplexe, 68
reelle, 71
Partialsumme, 303
partielle Integration, 222
partikul ̈are L ̈osung
eines LGS, 106
Pascalsches Dreieck, 36
PBZ,siehePartialbruchzerlegung
Periode, 52, 281
periodische Funktion, 281
Phasenverschiebung, 51, 52
Pol, 67
der Ordnungk, 67
einfacher, 67
mehrfacher, 67
Polardarstellung, 55
Polstelle,siehePol
Polynom, 61
Addition, 62
charakteristisches, 252
Division mit Rest, 62
Grad, 61
Koeffizienten, 61
Koeffizientenvergleich, 64
Linearfaktorzerlegung, 64
Multiplikation, 62

```
reelle Zerlegung, 65
reelles, 61
Skalarmultiplikation, 62
trigonometrisches, 282, 300
Vektorraum, 76
Polynomdivision, 62
Positive Definitheit
Norm, 264
Skalarprodukt, 265
Potenz
allgemeine, 196
einer linearen Abbildung, 117
Potenzreihe, 313
Ableitung, 316
Eintwicklungspunkt, 313
Koeffizienten, 313
Konvergenzkreis, 314
Konvergenzradius, 314
Produkt
leeres Produkt, 25
Produktformel von Cauchy, 305
Produktregel, 163
Produktzeichen, 25
Projektion, 274
punktsymmetrisch, 45
```
```
QR-Zerlegung, 276
Berechnung, 277
quadratische Gleichung
komplexe Koeffizienten, 60
reelle Koeffizienten, 31
quadratische Matrix, 91
quadratischer Fehler, 293
quadratisches Mittel, 293
Quadratwurzel, 20
qualitative Fehlerapproximation, 187
Quotientenkriterium, 311
Quotientenregel, 163
```
```
Rang, 109
rationale Funktion, 67
Pol, 67
rationale Zahlen, 17
Realteil, 29
```

Rechteckspannung, 292
rechtsseitiger Grenzwert, 146
reelle Folge, 131
reelle Fourierkoeffizienten, 285
reelle Partialbruchzerlegung, 71
reelle Zahlen, 18
reelle Zahlenfolge, 131
reelles Polynom, 61
reelles trigonometrisches Polynom, 282
Regel von l’Hospital, 168
Reihe, 142, 303
absolut konvergent, 309
Absolute Konvergenz, 309
divergent, 303
Divergenz, 303
Glied, 303
konvergent, 303
Konvergenz, 303
Partialsumme, 303
Summe, 303
unendliche, 303
Wert, 303
Rest, 62
Restglied, 180
Lagrange-Darstellung, 180
Riemann-Summe, 210
Riemannsche Summe, 210
R ̈uckw ̈artssubstitution, 100

Sandwich-Theorem, 139
Satz des Pythagoras, 268
Schnitt, 12
Schwingung
Cosinusschwingung, 53
harmonische, 53
Sinusschwingung, 51
Sekante, 159
senkrecht,sieheorthogonal
Sinus, 50, 164, 191, 199
Sinus hyperbolicus, 204
Sinusschwingung, 51
Skalar, 75
Skalarmultiplikation
von Matrizen, 93

```
von Vektoren, 75
Skalarprodukt, 264
Spaltenvektor, 92
Span, 79
Spat, 240
Spatprodukt, 240
spezielle L ̈osung
eines LGS, 106
spiegelsymmetrisch, 45
Stammfunktion, 217
Standardbasis vonKn, 86
Standardnorm, 264
Standardskalarprodukt, 265
Stelle
eines globalen Maximums, 172
eines globalen Minimums, 172
eines lokalen Maximums, 172
eines lokalen Minimums, 172
eines Maximums, 154
eines Minimums, 154
stetig
st ̈uckweise, 213
stetige Fortsetzung, 149
Stetigkeit, 146, 147
Streichungsmatrix, 241
strenges
Extremum, 172
Maximum, 172
Minimum, 172
striktes
Extremum , 172
Maximum, 172
Minimum, 172
st ̈uckweise
monoton, 213
stetig, 213
Substitutionsregel
```
1. Version, 225
2. Version, 227
Summationsindex, 23
Summe
einer Reihe, 303
geometrische Summe, 23, 24


leere Summe, 23
Summenzeichen, 22
Supremum, 156
surjektiv, 43

Tangens, 53, 200
Tangens hyperbolicus, 206
Tangente, 159, 161
Taylorformel, 180
Taylorpolynom, 180
Taylorreihe, 190
von cos, 192
von exp, 190, 191
von ln, 195
von sin, 191
Teilmenge, 12
Teilraum, 77
Teilraumkriterium, 77
Transponierte, 97
Transposition, 97
Trigonometrische Funktion
Cosinus, 50, 164, 192, 199
Cotangens, 200
Sinus, 50, 164, 191, 199
Tangens, 53, 200
Trigonometrischer Pythagoras, 50
trigonometrisches Polynom
komplex, 300
reell, 282

Umkehrabbildung, 43
einer linearen Abbildung, 117
Umkehrfunktion
Ableitung, 165
Unbekannte, 99
unbestimmtes Integral, 217
uneigentlich integrierbar, 231, 232, 234
uneigentliches Integral, 231, 232, 234
divergent, 232, 234
konvergent, 232, 234
unendlichdimensionaler Vektorraum, 86
unendliche Reihe, 303
ungerade Funktion, 45
unit ̈ar

```
Matrix, 272
unit ̈arer Vektorraum, 265
untere Integrationsgrenze, 210
Unterraum,sieheTeilraum
Untervektorraum,sieheTeilraum
Urbild, 40
```
```
Vektor, 75
Koordinaten, 88
Nullvektor, 75
orthogonal, 267
orthonormal, 267
Vektorraum, 75
Addition, 75
Basis, 85
der linearen Abbildungen, 116
der Polynome, 76
vom Grad h ̈ochstensn, 78
Dimension, 86
endlichdimensional, 86
euklidisch, 265
Kn, 76
Skalarmultiplikation, 75
unendlichdimensional, 86
unit ̈ar, 265
Vereinigung, 12
Verkettung, 42
Vielfachheit
algebraische, 252
einer Nullstelle, 64
geometrische, 251
vollst ̈andige Induktion, 33
```
```
Wahrheitstafel, 15
Wendepunkte, 168
Wertebereich, 39
Wheatstone-Br ̈ucke, 188
Wurzel, 20, 166
Ableitung, 166
komplexe, 59
n-te Wurzel, 20
Quadratwurzel, 20
Wurzelfolge, 140, 167
```

Zahlen
ganze Zahlen, 17
komplexe Zahlen, 27
nat ̈urliche Zahlen, 17
rationale Zahlen, 17
reelle Zahlen, 18
Zahlenfolge, 131
komplexe, 131, 136
reelle, 131
Zeilenstufenform, 101
normierte, 102
Zeilenvektor, 92
ZSF,sieheZeilenstufenform
Zuhaltemethode, 69
zweite Ableitung, 167
Zwischenwertsatz, 151, 154
""".encode("utf-8", errors="replace").decode("utf-8")

async def math():
    kb = KnowledgeBase(n_clusters=3, model_name="openrouter/mistralai/mistral-7b-instruct", requests_per_second=10, batch_size=20, chunk_size=36000, chunk_overlap=300)

    r = await kb.add_data([text], metadata=None)
    print(r)
    GraphVisualizer.visualize(kb.concept_extractor.concept_graph.convert_to_networkx(), output_file="Mathe_graph.html")

    kb.save("mathe.pkl")

    while u := input("User"):
        if u.startswith("C"):
            print("A:", await kb.query_concepts(u[1:]))
        if u.startswith("R"):
            print("A:", await kb.retrieve_with_overview(u[1:]))
        if u.startswith("U"):
            print("A:", await kb.unified_retrieve(u[1:]))

if __name__ == "__main__":
    get_app(name="main2")

    asyncio.run(math())

