import asyncio
import hashlib
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import deque

import faiss
import networkx as nx
import numpy as np
import umap
from litellm import batch_completion, ModelResponse
from pydantic import BaseModel, Field

from toolboxv2 import get_logger, get_app
from sklearn.cluster import HDBSCAN

from typing import Dict, List, Set, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import json
import asyncio
from collections import defaultdict

from typing import List
import re

from toolboxv2.mods.isaa.extras.adapter import litellm_complete

from pyvis.network import Network

i__ = [0, 0]

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
        metadata (Dict[str, Any]): Additional metadata associated with the concept.
    """
    name: str
    category: str
    relationships: Dict[str, List[str]]
    importance_score: float
    context_snippets: List[str]
    metadata: Dict[str, Any]

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
        concepts (List[Concept]): A list of Concept instances, each representing an individual key concept.
    """
    concepts: List[rConcept]

class ConceptAnalysis(BaseModel):
    """
    Represents the analysis of key concepts.

    Attributes:
        key_concepts (List[str]): A list of primary key concepts identified.
        relationships (List[str]): A list of relationships between the identified key concepts.
        importance_hierarchy (List[str]): A list that represents the hierarchical importance of the key concepts.
    """
    key_concepts: List[str]
    relationships: List[str]
    importance_hierarchy: List[str]


class TopicInsights(BaseModel):
    """
    Represents insights related to various topics.

    Attributes:
        primary_topics (List[str]): A list of main topics addressed.
        cross_references (List[str]): A list of cross-references that connect different topics.
        knowledge_gaps (List[str]): A list of identified gaps in the current knowledge.
    """
    primary_topics: List[str]
    cross_references: List[str]
    knowledge_gaps: List[str]


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
        if concept.name in self.concepts:
            # Merge relationships and context
            existing = self.concepts[concept.name]
            for rel_type, related in concept.relationships.items():
                if rel_type not in existing.relationships:
                    existing.relationships[rel_type] = set()
                existing.relationships[rel_type].update(related)
            existing.context_snippets.extend(concept.context_snippets)
            # Update importance score with rolling average
            existing.importance_score = (existing.importance_score + concept.importance_score) / 2
        else:
            self.concepts[concept.name] = concept

    def get_related_concepts(self, concept_name: str, relationship_type: Optional[str] = None) -> Set[str]:
        """Get related concepts, optionally filtered by relationship type"""
        if concept_name not in self.concepts:
            return set()

        concept = self.concepts[concept_name]
        if relationship_type:
            return concept.relationships.get(relationship_type, set())

        related = set()
        for relations in concept.relationships.values():
            related.update(relations)
        return related


    def convert_to_networkx(self) -> nx.DiGraph:
        """Convert ConceptGraph to NetworkX graph with layout"""
        G = nx.DiGraph()

        for concept in self.concepts.values():
            G.add_node(
                concept.name,
                size=concept.importance_score * 10,
                group=concept.category,
                title=f"""
                    <b>{concept.name}</b><br>
                    Category: {concept.category}<br>
                    Importance: {concept.importance_score:.2f}<br>
                    Context: {', '.join(concept.context_snippets[:3])}
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
        print(f"Graph saved to {output_file}. Open in browser to view.")
        if get_output:
            return open(output_file, "r", encoding="utf-8").read()

class ConceptExtractor:
    """Handles extraction of concepts and relationships from text"""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.concept_graph = ConceptGraph()

    def extract_concepts(self, texts: List[str]) -> List[List[Concept]]:
        """
        Extract concepts and their relationships from a list of texts using LLM batch processing.
        Returns a list of lists of Concept objects (one list per input text).
        """

        system_prompt = (
            "Analyze the given text and extract key concepts and their relationships. For each concept:\n"
            "1. Identify the concept name and category (technical, domain, method, property)\n"
            "2. Determine relationships with other concepts (uses, part_of, similar_to, depends_on)\n"
            "3. Assess importance (0-1 score) based on centrality to the text\n"
            "4. Extract relevant context snippets\n\n"
            "Format the response as a JSON array of concepts with their details."
        )

        # Create a list of messages for each text
        messages_list = []
        for text in texts:
            prompt = f"""
    Text to analyze:
    {text}
"""
    # Extract concepts following this JSON structure:
    # {{
    #     "concepts": [
    #         {{
    #             "name": "concept_name",
    #             "category": "category_name",
    #             "relationships": {{
    #                 "relationship_type": ["related_concept1", "related_concept2"]
    #             }},
    #             "importance_score": 0.0,
    #             "context_snippets": ["relevant text snippet"]
    #         }}
    #     ]
    # }}

            # Each conversation includes the system prompt and the user prompt.
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            messages_list.append(conversation)

        try:
            # Increment your counter (if needed; i__[1] is from your original code)
            i__[1] += len(messages_list)

            # Call batch_completion once for all texts.
            responses = batch_completion(
                model=self.kb.model_name,
                messages=messages_list,
                response_format=Concepts,
                # fallbacks=os.getenv("FALLBACKS_MODELS", '').split(','),

            )

            all_concepts = []  # This will be a list (per text) of lists of Concept objects
            for response in responses:
                if not isinstance(response, ModelResponse):
                    continue
                c: str = response.choices[0].message.content
                if c is None:
                    c: str = response.choices[0].message.tool_calls[0].function.arguments
                if c is None:
                    continue
                concept_data = json.loads(c)
                # print(concept_data)
                concepts = []
                for concept_info in concept_data.get("concepts", []):
                    concept = Concept(
                        name=concept_info["name"],
                        category=concept_info["category"],
                        relationships={k: set(v) for k, v in concept_info["relationships"].items()},
                        importance_score=concept_info["importance_score"],
                        context_snippets=concept_info["context_snippets"],
                        metadata={}
                    )
                    concepts.append(concept)
                    # Update your concept graph if needed
                    self.concept_graph.add_concept(concept)
                all_concepts.append(concepts)
            return all_concepts

        except Exception as e:
            print(f"Error extracting concepts: {str(e)}")
            # In case of error, return an empty list for each text
            return [[] for _ in texts]

    async def process_chunks(self, chunks: List[Chunk]) -> None:
        """
        Process all chunks in batch to extract and store concepts.
        Each chunk's metadata will be updated with the concept names and relationships.
        """
        # Gather all texts from the chunks.
        texts = [chunk.text for chunk in chunks]
        # Call extract_concepts once with all texts.
        all_concepts = self.extract_concepts(texts)

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
        chunk_size: int = 12_000,
        chunk_overlap: int = 200,
        separator: str = "\n"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

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

            # Move start position considering overlap
            start = end - self.chunk_overlap

        return chunks

class KnowledgeBase:
    def __init__(
        self,
        embedding_dim: int = 768,
        similarity_threshold: float = 0.7,
        batch_size: int = 64,
        n_clusters: int = 26,
        deduplication_threshold: float = 0.85,
        model_name = "groq/llama3-70b-8192",
        embedding_model = "gemini/text-embedding-004"
    ):
        """Initialize the knowledge base with given parameters"""
        self.chunks: deque = deque(maxlen=1_000_000_000)
        self.existing_hashes: Set[str] = set()
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.deduplication_threshold = deduplication_threshold
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.model_name = model_name
        self.sto: list = []

        self.text_splitter = TextSplitter()
        self.similarity_graph = {}
        self.concept_extractor = ConceptExtractor(self)

        # Initialize FAISS index
        self.quantizer = faiss.IndexFlatIP(embedding_dim)
        self.index = faiss.IndexIVFFlat(
            self.quantizer,
            embedding_dim,
            n_clusters,
            faiss.METRIC_INNER_PRODUCT
        )
        self.index.nprobe = min(20, n_clusters)
        self.is_trained = False

    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute SHA-256 hash of text"""
        return hashlib.sha256(text.encode()).hexdigest()

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

    def _find_similar_chunks(self, embeddings: np.ndarray, threshold: float) -> np.ndarray:
        """Find chunks that are similar above the threshold"""
        if not self.is_trained or len(self.chunks) == 0:
            return np.zeros(len(embeddings), dtype=bool)

        try:
            # Search for nearest neighbors
            k = min(10, len(self.chunks))  # Search top-k nearest neighbors
            D, _ = self.index.search(embeddings, k)

            # Check if any similarity is above threshold
            return np.any(D >= threshold, axis=1)
        except Exception as e:
            get_logger().error(f"Error finding similar chunks: {str(e)}")
            raise

    def _remove_similar_chunks(self, threshold: float = None) -> int:
        """Remove chunks that are too similar to each other"""
        if len(self.chunks) < 2:
            return 0

        if threshold is None:
            threshold = self.deduplication_threshold

        try:
            # Get all embeddings
            embeddings = np.vstack([c.embedding for c in self.chunks])
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
            unique_chunks = [chunk for chunk, keep in zip(self.chunks, keep_mask) if keep]
            removed_count = len(self.chunks) - len(unique_chunks)

            # Update chunks and hashes
            self.chunks = deque(unique_chunks, maxlen=self.chunks.maxlen)
            self.existing_hashes = {chunk.content_hash for chunk in self.chunks}

            # Rebuild index if chunks were removed
            if removed_count > 0:
                self._rebuild_index()


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

            texts, metadata, hashes = zip(*unique_data)
            texts = list(texts)
            metadata = list(metadata)
            hashes = list(hashes)

            # Get embeddings
            embeddings = await self._get_embeddings(texts)

            # Check for near-duplicates using embeddings
            if self.is_trained:
                similar_mask = self._find_similar_chunks(
                    embeddings,
                    self.deduplication_threshold
                )

                # Keep only non-similar chunks
                keep_mask = ~similar_mask
                texts = [t for t, k in zip(texts, keep_mask) if k]
                metadata = [m for m, k in zip(metadata, keep_mask) if k]
                hashes = [h for h, k in zip(hashes, keep_mask) if k]
                embeddings = embeddings[keep_mask]

            if not texts:  # All were similar to existing chunks
                return 0, len(unique_data)

            # Create and add new chunks
            new_chunks = [
                Chunk(text=t, embedding=e, metadata=m, content_hash=h)
                for t, e, m, h in zip(texts, embeddings, metadata, hashes)
            ]

            # Add new chunks
            self.chunks.extend(new_chunks)
            # Update index
            if new_chunks:
                all_embeddings = np.vstack([c.embedding for c in new_chunks])
                self._train_index(all_embeddings)

                if self.is_trained:
                    n = len(new_chunks)
                    start_idx = len(self.chunks) - n
                    self.index.add_with_ids(
                        all_embeddings,
                        np.arange(start_idx, start_idx + n, dtype=np.int64)
                    )

            # Remove similar chunks from the entire collection
            removed = self._remove_similar_chunks()
            get_logger().info(f"Removed {removed} similar chunks during deduplication")

            # Invalidate visualization cache

            if len(new_chunks) - removed > 0:
                # Process new chunks for concepts
                new_chunks = list(self.chunks)[-(len(new_chunks) - removed):]
                await self.concept_extractor.process_chunks(new_chunks)
            print("i________________________________________________________________", i__)
            return len(new_chunks) - removed, len(texts) - len(new_chunks) + removed

        except Exception as e:
            get_logger().error(f"Error adding data: {str(e)}")
            raise

    async def _retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        min_similarity: float = 0.7
    ) -> List[Chunk]:
        """Retrieve relevant chunks"""
        if not self.is_trained:
            get_logger().warning("Index not trained yet, returning empty results")
            return []

        try:
            query_embedding = normalize_vectors(
                query_embedding.reshape(1, -1)
            )

            D, I = self.index.search(query_embedding, k)
            mask = D[0] >= min_similarity
            results = [list(self.chunks)[i] for i in I[0][mask]]

            return sorted(
                results,
                key=lambda x: np.dot(x.embedding, query_embedding[0]),
                reverse=True
            )[:k]

        except Exception as e:
            get_logger().error(f"Error retrieving chunks: {str(e)}")
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
                print("slipp", len(self.sto), len(texts[0]))
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
        min_similarity: float = 0.7,
        include_connected: bool = True
    ) -> List[Chunk]:
        """Enhanced retrieval with connected information"""
        if query_embedding is None:
            query_embedding = (await self._get_embeddings([query]))[0]
        initial_results = await self._retrieve(query_embedding, k, min_similarity)

        if not include_connected or not initial_results:
            return initial_results

        # Find connected chunks
        connected_chunks = set()
        for chunk in initial_results:
            chunk_id = list(self.chunks).index(chunk)
            if chunk_id in self.similarity_graph:
                connected_chunks.update(self.similarity_graph[chunk_id])

        # Add connected chunks to results
        all_chunks = list(self.chunks)
        additional_results = [all_chunks[i] for i in connected_chunks
                              if all_chunks[i] not in initial_results]

        # Sort by similarity to query
        all_results = initial_results + additional_results

        return sorted(
            all_results,
            key=lambda x: np.dot(x.embedding, query_embedding),
            reverse=True
        )[:k * 2]  # Return more results when including connected information

    def _train_index(self, vectors: np.ndarray) -> None:
        """Train the IVF index if not already trained"""
        if not self.is_trained and vectors.shape[0] >= self.n_clusters:
            get_logger().info("Training IVF index...")
            try:
                self.index.train(vectors)
                self.is_trained = True
                get_logger().info("IVF index training completed")
            except Exception as e:
                get_logger().error(f"Error training index: {str(e)}")
                raise

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index"""
        if not self.chunks:
            return

        try:
            embeddings = np.vstack([c.embedding for c in self.chunks])
            n = embeddings.shape[0]

            self._train_index(embeddings)

            self.index.reset()
            if self.is_trained:
                self.index.add_with_ids(
                    embeddings,
                    np.arange(n, dtype=np.int64)
                )
        except Exception as e:
            get_logger().error(f"Error rebuilding index: {str(e)}")
            raise

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
            initial_count = len(self.chunks)

            def is_relevant(chunk: Chunk) -> bool:
                similarities = np.dot(chunk.embedding, irrelevant_embeddings.T)
                do_keep = np.max(similarities) < similarity_threshold
                if do_keep:
                    return True
                for c in chunk.metadata.get("concepts", []):
                    if c in self.concept_extractor.concept_graph.concepts:
                        del self.concept_extractor.concept_graph.concepts[c]
                return False

            relevant_chunks = deque(
                [chunk for chunk in self.chunks if is_relevant(chunk)],
                maxlen=self.chunks.maxlen
            )

            self.chunks = relevant_chunks
            self.existing_hashes = {chunk.content_hash for chunk in self.chunks}
            self._rebuild_index()


            return initial_count - len(self.chunks)

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
        min_similarity: float = 0.7,
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
        initial_ids = {list(self.chunks).index(chunk) for chunk in initial_results}
        related_ids = self._find_cross_references(
            initial_ids,
            depth=cross_ref_depth,
            query_embedding=query_embedding  # Pass query embedding for relevance scoring
        )

        # Get all relevant chunks with smarter filtering
        all_chunks = list(self.chunks)
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
                            [list(self.chunks)[cid]],
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
        min_similarity: float = 0.7,
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
            llm_response = await litellm_complete(
                model_name=self.model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                response_format=DataModel,
                #max_tokens=1000,
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

    def save(self, path: str) -> None:
        """
        Save the complete knowledge base to disk, including all sub-components

        Args:
            path (str): Path where the knowledge base will be saved
        """
        try:
            # Create a dictionary with all components that need to be saved
            data = {
                # Core components
                'chunks': list(self.chunks),
                'existing_hashes': self.existing_hashes,
                'index': faiss.serialize_index(self.index),
                'is_trained': self.is_trained,

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

            # Save to disk using pickle
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Knowledge base successfully saved to {path}")

        except Exception as e:
            print(f"Error saving knowledge base: {str(e)}")
            raise

    @classmethod
    def load(cls, path: str) -> 'KnowledgeBase':
        """
        Load a complete knowledge base from disk, including all sub-components

        Args:
            path (str): Path from where to load the knowledge base

        Returns:
            KnowledgeBase: A fully restored knowledge base instance
        """
        try:
            # Load data from disk
            with open(path, 'rb') as f:
                data = pickle.load(f)

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
            kb.chunks = deque(data['chunks'], maxlen=kb.chunks.maxlen)
            kb.existing_hashes = data['existing_hashes']
            kb.index = faiss.deserialize_index(data['index'])
            kb.is_trained = data['is_trained']

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

            print(f"Knowledge base successfully loaded from {path}")
            return kb

        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")
            raise

    def vis(self,output_file: str = "concept_graph.html", get_output=False):
        return GraphVisualizer.visualize(self.concept_extractor.concept_graph.convert_to_networkx(),
                                         output_file=output_file, get_output=get_output)

async def main():
    kb = KnowledgeBase(n_clusters=3)

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
    print("Total time from 7.57 to: 5.5 (one by one) ", t0)
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
    GraphVisualizer.visualize(nx_graph)
    kb.save("bas.pkl")

async def rgen():
    kb = KnowledgeBase.load("mem.plk")
    #print(await kb.forget_irrelevant(["lazy dog", "unimportant"], 0.51))
    print(await kb.query_concepts("AI"))
    print(await kb.retrieve("Evaluation metrics for assessing AI Agent performance"))
    print(kb.concept_extractor.concept_graph.concepts.keys())
    GraphVisualizer.visualize(kb.concept_extractor.concept_graph.convert_to_networkx(), output_file="concept_graph2.html")


if __name__ == "__main__":
    get_app(name="main2")

    asyncio.run(main())

