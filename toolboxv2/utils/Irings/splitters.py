from tqdm import tqdm
from toolboxv2 import Spinner
from .utils import ConceptSplitter, SplitConfig, SubConcept
from .zero import InputProcessor
from typing import List, Dict, Tuple, Set
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from nltk.tokenize import sent_tokenize
import networkx as nx


class SemanticConceptSplitter(ConceptSplitter):
    def __init__(self, input_processor: InputProcessor, config: SplitConfig = None):
        self.processor = input_processor
        self.config = config or SplitConfig()

    def split(self, text: str, context_vector: np.ndarray) -> List[SubConcept]:
        # Extract semantic units
        with Spinner("sent_tokenize 1/5"):
            sentences = sent_tokenize(text)
        with Spinner("sent_vectorize 2/5"):
            sentence_vectors = self._vectorize_sentences(sentences)

        # Create semantic graph
        with Spinner("Create semantic graph 3/5"):
            graph = self._build_semantic_graph(sentence_vectors)

        # Extract coherent chunks
        with Spinner("Extract coherent chunks 4/5"):
            chunks = self._extract_chunks(graph, sentences)
        with Spinner("Generate and rank subconcepts 5/5"):
            # Generate and rank subconcepts
            return self._create_subconcepts(chunks, context_vector)

    def _vectorize_sentences(self, sentences: List[str]) -> np.ndarray:
        vectors = []
        for sentence in sentences:
            if len(sentence.split()) >= 3:  # Skip very short sentences
                vector = self.processor.process_text(sentence)
                vectors.append(vector)
        return np.array(vectors)

    def _build_semantic_graph(self, vectors: np.ndarray) -> nx.Graph:
        graph = nx.Graph()

        # Add nodes
        for i in range(len(vectors)):
            graph.add_node(i, vector=vectors[i])

        # Add edges based on semantic similarity
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarity = np.dot(vectors[i], vectors[j])
                if similarity > self.config.similarity_threshold:
                    graph.add_edge(i, j, weight=similarity)

        return graph

    def _extract_chunks(self, graph: nx.Graph, sentences: List[str]) -> List[str]:
        # Find communities using Louvain method
        communities = nx.community.louvain_communities(graph)

        chunks = []
        current_chunk = []
        current_size = 0

        for community in communities:
            community_sentences = [sentences[i] for i in sorted(community)]
            community_text = ' '.join(community_sentences)
            community_size = len(community_text)

            if current_size + community_size <= self.config.max_chunk_size:
                current_chunk.extend(community_sentences)
                current_size += community_size
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = community_sentences
                current_size = community_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _create_subconcepts(
        self,
        chunks: List[str],
        context_vector: np.ndarray
    ) -> List[SubConcept]:
        subconcepts = []

        def helper(chunk):
            if len(chunk) < self.config.min_chunk_size:
                return

            # Process chunk
            chunk_vector = self.processor.process_text(chunk)
            importance = np.dot(chunk_vector, context_vector)

            if importance < self.config.min_importance:
                return

            # Find related concepts in chunk
            chunk_keywords = self._extract_keywords(chunk)
            relations = self._find_relations(chunk_vector, chunk_keywords)

            subconcepts.append(SubConcept(
                vector=chunk_vector,
                importance=importance,
                relations=relations,
                metadata={
                    "source_text": chunk,
                    "keywords": chunk_keywords
                }
            ))

        if len(chunks) == 1:
            helper(chunks[0])
        else:
            for chunk in tqdm(iterable=chunks, desc="Splitting chunks", total=len(chunks)):
                helper(chunk)

        return sorted(subconcepts, key=lambda x: x.importance, reverse=True)

    def _extract_keywords(self, text: str) -> Set[str]:
        # Implement keyword extraction using frequency analysis
        # This is a simplified version - consider using KeyBERT or similar
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1

        return set(sorted(word_freq, key=word_freq.get, reverse=True)[:10])

    def _find_relations(self, vector: np.ndarray, keywords: Set[str]) -> Set[str]:
        # Generate embeddings for keywords
        keyword_vectors = {
            word: self.processor.process_text(word)
            for word in keywords
        }

        # Find related concepts based on vector similarity
        relations = set()
        for word, word_vector in keyword_vectors.items():
            if np.dot(vector, word_vector) > self.config.similarity_threshold:
                relations.add(word)

        return relations


class TransformerSplitter(ConceptSplitter):
    def __init__(self, input_processor: InputProcessor):
        self.processor = input_processor

    def split(self, text: str, vector: np.ndarray) -> List[SubConcept]:
        # Split text into semantic chunks
        chunks = self._chunk_text(text)
        subconcepts = []

        for chunk in chunks:
            chunk_vector = self.processor.process_text(chunk)
            importance = np.dot(chunk_vector, vector)

            subconcepts.append(SubConcept(
                vector=chunk_vector,
                importance=importance,
                relations=set(),
                metadata={"source_text": chunk}
            ))

        return sorted(subconcepts, key=lambda x: x.importance, reverse=True)

    def _chunk_text(self, text: str) -> List[str]:
        # Implement semantic text chunking
        sentences = text.split('. ')
        return [s.strip() for s in sentences if s.strip()]

