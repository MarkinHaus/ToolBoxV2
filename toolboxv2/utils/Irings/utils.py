from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Set
from dataclasses import dataclass
from tqdm import tqdm
from toolboxv2 import Spinner
from nltk.tokenize import sent_tokenize
import networkx as nx

lang_dict = {
    'cs': 'czech',
    'da': 'danish',
    'de': 'german',
    'el': 'greek',
    'en': 'english',
    'es': 'spanish',
    'et': 'estonian',
    'fi': 'finnish',
    'fr': 'french',
    'it': 'italian',
    'nl': 'dutch',
    'no': 'norwegian',
    'pl': 'polish',
    'pt': 'portuguese',
    'ru': 'russian',
    'sl': 'slovene',
    'sv': 'swedish',
    'tr': 'turkish',
}
@dataclass
class SplitConfig:
    min_chunk_size: int = 50
    max_chunk_size: int = 2500
    similarity_threshold: float = 0.7
    min_importance: float = 0.3


@dataclass
class SubConcept:
    vector: np.ndarray
    importance: float
    relations: Set[str]
    metadata: Dict


class ConceptSplitter(ABC):
    @abstractmethod
    def split(self, text: str, vector: np.ndarray) -> List[SubConcept]:
        pass


class SemanticConceptSplitter(ConceptSplitter):
    def __init__(self, input_processor: 'InputProcessor', config: SplitConfig = None):
        self.processor = input_processor
        self.config = config or SplitConfig()
        print("Initializing", self.config)

    def split(self, text: str, context_vector: np.ndarray) -> List[SubConcept]:
        # Extract semantic units
        from langdetect import detect
        lang = lang_dict.get(detect(text), "english")
        sentences = sent_tokenize(text, lang)
        sentence_vectors = self._vectorize_sentences(sentences)

        # Create semantic graph
        graph = self._build_semantic_graph(sentence_vectors)

        # Extract coherent chunks
        chunks = self._extract_chunks(graph, sentences)

        subconcepts = self._create_subconcepts(chunks, context_vector)
        return subconcepts

    def _vectorize_sentences(self, sentences: List[str]) -> np.ndarray:
        vectors = []
        fist_len = 0
        for sentence in sentences:
            if len(sentence.split()) >= 3:  # Skip very short sentences
                if len(sentence) > 2500:
                    vector0 = self.processor.process_text(sentence[:1250])
                    if vector0 is not None and len(vector0) == 0:
                        fist_len = len(vector0)

                    if vector0 is not None and len(vector0) == fist_len:
                        vectors.append(vector0)
                    sentence = sentence[1250:]
                vector = self.processor.process_text(sentence)
                if vector is None:
                    print("Error processing v", len(vectors), len(sentences), len(sentence))
                    continue
                if len(vector) == 0:
                    fist_len = len(vector)

                if len(vector) == fist_len:
                    vectors.append(vector)
        return np.array(vectors)

    def _build_semantic_graph(self, vectors: np.ndarray) -> nx.Graph:
        """
        Builds a semantic graph from the given sentence vectors.

        Args:
        vectors (np.ndarray): Sentence vectors.

        Returns:
        nx.Graph: The built semantic graph.
        """
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
        if len(chunks) == 0:
            chunks = sentences
        return chunks

    def _create_subconcepts(
        self,
        chunks: List[str],
        context_vector: np.ndarray
    ) -> List[SubConcept]:
        subconcepts = []

        avg_min_importance = [0, 0]
        if context_vector is None:
            context_vector = self.processor.process_text('')

        def helper(chunk):
            if len(chunk) < self.config.min_chunk_size:
                return

            if len(chunk) > self.config.max_chunk_size:
                helper(chunk[self.config.max_chunk_size:])
                chunk = chunk[:self.config.max_chunk_size]

            # Process chunk
            chunk_vector = self.processor.process_text(chunk)
            if chunk_vector is None:
                return
            importance = np.dot(chunk_vector, context_vector)
            avg_min_importance[1] += 1
            if avg_min_importance[0] == 0:
                avg_min_importance[0] = importance
            else:
                avg_min_importance[0] += max(importance, 0)

            if importance < avg_min_importance[0]/avg_min_importance[1]:
                return

            # Find related concepts in chunk
            chunk_keywords = self._extract_keywords(chunk)
            # relations = self._find_relations(chunk_vector, chunk_keywords)

            subconcepts.append(SubConcept(
                vector=chunk_vector,
                importance=importance,
                relations=chunk_keywords,
                metadata={
                    "text": chunk,
                    "keywords": chunk_keywords
                }
            ))

        if len(chunks) == 1:
            helper(chunks[0])
        elif len(chunks) <= 10:
            for chunk in chunks:
                helper(chunk)
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
            for word in keywords[:10]
        }

        # Find related concepts based on vector similarity
        relations = set()
        for word, word_vector in keyword_vectors.items():
            if np.dot(vector, word_vector) > self.config.similarity_threshold:
                relations.add(word)

        return relations


class TransformerSplitter(ConceptSplitter):
    def __init__(self, input_processor: 'InputProcessor'):
        self.processor = input_processor

    def split(self, text: str, vector: np.ndarray) -> List[SubConcept]:
        # Split text into semantic chunks
        chunks = self._chunk_text(text)
        subconcepts = []
        for chunk in chunks:
            chunk_vector = self.processor.process_text(chunk)
            importance = np.dot(chunk_vector, vector) if vector is not None else -1

            subconcepts.append(SubConcept(
                vector=chunk_vector,
                importance=importance,
                relations=set(),
                metadata={"text": chunk, 'importance': ["hinge", "medium", "low"][0 if importance > 0.85 else (1 if importance > 0.55 else 2)]}
            ))

        return sorted(subconcepts, key=lambda x: x.importance, reverse=True)

    def _chunk_text(self, text: str) -> List[str]:
        # Implement semantic text chunking
        sentences = text.split('. ')
        return [s.strip() for s in sentences if s.strip()]

