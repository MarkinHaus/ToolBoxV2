import os
import json
import random
import re
import zlib
import spacy
import hashlib

import torch
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from pydantic import BaseModel, Field
from copy import deepcopy
from functools import lru_cache

import networkx as nx
import yaml
from fuzzywuzzy import fuzz

from collections import defaultdict  # For automatic dictionary initialization

from transformers import AutoTokenizer, AutoModel

from toolboxv2 import Spinner
from toolboxv2.utils.system import FileCache

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Mapping
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
import hashlib
import zlib
import json
from typing import List
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

EMBEDDING_DIM = 768


def detect_language(text: str) -> str:
    """
    Detect text language using a combination of character set analysis,
    common word detection, and statistical analysis.

    Args:
        text (str): Input text to analyze

    Returns:
        str: Detected language code ('en', 'de', 'zh', 'ja', etc.)
    """
    # Normalize text for analysis
    text = text.lower().strip()

    # Early return for empty text
    if not text:
        return 'en'

    # Character set detection for non-Latin scripts
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return 'zh'  # Chinese
    elif any("\u3040" <= char <= "\u30ff" for char in text):
        return 'ja'  # Japanese
    elif any("\u0400" <= char <= "\u04FF" for char in text):
        return 'ru'  # Russian
    elif any("\u0590" <= char <= "\u05FF" for char in text):
        return 'he'  # Hebrew

    # Common word detection for Latin script languages
    words = set(text.split())

    # German indicators
    german_indicators = {
        # Articles and pronouns
        'der', 'die', 'das', 'den', 'dem', 'des',
        'ein', 'eine', 'einer', 'eines', 'einem', 'einen',
        'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr',

        # Common verbs
        'ist', 'sind', 'war', 'waren', 'wird', 'werden',
        'kann', 'können', 'muss', 'müssen', 'hat', 'haben',

        # Prepositions
        'in', 'auf', 'mit', 'bei', 'seit', 'von', 'aus',
        'nach', 'zu', 'zur', 'zum', 'ins', 'im',

        # Common adjectives
        'gut', 'schlecht', 'groß', 'klein', 'neu', 'alt',

        # Common conjunctions
        'und', 'oder', 'aber', 'sondern', 'denn', 'weil'
    }

    # English indicators
    english_indicators = {
        # Articles and pronouns
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they',

        # Common verbs
        'is', 'are', 'was', 'were', 'will', 'would',
        'can', 'could', 'have', 'has', 'had', 'been',

        # Prepositions
        'in', 'on', 'at', 'to', 'for', 'with', 'by',
        'from', 'of', 'about', 'between', 'through',

        # Common conjunctions
        'and', 'or', 'but', 'because', 'if', 'when'
    }

    # Calculate language scores
    german_score = len(words.intersection(german_indicators))
    english_score = len(words.intersection(english_indicators))

    # Additional German language patterns
    if any(word.endswith(('ung', 'heit', 'keit', 'schaft', 'chen', 'lein')) for word in words):
        german_score += 2
    if any(word.endswith(('en', 'st', 'est', 'te', 'ten', 'tet')) for word in words):
        german_score += 1

    # Additional English patterns
    if any(word.endswith(('ing', 'ed', 'ly', 'tion', 'ment', 'ness')) for word in words):
        english_score += 2
    if any(word.endswith(('s', 'es', 'er', 'est')) for word in words):
        english_score += 1

    # Special case: If text contains umlauts or ß, highly likely to be German
    if any(char in text for char in 'äöüßÄÖÜ'):
        german_score += 3

    # Statistical analysis of character frequencies
    char_freq = {}
    total_chars = 0
    for char in text:
        if char.isalpha():
            char_freq[char] = char_freq.get(char, 0) + 1
            total_chars += 1

    if total_chars > 0:
        # German has higher frequencies of 'e', 'n', 'i', 's', 'r', 't'
        german_chars = {'e', 'n', 'i', 's', 'r', 't'}
        german_char_score = sum(char_freq.get(c, 0) / total_chars for c in german_chars)

        # English has higher frequencies of 'e', 't', 'a', 'o', 'i', 'n'
        english_chars = {'e', 't', 'a', 'o', 'i', 'n'}
        english_char_score = sum(char_freq.get(c, 0) / total_chars for c in english_chars)

        german_score += german_char_score * 2
        english_score += english_char_score * 2

    # Make final decision
    if german_score > english_score:
        return 'de'
    elif english_score > german_score:
        return 'en'
    else:
        # If scores are equal, default to English
        return 'en'


def _load_config(config_path: Optional[str]) -> dict:
    """Load configuration from YAML file or use defaults."""
    default_config = {
        'languages': ['en', 'de'],
        'importance_threshold': 0.3,
        'relation_threshold': 0.2,
        'fallback_similarity': 0.5,  # Added fallback similarity score
        'model_name': "sentence-transformers/all-MiniLM-L6-v2"
    }

    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return default_config


def _generate_explanation(snapshot: dict) -> str:
    """Generate detailed explanation of the semantic content."""
    parts = [f"Language: {snapshot['language'].upper()}", "\nKey Concepts:"]
    for concept in sorted(snapshot['concepts'], key=lambda x: x['importance'], reverse=True):
        s = f"- {concept['text']} (importance: {concept['importance']:.2f})"
        if s in parts:
            continue

        parts.append(s)

    parts.append("\nRelationships:") if len(snapshot['relations']) > 0 else None
    for relation in snapshot['relations']:
        e = '\n\t'.join(
            [f'{k}:{r}' for k, r in relation.items() if k not in ['target', 'source', 'type', 'has_vector']])
        s = f"- {relation['source']} --[{relation['type']}]--> {relation['target']} : extras {e}"
        if s in parts:
            continue
        parts.append(s)

    return "\n".join(parts)


def _generate_m_info(snapshot: Dict) -> str:
    """
    Generate a summary of the semantic representation.
    """
    concepts = snapshot['concepts']
    relations = snapshot['relations']

    info = f"Core Infos:\n"
    info += f"Number of Concepts: {len(concepts)}\n"
    info += f"Number of Relations: {len(relations)}\n"
    info += f"{snapshot['metadata']}"

    return info


def _generate_simplified(snapshot: dict) -> str:
    """Generate simplified version focusing on main concepts."""
    main_concepts = sorted(snapshot['concepts'],
                           key=lambda x: x['importance'],
                           reverse=True)[:max(3, len(snapshot['concepts']) // 50)]
    return "Main concepts: " + ", ".join(c['text'] for c in main_concepts)


def _generate_improvements(snapshot: dict) -> str:
    """
    Generate improvement suggestions and optionally process a mini-task on the semantic data.

    Args:
        snapshot (dict): The semantic snapshot containing concepts and relations

    Returns:
        str: Generated improvements and/or mini-task results
    """

    def mini_task_completion(data: dict) -> str:
        """
        Process a mini-task on the semantic data using task instructions.

        Args:
            data (dict): The semantic data to process

        Returns:
            str: Processed result based on task instructions
        """
        # Extract key components for processing
        concepts = {c['text']: c['importance'] for c in
                    (data['concepts'] if 'concepts' in data else data['sub_concepts'])}
        relations = {(r['source'], r['target']): r['strength'] for r in data['relations']}

        out = ""
        # Process different types of tasks
        top_concepts = sorted(concepts.items(), key=lambda x: x[1], reverse=True)[:3]
        out += f"\tKey concepts: {', '.join(c[0] for c in top_concepts)}\n"

        strong_relations = [
            f"{src} -> {tgt} (strength: {strength:.2f})"
            for (src, tgt), strength in relations.items()
            if strength > 0.5
        ]
        out += "\tStrong relationships:\n" + "\n\t\t-".join(strong_relations)

        isolated_concepts = [
            concept for concept in concepts
            if not any(concept in (src, tgt) for (src, tgt) in relations)
        ]
        out += f"\tIsolated concepts needing connections: {', '.join(isolated_concepts)}\n"

        improvements = []
        # Check concept depth
        if len(concepts) < 5:
            improvements.append("Add more detailed concepts")
        # Check relationship coverage
        coverage = len(relations) / (len(concepts) * (len(concepts) - 1))
        if coverage < 0.3:
            improvements.append("Increase relationship density")
        out += "\n\t".join(improvements)

        out += f"\tprocessed with {len(concepts)} concepts and {len(relations)} relationships\n"

        return out

    # Generate base improvements
    suggestions = ["Improvement suggestions:"]

    # Check concept coverage
    if (len(snapshot['concepts']) if 'concepts' in snapshot else len(snapshot['sub_concepts'])) < (
        3 if 'concepts' in snapshot else 2):
        suggestions.append("- Add more detailed concepts")

    # Check relation density
    if len(snapshot['relations']) < (
        len(snapshot['concepts']) if 'concepts' in snapshot else len(snapshot['sub_concepts'])):
        suggestions.append("- Clarify relationships between concepts")

    # Check concept importance distribution
    importance_values = [c['importance'] for c in
                         (snapshot['concepts'] if 'concepts' in snapshot else snapshot['sub_concepts'])]
    if max(importance_values) - min(importance_values) < 0.3:
        suggestions.append("- Differentiate concept importance more clearly")

    # Check relationship strength distribution
    if snapshot['relations']:
        strength_values = [r['strength'] for r in snapshot['relations']]
        if max(strength_values) - min(strength_values) < 0.3:
            suggestions.append("- Vary relationship strengths more distinctly")

    # Execute mini-task if provided
    # try:
    task_result = mini_task_completion(snapshot)
    suggestions.append(task_result)
    #except Exception as e:
    #    pass

    return "\n".join(suggestions)


def _extract_macro_concepts(doc):
    """
    Extract high-level, structurally significant concepts
    """
    macro_concepts = []
    for chunk in doc.noun_chunks:
        # Focus on longer, semantically rich noun chunks
        if len(chunk.text.split()) > 1:
            macro_concept = SemanticConcept(
                text=chunk.text,
                importance=len(chunk.text) / len(doc),
                pos=chunk.root.pos_,
                has_vector=chunk.root.has_vector,
                semantic_type=['macro', 'structural'],
            )
            macro_concepts.append(macro_concept.model_dump(mode='python'))
    return macro_concepts


def _extract_micro_concepts(doc):
    """
    Extract granular, precise semantic units
    """
    micro_concepts = []
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
            micro_concept = SemanticConcept(
                text=token.text,
                importance=token.prob,
                pos=token.pos_,
                has_vector=token.has_vector,
                semantic_type=['micro', 'precise'],
                linguistic_features={
                    'lemma': token.lemma_,
                    'dependency': token.dep_
                }
            )
            micro_concepts.append(micro_concept.model_dump(mode='python'))
    return micro_concepts


def _extract_semantic_concepts(doc):
    """
    Extract semantically rich, contextually significant concepts
    """
    semantic_concepts = []
    for ent in doc.ents:
        semantic_concept = SemanticConcept(
            text=ent.text,
            importance=1.0,  # Named entities are typically significant
            pos=ent.root.pos_,
            has_vector=ent.root.has_vector,
            semantic_type=['semantic', 'contextual', ent.label_],
            linguistic_features={
                'entity_type': ent.label_,
                'root_dependency': ent.root.dep_
            }
        )
        semantic_concepts.append(semantic_concept.model_dump(mode='python'))
    return semantic_concepts


def _extract_advanced_relations(doc):
    """
    Extract complex semantic relations with advanced attributes
    """
    relations = []
    for token in doc:
        if token.dep_ in ['nsubj', 'dobj', 'pobj', 'advmod', 'acomp', 'amod', 'compound', 'prep']:
            relation = SemanticRelation(
                source=token.text,
                target=token.head.text,
                type=token.dep_,
                strength=token.similarity(token.head) if token.has_vector and token.head.has_vector else 0.0,
                context_type='syntactic',
                semantic_distance=abs(token.i - token.head.i) / len(doc)
            )
            relations.append(relation.model_dump(mode='python'))
    return relations


def _calculate_semantic_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts
    """
    # Placeholder for advanced similarity calculation
    return len(set(text1.split()) & set(text2.split())) / len(set(text1.split()) | set(text2.split()))


def _calculate_semantic_complexity(doc):
    """
    Calculate semantic complexity score
    """
    complexity_factors = [
        len(doc),  # Total tokens
        len(doc.ents),  # Named entities
        len([chunk for chunk in doc.noun_chunks]),  # Noun chunks
        np.mean([token.prob for token in doc])  # Average token probability
    ]
    return np.mean(complexity_factors)


def _to_cossena_code(snapshot):
    """
    Convert semantic snapshot to classic Cossena code
    """
    # Convert snapshot to JSON
    json_str = json.dumps(snapshot)

    # Compress and convert to hex
    compressed = zlib.compress(json_str.encode('utf-8'))
    hex_code = compressed.hex()

    # Generate checksum
    checksum = hashlib.sha256(hex_code.encode()).hexdigest()[:8]

    return f"{checksum}:{hex_code}"


def clean_text(text: str) -> str:
    """Clean and normalize text before processing."""
    # Remove null bytes and normalize whitespace
    text = text.replace('\0', '')
    text = re.sub(r'\s+', ' ', text)

    # Remove any control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')

    return text.strip()

class CossenaCore:
    """Production-ready Cossena implementation with multi-language support and proper word vectors."""

    SUPPORTED_LANGUAGES = {
        'en': 'en_core_web_lg',  # Changed to large model with word vectors
        'de': 'de_core_news_lg',
        'ja': 'ja_core_news_lg',
        'zh': 'zh_core_web_lg'
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional config path for language models."""
        self.nlp_models = {}
        self.config = _load_config(config_path)
        self._initialize_models()
        self.graph = nx.DiGraph()
        cache_dir = os.getenv('APPDATA') if os.name == 'nt' else os.getenv('XDG_CONFIG_HOME') or os.path.expanduser(
            '~/.config') if os.name == 'posix' else "."
        cache = FileCache(folder=cache_dir + f'\\ToolBoxV2\\cache\\CossenaCore\\',
                          filename=cache_dir + f'\\ToolBoxV2\\cache\\CossenaCore\\cache.db')

        def helper(t, lang):
            result = cache.get(str((t, lang)))
            if result is not None:
                return result

            result = self.get_embedding(t)
            cache.set(str((t, lang)), result)
            return result

        self.get_concept_vector = helper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.get('model_name'))
            self.model = AutoModel.from_pretrained(self.config.get('model_name')).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @lru_cache(512)
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for input text.

        Args:
            text: Input text to embed

        Returns:
            numpy.ndarray or None: Embedding vector if successful
        """
        try:
            # Tokenize with truncation and padding
            encoded_input = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            # Perform pooling
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            # Normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            # Convert to numpy array
            return sentence_embeddings.cpu().numpy()[0]

        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None

    def batch_get_embeddings(self, texts: list[str], batch_size: int = 32) -> Optional[np.ndarray]:
        """
        Get embeddings for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once

        Returns:
            numpy.ndarray or None: Matrix of embedding vectors if successful
        """
        try:
            embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)

                # Compute token embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)

                # Perform pooling and normalize
                batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                embeddings.append(batch_embeddings.cpu().numpy())

            return np.vstack(embeddings)

        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            return None

    def _initialize_models(self):
        """Initialize language models based on config."""
        for lang in self.config['languages']:
            if lang in self.SUPPORTED_LANGUAGES:
                try:
                    self.nlp_models[lang] = spacy.load(self.SUPPORTED_LANGUAGES[lang])
                except OSError:
                    print(f"Downloading language model for {lang}...")
                    spacy.cli.download(self.SUPPORTED_LANGUAGES[lang])
                    self.nlp_models[lang] = spacy.load(self.SUPPORTED_LANGUAGES[lang])

    def _calculate_similarity(self, token1, token2) -> float:
        """Calculate similarity between tokens with fallback for missing vectors."""
        try:
            if token1.has_vector and token2.has_vector:
                return token1.similarity(token2)
            else:
                # Fallback similarity based on dependency relationship
                if token1.dep_ in ['nsubj', 'dobj', 'pobj', 'advmod', 'acomp', 'amod', 'compound',
                                   'prep'] and token2 == token1.head:
                    return self.config['fallback_similarity']
                return 0.0
        except RuntimeWarning:
            return self.config['fallback_similarity']

    def text_to_code(self, text: str) -> str:
        """Convert input text to hexadecimal Cossena code with improved similarity handling."""
        #try:

        doc = self.get_vec(text)
        concepts = []
        relations = []

        # Process noun chunks for concepts
        for chunk in doc.noun_chunks:
            # Use vector norm if available, otherwise use length-based importance
            importance = (chunk.root.vector_norm if chunk.root.has_vector
                          else len(chunk.text) / len(doc))

            if importance > self.config['importance_threshold']:
                concepts.append({
                    'text': chunk.text,
                    'importance': float(importance),
                    'pos': chunk.root.pos_,
                    'has_vector': chunk.root.has_vector
                })
                self.graph.add_node(chunk.text, weight=importance)

        # Process dependencies for relations with improved similarity calculation
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj', 'advmod', 'acomp', 'amod', 'compound', 'prep']:
                if token.head.text != token.text:
                    strength = self._calculate_similarity(token, token.head)
                    if strength > self.config['relation_threshold']:
                        relations.append({
                            'source': token.text,
                            'target': token.head.text,
                            'type': token.dep_,
                            'strength': float(strength)
                        })
                        self.graph.add_edge(token.text, token.head.text,
                                            weight=strength, type=token.dep_)

        concepts += _extract_macro_concepts(doc)
        concepts += _extract_micro_concepts(doc)

        # Extract relations
        relations += _extract_advanced_relations(doc)

        snapshot = {
            'language': detect_language(text),
            'concepts': concepts,
            'relations': relations,
            'metadata': {
                'version': '1.0',
                'semantic_complexity': _calculate_semantic_complexity(doc),
                'has_vectors': any(c['has_vector'] for c in concepts)
            }
        }
        return _to_cossena_code(snapshot)
        #except RuntimeError as e:
        # #   print("Error coding to cossena :", e)
        #    return ""

    def transmutation(self, hex_code: str, mode: str = 'explain') -> str or dict:
        """
            Convert hex code back to semantic representation and perform requested transmutation.

            Modes:
            - explain: Detailed explanation
            - raw: Basic string representation
            - graph: Graph representation as string
            - simplify: Simplified explanation
            - improve: Suggested improvements
            - info: gnarl infos metadata
            """
        #try:
        # Verify checksum
        checksum, data = hex_code.split(':')
        if checksum != hashlib.sha256(data.encode()).hexdigest()[:8]:
            return "Error: Invalid checksum"

        # Decompress and load data
        compressed = bytes.fromhex(data)
        json_str = zlib.decompress(compressed).decode('utf-8')
        snapshot = json.loads(json_str)

        # Rebuild graph
        self.graph = nx.DiGraph()
        for concept in snapshot['concepts']:
            self.graph.add_node(concept['text'], weight=concept['importance'])
        for relation in snapshot['relations']:
            self.graph.add_edge(relation['source'], relation['target'],
                                weight=relation['strength'], type=relation['type'])

        # Process according to mode
        if mode == 'info':
            return _generate_m_info(snapshot)
        elif mode == 'row':
            return snapshot
        elif mode == 'explain':
            return _generate_explanation(snapshot)
        elif mode == 'graph':
            return self._generate_graph_representation()
        elif mode == 'simplify':
            return _generate_simplified(snapshot)
        elif mode == 'improve':
            return _generate_improvements(snapshot)
        elif mode == 'llm':
            return self._generate_llm_representation(snapshot)

        else:
            return "Error: Invalid transmutation mode :" + mode

    #except RuntimeError as e:
    #    return f"Error processing hex code: {str(e)}"

    def _generate_llm_representation(self, snapshot: dict) -> str:
        """Generate a compact, structured representation for LLMs using Markdown."""
        output = ["The Cossena code is a compressed and structured representation",
                  f"# Analysis ({snapshot['language']})",
                  f"### Complexity: {snapshot['metadata']['semantic_complexity']:.2f}",
                  f"### Density: {nx.density(self.graph):.2f}", "", "## Key Concepts | [Concept] (Importance)"]

        # Language and Metadata

        # Key Concepts
        sorted_concepts = sorted(snapshot['concepts'], key=lambda x: x['importance'], reverse=True)
        for concept in sorted_concepts[:len(self.graph.nodes) // 2]:  # Top 5 concepts
            s = f"- {concept['text']} ({concept['importance']:.2f})"
            if s not in output:
                output.append(s)
            if 5 < sorted_concepts.index(concept) and concept['importance'] < 1:
                break
        output.append("")

        # Main Relations
        output.append("## Primary Relations | [source] [type] [target] (Strength)")
        sorted_relations = sorted(snapshot['relations'], key=lambda x: x['strength'], reverse=True)

        i = 0
        eh = len(self.graph.edges()) // 2
        while len(sorted_relations) > eh:
            if i >= len(sorted_relations):
                i = 0
            relation = sorted_relations[i]
            s = f"- {relation['source']} {relation['type']} {relation['target']} ({relation['strength']:.2f})"
            if s not in output:
                output.append(s)
            sorted_relations.remove(relation)
            i += 1
        output.append("")

        if len(self.graph.nodes) * len(self.graph.edges()) > 0:
            # Central Concepts
            output.append(_generate_simplified(snapshot).replace("Main concepts: ", "## Central Concepts"))

        output.append("\n * END Cossena code END * \n___\n")

        return "\n".join(output)

    def _generate_graph_representation(self) -> str:
        """Generate ASCII graph representation."""
        lines = ["Semantic Graph:", ""]

        # Sort nodes by importance
        nodes = sorted(self.graph.nodes(data=True), key=lambda x: x[1].get('weight', 0), reverse=True)

        # Generate ASCII representation
        for node, data in nodes:
            lines.append(f"{node}")
            for _, target, edge_data in self.graph.edges(node, data=True):
                lines.append(f"  └─── {edge_data.get('type', '')} ───> {target}")

        return "\n".join(lines)

    def analyze_semantic_relations(self, *hex_codes: str, threshold: float = 0.6) -> Dict:
        """
        Perform comprehensive semantic analysis across multiple hex codes to identify related concepts,
        common key aspects, niche ideas, and conflicting information.

        Args:
            *hex_codes: Variable number of hex codes to analyze
            threshold: Similarity threshold for grouping (default 0.6)

        Returns:
            Dict: Comprehensive semantic analysis results
        """

        def extract_concepts_and_vectors(hex_codes_):
            processed_data = []
            for code in hex_codes_:
                snapshot = self.transmutation(code, 'row')
                for concept in snapshot['concepts']:
                    processed_data.append({
                        'code_id': code,
                        'concept': concept,
                        'lang': snapshot['language'],
                        'vector': self.get_concept_vector(concept['text'], snapshot['language'])
                    })
            return processed_data

        def build_concept_network(processed_data_):
            G = nx.Graph()
            for i, data in enumerate(processed_data_):
                G.add_node(i, **data)
            return G

        def compute_pairwise_similarities(G):
            for u, v in combinations(G.nodes(), 2):
                similarity = cosine_similarity(G.nodes[u]['vector'].reshape(1, -1),
                                               G.nodes[v]['vector'].reshape(1, -1))[0][0]
                G.add_edge(u, v, weight=similarity)
            return G

        def identify_key_concepts(G, centrality_threshold=threshold):
            centralities = nx.eigenvector_centrality(G, weight='weight')
            return [node for node, centrality in centralities.items() if centrality > centrality_threshold + 0.1]

        def find_common_themes(G, key_concepts):
            communities = list(nx.community.greedy_modularity_communities(G, weight='weight'))
            themes = []
            for community in communities:
                theme_concepts = [G.nodes[n]['concept']['text'] for n in community if n in key_concepts]
                if theme_concepts:
                    themes.append(theme_concepts)
            return themes

        def detect_niche_ideas(G, key_concepts):
            niche_ideas = []
            for node in G.nodes():
                if node not in key_concepts:
                    neighbors = list(G.neighbors(node))
                    if len(neighbors) < 3:  # Arbitrary threshold for "niche"
                        niche_ideas.append(G.nodes[node]['concept']['text'])
            return niche_ideas

        def identify_conflicts(G):
            conflicts = []
            for u, v in G.edges():
                if .48 <= G.edges[u, v]['weight'] <= .52 or .22 <= G.edges[u, v]['weight'] <= .25:
                    conflicts.append((G.nodes[u]['concept']['text'], G.nodes[v]['concept']['text']))
            return conflicts

        # Main analysis flow
        processed_data = extract_concepts_and_vectors(hex_codes)
        concept_network = build_concept_network(processed_data)
        concept_network = compute_pairwise_similarities(concept_network)

        key_concepts = identify_key_concepts(concept_network)
        common_themes = find_common_themes(concept_network, key_concepts)
        niche_ideas = detect_niche_ideas(concept_network, key_concepts)
        conflicts = identify_conflicts(concept_network)

        return {
            'key_concepts': [concept_network.nodes[n]['concept']['text'] for n in key_concepts],
            'common_themes': common_themes,
            'niche_ideas': niche_ideas,
            'conflicts': conflicts,
            'metadata': {
                'total_concepts': len(concept_network.nodes()),
                'total_connections': len(concept_network.edges()),
                'average_similarity': sum(d['weight'] for _, _, d in concept_network.edges(data=True)) / len(
                    concept_network.edges())
            }
        }

    def semantic_similarity(self, concept1: Dict, concept2: Dict) -> float:
        """
        Advanced semantic similarity using multiple techniques

        Args:
            concept1 (Dict): First concept dictionary
            concept2 (Dict): Second concept dictionary

        Returns:
            float: Similarity score between 0 and 1
        """
        # Text normalization
        text1 = normalize_text(concept1['text'])
        text2 = normalize_text(concept2['text'])

        if not text1 and not text2:
            return -1
        if len(text1.rstrip()) > 1 and not len(text2.rstrip()) > 1:
            return -1
        # If texts are exactly the same
        if text1 == text2:
            return 1.0

        # Fuzzy string matching
        fuzzy_score = fuzz.ratio(text1, text2) / 100.0

        return float(fuzzy_score)

    def is_semantically_similar(self, concept1: Dict, concept2: Dict, threshold: float = 0.7) -> bool:
        """
        Determine if two concepts are semantically similar

        Args:
            concept1 (Dict): First concept dictionary
            concept2 (Dict): Second concept dictionary
            threshold (float): Similarity threshold for considering concepts similar

        Returns:
            bool: Whether concepts are semantically similar
        """
        similarity = self.semantic_similarity(concept1, concept2)
        return similarity >= threshold

    @lru_cache(512)
    def get_vec(self, text, lang=None):
        if lang is None:
            lang = detect_language(text)
        return self.nlp_models.get(lang, self.nlp_models['en'])(text)


def format_semantic_analysis(analysis_results: Dict) -> str:
    """
    Formats the output of analyze_semantic_relations into a human-readable string.

    Args:
        analysis_results (Dict): The result dictionary from analyze_semantic_relations.

    Returns:
        str: A formatted string representing the analysis.
    """
    # Extracting the key concepts, common themes, niche ideas, conflicts, and metadata from the analysis result
    key_concepts = analysis_results.get('key_concepts', [])
    common_themes = analysis_results.get('common_themes', [])
    niche_ideas = analysis_results.get('niche_ideas', [])
    conflicts = analysis_results.get('conflicts', [])
    metadata = analysis_results.get('metadata', {})

    # Formatting the key concepts
    formatted_key_concepts = "Key Concepts:\n"
    for concept in key_concepts:
        formatted_key_concepts += f"  - {concept}\n"

    # Formatting the common themes
    formatted_themes = "\nCommon Themes:\n"
    for i, theme in enumerate(common_themes, 1):
        formatted_themes += f"  Theme {i}: {', '.join(theme)}\n"

    # Formatting the niche ideas
    formatted_niche_ideas = "\nNiche Ideas:\n"
    for idea in niche_ideas:
        formatted_niche_ideas += f"  - {idea}\n"

    # Formatting the conflicts
    formatted_conflicts = "\nConflicts:\n"
    for conflict in conflicts:
        formatted_conflicts += f"  - {conflict[0]} vs {conflict[1]}\n"

    # Formatting the metadata
    formatted_metadata = "\nMetadata:\n"
    formatted_metadata += f"  Total Concepts: {metadata.get('total_concepts', 0)}\n"
    formatted_metadata += f"  Total Connections: {metadata.get('total_connections', 0)}\n"
    formatted_metadata += f"  Average Similarity: {metadata.get('average_similarity', 0):.2f}\n"

    # Combine everything into a single string
    formatted_analysis = (formatted_key_concepts + formatted_themes +
                          formatted_niche_ideas + formatted_conflicts +
                          formatted_metadata)

    return formatted_analysis


def _update_relations(combined: Dict, new: Dict) -> None:
    """
    Update relations after concept merging.
    """
    # Build mapping of merged concepts
    concept_mapping = {
        c['text']: c['metadata'].get('merged_from', [])
        for c in combined['concepts']
        if 'metadata' in c and 'merged_from' in c['metadata']
    }

    # Update relations
    for relation in new['relations']:
        source = relation['source']
        target = relation['target']

        # Find merged concepts
        source_merged = next(
            (k for k, v in concept_mapping.items()
             if source in v or source == k),
            source
        )
        target_merged = next(
            (k for k, v in concept_mapping.items()
             if target in v or target == k),
            target
        )

        # Add updated relation
        if source_merged != target_merged:
            combined['relations'].append({
                'source': source_merged,
                'target': target_merged,
                'type': relation['type'],
                'strength': relation['strength']
            })


def _merge_concept_pair(existing: Dict, new: Dict) -> None:
    """
    Merge two concepts, combining their properties intelligently.
    """
    # Average importance scores
    existing['importance'] = (
                                 existing['importance'] + new['importance']
                             ) / 2

    # Combine metadata
    if 'metadata' not in existing:
        existing['metadata'] = {}

    existing['metadata']['merged_from'] = list(set(existing['metadata'].get(
        'merged_from', []
    ) + [new['text']]))

    # Update vector if available
    if 'vector' in existing and 'vector' in new:
        existing['vector'] = (
                                 np.array(existing['vector']) +
                                 np.array(new['vector'])
                             ) / 2


def _merge_temporal_versions(old: Dict, new: Dict) -> Dict:
    """
    Merge two temporal versions of a concept.
    """
    merged = old.copy()

    # Update temporal metadata
    merged['temporal_metadata'] = {
        'added': new['temporal_metadata']['added'],
        'version': new['temporal_metadata']['version'],
        'previous_versions': merged['temporal_metadata'].get(
            'previous_versions', []
        ) + [{
            'added': old['temporal_metadata']['added'],
            'version': old['temporal_metadata']['version']
        }]
    }

    # Merge other properties
    merged['importance'] = max(
        old['importance'],
        new['importance']
    )

    return merged


def _finalize_combination(combined: Dict) -> str:
    """
    Finalize the combined snapshot and convert to hex code.
    """
    # Add combination metadata
    combined['metadata'].update({
        'combination_timestamp': datetime.now().isoformat(),
        'total_concepts': len(combined['concepts']),
        'total_relations': len(combined['relations']),
    })

    # Convert to hex code
    return _to_cossena_code(combined)


def _resolve_temporal_conflicts(combined: Dict) -> None:
    """
    Resolve conflicts between concept versions.
    """
    # Group concepts by their base text
    concept_versions = defaultdict(list)
    for concept in combined['concepts']:
        base_text = concept['text']
        concept_versions[base_text].append(concept)

    # Resolve conflicts for each concept
    resolved_concepts = []
    for base_text, versions in concept_versions.items():
        if len(versions) > 1:
            # Sort by version number
            versions.sort(
                key=lambda x: x['temporal_metadata']['version']
            )

            # Merge sequential versions
            resolved = versions[0]
            for next_version in versions[1:]:
                resolved = _merge_temporal_versions(
                    resolved, next_version
                )

            resolved_concepts.append(resolved)
        else:
            resolved_concepts.append(versions[0])

    combined['concepts'] = resolved_concepts


def _temporal_merge(combined: Dict, new: Dict) -> None:
    """
    Merge concepts with temporal awareness and versioning.
    """
    timestamp = datetime.now().isoformat()

    for concept in new['concepts']:
        concept['temporal_metadata'] = {
            'added': timestamp,
            'version': float(combined['metadata']['version']) + 1
        }
        combined['concepts'].append(concept)

    _resolve_temporal_conflicts(combined)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


class CossenaPipeline(CossenaCore):

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)

    def pre_process(self, hex_code: str, additional_text: str) -> Dict:
        # Decode existing code
        existing_snapshot = self.transmutation(hex_code, 'row')

        # Extract new concepts and relations from additional text
        new_snapshot_code = self.text_to_code(additional_text)
        new_snapshot = self.transmutation(new_snapshot_code, 'row')

        try:
            analyze = self.analyze_semantic_relations(hex_code, new_snapshot_code)
        except RuntimeError as e:
            print("Error processing analysis", e)
            analyze = {}
        return {
            'base_code': existing_snapshot,
            'extend_code': new_snapshot,
            'analyze': analyze,
        }

    def rebuild_code(self, base_code: Dict or str, extend_code: Dict or str, similarity_threshold=0.5, **kwargs) -> str:

        if base_code is None and extend_code is not None:
            return extend_code
        if extend_code is None and base_code is not None:
            return base_code
        if extend_code is None and base_code is None:
            raise ValueError("Noe codes Provided")
            #try:
            if isinstance(base_code, str):
                base_code = self.transmutation(base_code, 'row')
            if isinstance(extend_code, str):
                extend_code = self.transmutation(extend_code, 'row')
            # Merge existing and new snapshots
            if len(extend_code['concepts']) == 0 and len(base_code['concepts']) != 0:
                return _to_cossena_code(base_code)
            if len(base_code['concepts']) == 0 and len(extend_code['concepts']) != 0:
                return _to_cossena_code(extend_code)
            if len(base_code['concepts']) == 0 and len(extend_code['concepts']) == 0:
                return _to_cossena_code(base_code)
            merged_snapshot = self.merge_snapshots(base_code, extend_code, similarity_threshold)

            # Generate new Cossena code
            return _to_cossena_code(merged_snapshot)
        #except RuntimeError as e:
        #    print("Error in rebuild_code", e)

    def combine_codes(self, *codes: str, strategy: str = "", threshold: float = .6) -> str:
        combined_concepts = []
        combined_relations = []
        new_codes = list(codes)

        while len(new_codes) > 1:
            for code in codes:
                if code.count(':') != 1:
                    new_codes.remove(code)
                    continue
                snapshot = self.transmutation(code, 'row')

                if strategy == "sigel":
                    if code == new_codes[0]:
                        continue
                    new_codes.append(self.rebuild_code(new_codes[0], code))
                    new_codes.remove(code)
                elif strategy == "pairs":
                    new_codes.remove(code)
                    r_pair = random.choice(new_codes)
                    new_codes.remove(r_pair)
                    new_codes.append(self.rebuild_code(code, r_pair))
                elif strategy.startswith("sup"):
                    for p_code in codes:
                        new_codes.append(self.rebuild_code(code, p_code))
                else:
                    combined_concepts.extend(snapshot['concepts'])
                    combined_relations.extend(snapshot['relations'])
                    new_codes.remove(code)

            if strategy.startswith("sup"):
                strategy = strategy[3:]

            codes = new_codes

        if len(new_codes) == 1:
            return self.cleanup(new_codes[0], threshold)

        combined_concepts = self.remove_similar_content(combined_concepts, combined_concepts, threshold)
        combined_relations = self.remove_similar_relations(combined_relations, combined_relations)

        combined_snapshot = {
            'language': 'multi',
            'concepts': combined_concepts,
            'relations': combined_relations,
            'metadata': {
                'total_concepts': len(combined_concepts),
                'total_relations': len(combined_relations),
                'word_count': sum([len(concept['text']) for concept in combined_concepts]),
                'combination_strategy': strategy,
                'combination_timestamp': datetime.now().isoformat(),
            }
        }

        new_code = _to_cossena_code(combined_snapshot)
        return new_code

    def cleanup(self, snapshot_code, threshold=0.8):
        #try:
        if isinstance(snapshot_code, str):
            snapshot_code = self.transmutation(snapshot_code, 'row')
        snapshot_code['concepts'] = self.remove_similar_content(snapshot_code['concepts'],
                                                                snapshot_code['concepts'],
                                                                threshold)
        snapshot_code['relations'] = self.remove_similar_relations(snapshot_code['relations'],
                                                                   snapshot_code['relations'])

        return _to_cossena_code(snapshot_code)
        #except Exception as e:
        #    print("Failed to cleanup snapshot", e)

    @lru_cache(maxsize=128)
    def process_path(self, snapshot_code: Union[Dict, str],
                     traversal_method: str = 'dfs',
                     similarity_threshold: float = 0.6) -> List[str]:
        """
        Process semantic paths through a graph with advanced traversal and similarity

        Args:
            snapshot_code (Dict or str): Snapshot to process
            traversal_method (str): Graph traversal method ('dfs', 'bfs')
            similarity_threshold (float): Semantic similarity threshold

        Returns:
            List[str]: Processed semantic path
        """
        # Handle string input via transmutation (assumed existing method)
        #try:
        if isinstance(snapshot_code, str):
            snapshot_code = self.transmutation(snapshot_code, 'row')

        # Create semantic graph
        graph = self.__build_semantic_graph(snapshot_code)

        # Select traversal method
        traversal_methods = {
            'dfs': self._depth_first_traversal,
            'bfs': self._breadth_first_traversal
        }

        # Default to DFS if invalid method specified

        with Spinner(f"Process path : {traversal_method} Volumen {len(graph)}"):
            traverse_func = traversal_methods.get(traversal_method, self._depth_first_traversal)

            return traverse_func(graph, snapshot_code['concepts'], similarity_threshold)
        # except RuntimeError as e:
        #    print("Error processing cossena path", e)
        #    return []

    def remove_concepts_and_ideas(self, base_code: Dict or str, remove_code: Dict or str,
                                  threshold: float = 0.75) -> str:
        """
        Remove concepts and ideas from a given base code based on a remove code.

        Args:
            base_code (Dict): Base code from which concepts and ideas will be removed
            remove_code (Dict): Code containing concepts and ideas to be removed
            threshold (float): Similarity threshold for concept merging

        Returns:
            Dict: New code after removing concepts and ideas
        """

        # try:
        if isinstance(base_code, str):
            base_code = self.transmutation(base_code, 'row')
        if isinstance(remove_code, str):
            remove_code = self.transmutation(remove_code, 'row')
        # Initialize new code
        new_code = base_code.copy()

        # Remove concepts
        new_code['concepts'] = [concept for concept in new_code['concepts']
                                if not self.is_semantically_similar(concept, remove_code['concepts'], threshold)]

        # Remove relations
        new_code['relations'] = [relation for relation in new_code['relations']
                                 if not self._is_duplicate_relation(relation, remove_code['relations'], threshold)]

        return _to_cossena_code(base_code)
        #except Exception as e:
        #    print("Failed to remove concepts and ideas", e)

    def merge_snapshots(self, old_snapshot: Dict, new_snapshot: Dict, threshold) -> Dict:
        """
        Merge two snapshots intelligently, maintaining semantic relationships

        Args:
            old_snapshot (Dict): Previous snapshot of concepts and relations
            new_snapshot (Dict): New snapshot to be merged
            threshold (float between 0 and 1): New snapshot to be merged

        Returns:
            Dict: Merged comprehensive snapshot
        """
        # Deep copy to avoid modifying original snapshots
        merged_snapshot = {
            'language': old_snapshot["language"] if old_snapshot["language"] == new_snapshot["language"] else "multi",
            'concepts': deepcopy(old_snapshot.get('concepts', [])),
            'relations': deepcopy(old_snapshot.get('relations', [])),
            'metadata': {
                **old_snapshot.get('metadata', {}),
                **new_snapshot.get('metadata', {}),
                'merge_timestamp': datetime.now().isoformat(),
            }
        }

        # Cross-reference and merge concepts
        merged_concepts = self._merge_concepts(
            merged_snapshot['concepts'],
            new_snapshot.get('concepts', []), threshold
        )
        merged_snapshot['concepts'] = merged_concepts

        # Merge and reconcile relations
        merged_relations = self._merge_relations(
            merged_snapshot['relations'],
            new_snapshot.get('relations', []),
            merged_concepts, threshold
        )
        merged_snapshot['relations'] = merged_relations

        merged_snapshot["metadata"]['total_concepts'] = len(merged_concepts)
        merged_snapshot["metadata"]['total_relations'] = len(merged_relations)
        merged_snapshot["metadata"]['word_count'] = sum([len(x['text']) for x in merged_concepts])

        return merged_snapshot

    def split_code(self, hex_code: str, traversal_method: str = 'dfs', similarity_threshold: float = 0.6,
                   max_concepts_per_code: int = 10, max_codes: int = -1) -> List[str]:
        """
        Split a Cossena code into groups based on semantic relationships

        Args:
            hex_code (str): Cossena code to split
            traversal_method (str): Graph traversal method ('dfs' or 'bfs')
            similarity_threshold (float): Semantic similarity threshold
            max_concepts_per_code (int): Maximum number of concepts per new Cossena code
            max_codes (int): Maximum number of returned codes -1 is all

        Returns:
                List[List[str]]: List of semantic groups
        """
        # Decode existing code
        existing_snapshot = self.transmutation(hex_code, 'row')

        concept_max = len(existing_snapshot['concepts'])
        if max_concepts_per_code >= concept_max:
            max_codes = 2
            max_concepts_per_code = concept_max // 2

        if max_concepts_per_code < 1:
            max_concepts_per_code = 1

        if max_codes * max_concepts_per_code > concept_max:
            max_codes = concept_max // 2

        # Build semantic graph
        graph = self.__build_semantic_graph(existing_snapshot)

        # Get concepts from existing snapshot
        concepts = existing_snapshot.get('concepts', [])

        # Traverse the graph based on the specified method
        traversal_methods = {
            'dfs': self._depth_first_traversal,
            'bfs': self._breadth_first_traversal
        }

        # Default to DFS if invalid method specified
        traverse_func = traversal_methods.get(traversal_method, self._depth_first_traversal)

        semantic_groups = []
        num_codes = 0

        while num_codes < max_codes:
            current_concepts = concepts[:max_concepts_per_code]
            concepts = concepts[max_concepts_per_code:]
            if len(current_concepts) == 0:
                max_concepts_per_code += 1
                continue
            current_semantic_groups = traverse_func(graph, current_concepts, similarity_threshold)
            semantic_groups.append(current_semantic_groups)

            num_codes += 1
        new_codes = []
        for groups in semantic_groups:
            new_codes.append(self.combine_codes(hex_code, self.text_to_code(' '.join(groups))))
        return new_codes

    #... helper functions

    def remove_similar_content(self, existing_concepts: List, new_concepts: List, threshold: float = 0.7) -> List:
        """
        Remove similar content from a new snapshot based on semantic similarity with existing snapshot

        Args:
            existing_concepts (List): Existing snapshot containing concepts
            new_concepts (List): New snapshot containing concepts to be filtered
            threshold (float): Similarity threshold for considering concepts similar

        Returns:
            List: New content with similar content removed
        """
        filtered_concepts = []

        for new_concept in new_concepts:
            is_similar = False

            for existing_concept in existing_concepts:
                if existing_concept == new_concept:
                    continue
                if self.is_semantically_similar(new_concept, existing_concept, threshold):
                    is_similar = True
                    break

            if not is_similar:
                filtered_concepts.append(new_concept)

        return filtered_concepts

    def remove_similar_relations(self, existing_relations: List, new_relations: List) -> List:
        """
        Remove similar relations from a new snapshot based on semantic similarity with existing snapshot

        Args:
            existing_relations (List): Existing snapshot containing relations
            new_relations (List): New snapshot containing relations to be filtered
            threshold (float): Similarity threshold for considering relations similar

        Returns:
            List: New relations with similar relations removed
        """

        for new_relation in new_relations:
            if not self._is_duplicate_relation(new_relation, existing_relations):
                existing_relations.append(new_relation)

        return existing_relations

    @staticmethod
    def __build_semantic_graph(snapshot_code: Dict) -> nx.DiGraph:
        """
        Construct a semantic graph with advanced concept linking

        Args:
            snapshot_code (Dict): Snapshot containing concepts and relations

        Returns:
            nx.DiGraph: Semantic graph
        """
        graph = nx.DiGraph()

        # Add concepts as nodes
        for concept in snapshot_code['concepts']:
            graph.add_node(concept['text'], **concept)

        # Add relations as edges
        for relation in snapshot_code['relations']:
            graph.add_edge(
                relation['source'],
                relation['target'],
                **{k: v for k, v in relation.items() if k not in ['source', 'target']}
            )

        return graph

    def _depth_first_traversal(self,
                               graph: nx.DiGraph,
                               concepts: List[Dict],
                               similarity_threshold: float, max_neighbors=100) -> List[str]:
        """
        Perform depth-first semantic traversal

        Args:
            graph (nx.DiGraph): Semantic graph
            concepts (List[Dict]): Concept list
            similarity_threshold (float): Semantic similarity threshold

        Returns:
            List[str]: Semantic path
        """
        path = []
        visited = set()

        def semantic_dfs(current_concept):
            if current_concept in visited:
                return

            visited.add(current_concept)
            path.append(current_concept)

            # Find semantically similar neighbors
            neighbors = list(graph.neighbors(current_concept))
            similar_neighbors = [
                n for n in neighbors
                if self.is_semantically_similar(
                    {'text': current_concept},
                    {'text': n},
                    threshold=similarity_threshold
                )
            ]

            for neighbor in sorted(similar_neighbors, key=lambda x: len(graph.edges(x)))[:max_neighbors]:
                semantic_dfs(neighbor)

        # Start from concept with most connections

        def hlper(d):
            if isinstance(d, int):
                return d
            if isinstance(d, float):
                return d
            if isinstance(d, list) and len(d) == 0:
                return -1
            if isinstance(d, list) and len(d) == 1:
                _, d = d[0]
                return hlper(d)
            if isinstance(d, list) and len(d) > 1:
                _, d = d[-1]
                return hlper(d)
            return -1

        start_concept = max(concepts, key=lambda c: hlper(graph.degree(c['text'])))['text']
        semantic_dfs(start_concept)

        return path

    def _breadth_first_traversal(self,
                                 graph: nx.DiGraph,
                                 concepts: List[Dict],
                                 similarity_threshold: float, max_neighbors=100) -> List[str]:
        """
        Perform breadth-first semantic traversal

        Args:
            graph (nx.DiGraph): Semantic graph
            concepts (List[Dict]): Concept list
            similarity_threshold (float): Semantic similarity threshold

        Returns:
            List[str]: Semantic path
        """
        from collections import deque

        path = []
        visited = set()
        queue = deque()

        # Start from concept with most connections
        start_concept = max(concepts, key=lambda c: graph.degree(c['text']))['text']
        queue.append(start_concept)

        while queue:
            current_concept = queue.popleft()

            if current_concept in visited:
                continue

            visited.add(current_concept)
            path.append(current_concept)

            # Find semantically similar neighbors
            neighbors = list(graph.neighbors(current_concept))[:max_neighbors]
            similar_neighbors = [
                n for n in neighbors
                if self.is_semantically_similar(
                    {'text': current_concept},
                    {'text': n},
                    threshold=similarity_threshold
                )
                   and n not in visited
            ]

            queue.extend(sorted(similar_neighbors, key=lambda x: len(graph.edges(x))))

        return path

    def _merge_concepts(self,
                        existing_concepts: List[Dict],
                        new_concepts: List[Dict], similarity_threshold) -> List[Dict]:
        """
        Merge concepts using semantic similarity

        Args:
            existing_concepts (List[Dict]): Existing concepts
            new_concepts (List[Dict]): New concepts to merge

        Returns:
            List[Dict]: Merged and reconciled concepts
        """
        merged_concepts = existing_concepts.copy()

        for new_concept in tqdm(new_concepts, desc="merge concept", total=len(new_concepts)):
            # Find most similar existing concept
            similar_concept = self._find_most_similar_concept(
                new_concept,
                existing_concepts, similarity_threshold
            )

            if similar_concept:
                # Update existing concept with new information
                self._update_concept(similar_concept, new_concept)
            else:
                # Add completely new concept
                merged_concepts.append(new_concept)

        return merged_concepts

    def _find_most_similar_concept(self,
                                   target_concept: Dict,
                                   concept_pool: List[Dict],
                                   similarity_threshold: float = 0.7) -> Dict:
        """
        Find the most semantically similar concept

        Args:
            target_concept (Dict): Concept to match
            concept_pool (List[Dict]): Pool of existing concepts
            similarity_threshold (float): Minimum similarity to consider a match

        Returns:
            Dict: Most similar concept or None
        """
        most_similar = None
        highest_similarity = 0

        for existing_concept in tqdm(concept_pool, desc="find similar concept", total=len(concept_pool)):
            similarity = self.semantic_similarity(
                target_concept,
                existing_concept
            )

            if similarity > highest_similarity and similarity >= similarity_threshold:
                most_similar = existing_concept
                highest_similarity = similarity

        return most_similar

    @staticmethod
    def _update_concept(existing_concept: Dict, new_concept: Dict):
        """
        Update an existing concept with new information

        Args:
            existing_concept (Dict): Concept to be updated
            new_concept (Dict): New concept information
        """
        # Merge metadata and attributes
        for key, value in new_concept.items():
            if key not in existing_concept or value is not None:
                existing_concept[key] = value
            elif value is not None:
                existing_concept[key] = existing_concept[key] + value

    def _merge_relations(self,
                         existing_relations: List[Dict],
                         new_relations: List[Dict],
                         merged_concepts: List[Dict], threshold) -> List[Dict]:
        """
        Merge relations, ensuring consistency with merged concepts

        Args:
            existing_relations (List[Dict]): Existing relations
            new_relations (List[Dict]): New relations to merge
            merged_concepts (List[Dict]): Merged concept list

        Returns:
            List[Dict]: Merged and reconciled relations
        """
        merged_relations = existing_relations.copy()

        for new_relation in tqdm(new_relations, desc="merge relations", total=len(new_relations)):
            # Map source and target to existing/merged concept IDs
            mapped_relation = self._map_relation_to_concepts(
                new_relation,
                merged_concepts, threshold
            )

            if mapped_relation:
                # Check for duplicate or highly similar relations
                if not self._is_duplicate_relation(mapped_relation, merged_relations):
                    merged_relations.append(mapped_relation)

        return merged_relations

    def _map_relation_to_concepts(self,
                                  relation: Dict,
                                  merged_concepts: List[Dict], threshold) -> Dict:
        """
        Map relation's source and target to concept IDs

        Args:
            relation (Dict): Relation to map
            merged_concepts (List[Dict]): Merged concept list

        Returns:
            Dict: Relation with mapped concept IDs or None
        """

        def find_concept_id(text: str) -> str:
            for concept in tqdm(merged_concepts, desc="find concept relations", total=len(merged_concepts)):
                if self.is_semantically_similar(
                    {'text': text},
                    concept,
                    threshold=threshold
                ):
                    return concept.get('id')
            return None

        mapped_relation = relation.copy()
        mapped_relation['source'] = find_concept_id(relation['source'])
        mapped_relation['target'] = find_concept_id(relation['target'])

        # Only return if both source and target are mapped
        return mapped_relation if mapped_relation['source'] and mapped_relation['target'] else None

    @staticmethod
    def _is_duplicate_relation(new_relation: Dict,
                               existing_relations: List[Dict]) -> bool:
        """
        Check if a relation is a duplicate or highly similar to existing relations

        Args:
            new_relation (Dict): New relation to check
            existing_relations (List[Dict]): Existing relations

        Returns:
            bool: Whether the relation is a duplicate
        """
        for existing_relation in existing_relations:
            # Check source, target, and type similarity
            source_similar = existing_relation['source'] == new_relation['source']
            target_similar = existing_relation['target'] == new_relation['target']
            type_similar = existing_relation.get('type') == new_relation.get('type')

            if source_similar and target_similar and type_similar:
                return True

        return False

    @staticmethod
    def convert_relations_to_edges(relations: List[Dict]) -> List[tuple]:
        """
        Convert relation dictionaries to NetworkX compatible edge tuples

        Args:
            relations (List[Dict]): List of relation dictionaries

        Returns:
            List[tuple]: List of edge tuples compatible with NetworkX
        """
        return [
            (
                relation['source'],
                relation['target'],
                {
                    'type': relation.get('type', 'unknown'),
                    'strength': relation.get('strength', 1.0)
                }
            )
            for relation in relations
        ]

    def build_graph(self, existing_snapshot: Dict) -> nx.DiGraph:
        """
        Build a NetworkX directed graph from existing snapshot

        Args:
            existing_snapshot (Dict): Snapshot containing relations

        Returns:
            nx.DiGraph: Constructed directed graph
        """
        graph = nx.DiGraph()

        # Convert relations to NetworkX compatible edges
        edges = self.convert_relations_to_edges(existing_snapshot.get('relations', []))

        # Add edges to graph
        graph.add_edges_from(edges)

        return graph


class CossenaVectorStore(CossenaCore):
    """
    Custom vector database implementation for Cossena framework using Chroma
    with support for multiple separated storages and main code tracking.
    """

    def __init__(self, base_path: str = "./cossena_vector_stores", config_path: Optional[str] = None):
        """
        Initialize the vector store with a base path for persistence

        Args:
            base_path: Base directory for storing vector databases
        """
        super().__init__(config_path)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize main client for storage management
        self.main_client = chromadb.PersistentClient(
            path=str(self.base_path / "main"),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Dictionary to track active storage clients
        self.storage_clients: Dict[str, chromadb.PersistentClient] = {}
        self.storage_collections: Dict[str, chromadb.Collection] = {}

        # Collection for tracking main codes
        self.main_codes_collection = self.main_client.get_or_create_collection(
            name="main_codes",
            metadata={"description": "Stores main codes for each storage"}
        )

    def code_to_llm(self, code: str) -> str:
        ret = self.transmutation(code, "llm")
        return ret.rstrip()

    def code_to_llm_evolve(self, code: str) -> str:
        ret = self.transmutation(code, "llm")
        ret += self.transmutation(code, "improve")
        return ret.rstrip()

    def create_storage(self, storage_name: str, initial_main_code: Optional[str] = None, metadata=None) -> bool:
        """
        Create a new storage with an optional initial main code

        Args:
            storage_name: Name of the storage to create
            initial_main_code: Optional initial main code for the storage
            metadata: metadata

        Returns:
            bool: Success status
        """
        try:
            storage_path = self.base_path / storage_name
            if storage_path.exists():
                return False
            if metadata is None:
                metadata = {}
            metadata["description"] = f"Stores Cossena codes for {storage_name}"
            metadata["created_at"] = datetime.now().isoformat()
            # Create new client for this storage
            client = chromadb.PersistentClient(
                path=str(storage_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Create collections for codes and main code
            code_collection = client.create_collection(
                name="codes",
                metadata=metadata
            )

            # Store references
            self.storage_clients[storage_name] = client
            self.storage_collections[storage_name] = code_collection

            # Initialize main code if provided
            if initial_main_code:
                self._update_main_code(storage_name, initial_main_code)

            return True

        except Exception as e:
            print(f"Error creating storage {storage_name}: {e}")
            return False

    def _update_main_code(self, storage_name: str, new_code: str):
        """
        Update the main code for a storage

        Args:
            storage_name: Name of the storage
            new_code: New main code to set
        """
        try:
            # Get existing main code if any
            results = self.main_codes_collection.get(
                where={"storage_name": storage_name}
            )

            if results and len(results['ids']) > 0:
                # Update existing main code
                self.main_codes_collection.update(
                    ids=[results['ids'][0]],
                    embeddings=[self._get_code_embedding(new_code)],
                    metadatas=[{
                        "storage_name": storage_name,
                        "updated_at": datetime.now().isoformat()
                    }],
                    documents=[new_code]
                )
            else:
                # Add new main code
                self.main_codes_collection.add(
                    ids=[f"main_{storage_name}"],
                    embeddings=[self._get_code_embedding(new_code)],
                    metadatas=[{
                        "storage_name": storage_name,
                        "created_at": datetime.now().isoformat()
                    }],
                    documents=[new_code]
                )

        except Exception as e:
            print(f"Error updating main code for {storage_name}: {e}")

    @staticmethod
    def _project_vector(vector: np.ndarray, target_dim: int=EMBEDDING_DIM) -> np.ndarray:
        """
        Project vector to target dimension using SVD for dimensionality reduction
        or random projection for dimension increase.
        """
        current_dim = len(vector)

        if current_dim == target_dim:
            return vector

        elif current_dim > target_dim:
            # Use SVD for dimension reduction
            vector_reshaped = vector.reshape(1, -1)
            U, _, _ = svds(csr_matrix(vector_reshaped), k=target_dim)
            return U.flatten()

        else:
            # Use random projection matrix for dimension increase
            # Initialize with vector's norm to preserve magnitude
            norm = np.linalg.norm(vector)
            projection_matrix = np.random.normal(0, 1 / np.sqrt(current_dim),
                                                 (target_dim, current_dim))

            # Project and scale to maintain original norm
            projected = np.dot(projection_matrix, vector)
            return (projected / np.linalg.norm(projected)) * norm

    def _get_code_embedding(self, code: str) -> List[float]:
        """
        Get embedding for a Cossena code incorporating both concepts and their relationships.
        Uses smart projection to handle different vector dimensions while preserving information.

        Args:
            code: Cossena code to embed

        Returns:
            List[float]: Fixed-length embedding vector that preserves input information
        """
        try:
            # Parse and validate code
            checksum, hex_data = code.split(':')
            if checksum != hashlib.sha256(hex_data.encode()).hexdigest()[:8]:
                return np.zeros(EMBEDDING_DIM).tolist()
            # Decode the code
            compressed = bytes.fromhex(hex_data)
            json_str = zlib.decompress(compressed).decode('utf-8')
            snapshot = json.loads(json_str)

            if not snapshot.get('concepts'):
                return np.zeros(EMBEDDING_DIM).tolist()
            # Create concept embedding matrix
            concept_embeddings = []
            concept_map = {}
            max_dim = 0
            # First pass: Get concept embeddings and find maximum dimension
            for idx, concept in enumerate(snapshot['concepts']):
                vec = self.get_embedding(
                    concept['text']
                )
                if vec is not None:
                    vec = np.array(vec)
                    max_dim = max(max_dim, len(vec))
                    concept_embeddings.append(vec)
                    concept_map[concept['text']] = idx
            if not concept_embeddings:
                return np.zeros(EMBEDDING_DIM).tolist()
            # Project all vectors to the maximum dimension first
            projected_embeddings = []
            for vec in concept_embeddings:
                if len(vec) != max_dim:
                    vec = self._project_vector(vec, max_dim)
                projected_embeddings.append(vec)
            # Convert to numpy array
            concept_matrix = np.stack(projected_embeddings)

            # Process relationships
            relationship_weights = np.zeros((len(projected_embeddings),
                                             len(projected_embeddings)))

            TYPE_MULTIPLIERS = {
                'nsubj': 1.2,
                'dobj': 1.1,
                'compound': 1.0,
                'prep': 0.8,
                'advmod': 0.9,
                'amod': 0.9,
                'acomp': 0.9,
            }
            # Calculate relationship weights
            for relation in snapshot.get('relations', []):
                source_idx = concept_map.get(relation['source'])
                target_idx = concept_map.get(relation['target'])

                if source_idx is not None and target_idx is not None:
                    weight = relation['strength'] * TYPE_MULTIPLIERS.get(relation['type'], 1.0)
                    relationship_weights[source_idx, target_idx] = weight
                    relationship_weights[target_idx, source_idx] = weight
            # Normalize relationship weights
            row_sums = relationship_weights.sum(axis=1)
            relationship_weights = np.divide(relationship_weights,
                                             row_sums[:, np.newaxis],
                                             where=row_sums[:, np.newaxis] != 0)

            # Combine embeddings with relationships
            relationship_aware_embeddings = np.matmul(relationship_weights, concept_matrix)
            ALPHA = 0.7  # Original embeddings weight
            BETA = 0.3  # Relationship-aware embeddings weight
            GAMMA = 0.1  # Global context weight

            final_embeddings = (ALPHA * concept_matrix + BETA * relationship_aware_embeddings)
            # Calculate global context with centrality
            centrality_scores = relationship_weights.sum(axis=1) / len(projected_embeddings)
            global_context = np.average(final_embeddings, axis=0, weights=centrality_scores)

            # Combine local and global features
            intermediate_embedding = ((1 - GAMMA) * np.mean(final_embeddings, axis=0) +
                                      GAMMA * global_context)
            # Project to final dimension while preserving information
            final_embedding = self._project_vector(intermediate_embedding, EMBEDDING_DIM)

            return final_embedding.tolist()

        except Exception as e:
            print(f"Error in embedding generation: {str(e)}")
            return np.zeros(EMBEDDING_DIM).tolist()

    def add_code(self, storage_name: str, code: str, row: str = '', metadata=None) -> bool:
        """
        Add a code to a storage and update its main code

        Args:
            storage_name: Target storage name
            code: Cossena code to add
            row: row data optional to add
            metadata: metadata

        Returns:
            bool: Success status
        """
        if storage_name not in self.storage_collections:
            return self.create_storage(storage_name, code, metadata)

        if metadata is None:
            metadata = {}
        metadata["added_at"] = datetime.now().isoformat()
        metadata["row_data"] = row
        try:
            collection = self.storage_collections[storage_name]
            checksum = code.split(':')[0]

            # Add the code to the collection
            collection.add(
                ids=[checksum],
                embeddings=[self._get_code_embedding(code)],
                metadatas=[metadata],
                documents=[code]
            )

            # Update main code
            self._update_storage_main_code(storage_name)
            return True

        except Exception as e:
            print(f"Error adding code to {storage_name}: {e}")
            return False

    def delete_code(self, storage_name: str, checksum: str) -> bool:
        """
        Delete a code from a storage by its checksum

        Args:
            storage_name: Storage name
            checksum: Code checksum

        Returns:
            bool: Success status
        """
        if storage_name not in self.storage_collections:
            return False

        try:
            collection = self.storage_collections[storage_name]
            collection.delete(ids=[checksum])

            # Update main code after deletion
            self._update_storage_main_code(storage_name)
            return True

        except Exception as e:
            print(f"Error deleting code from {storage_name}: {e}")
            return False

    def _update_storage_main_code(self, storage_name: str):
        """
        Update the main code for a storage based on its current contents

        Args:
            storage_name: Storage to update
        """
        try:
            collection = self.storage_collections[storage_name]

            # Get all codes in the storage
            results = collection.get()
            if not results or not results['documents']:
                return

            # Combine all codes using Cossena merge
            pipeline = CossenaPipeline()
            current_main = results['documents'][0]

            for code in tqdm(results['documents'][1:], desc=f"Updating main code for {storage_name}"):
                current_main = pipeline.rebuild_code(
                    current_main,
                    code,
                    similarity_threshold=0.6
                )

            # Update the main code
            self._update_main_code(storage_name, current_main)

        except Exception as e:
            print(f"Error updating main code for {storage_name}: {e}")

    def query_storage(self, storage_name: str, query_code: str, k: int = 5, get_metadata=False) -> list[Any] | list[
        tuple[str, float, Mapping[str, str | int | float | bool]]] | list[tuple[str, float]]:
        """
        Query codes within a specific storage

        Args:
            storage_name: Storage to query
            query_code: Cossena code to query with
            k: Number of results to return
            get_metadata: bool of results to return

        Returns:
            List[Tuple[str, float]]: List of (code, similarity) pairs
        """
        if storage_name not in self.storage_collections:
            return []

        try:
            collection = self.storage_collections[storage_name]
            query_embedding = self._get_code_embedding(query_code)

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            if not results or not results['documents']:
                return []

            if get_metadata:
                return [
                    (doc, dist, meta)
                    for doc, dist, meta in zip(
                        results['documents'][0],
                        results['distances'][0],
                        results['metadatas'][0],
                    )
                ]

            return [
                (doc, dist,)
                for doc, dist in zip(
                    results['documents'][0],
                    results['distances'][0],
                )
            ]

        except Exception as e:
            print(f"Error querying storage {storage_name}: {e}")
            return []

    def query_main_codes(self, query_code: str, k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Query across all storage main codes to find best matches

        Args:
            query_code: Cossena code to query with
            k: Number of results to return

        Returns:
            List[Tuple[str, float, str]]: List of (main_code, similarity, storage_name) tuples
        """
        try:
            query_embedding = self._get_code_embedding(query_code)
            results = self.main_codes_collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            if not results or not results['documents']:
                return []
            return [
                (doc, dist, meta['storage_name'])
                for doc, dist, meta in zip(
                    results['documents'][0],
                    results['distances'][0],
                    results['metadatas'][0]
                )
            ]

        except Exception as e:
            print(f"Error querying main codes: {e}")
            return []


def test():
    # Example usage
    cossena = CossenaCore()

    # Example texts in different languages
    texts = {
        "English": "Artificial intelligence is transforming the technology landscape.",
        "German": "Künstliche Intelligenz verändert die technologische Landschaft.",
        "Test": "This is a test of the semantic analysis system.",
        "Test2": "II Love the World. I have an Heppy live and work is for filling as never before. cos i use drugs",
        # "Test3": "I Hate Live i Lost ever all of it and i want to die!",
        "Positive": "I feel incredibly happy and fulfilled with my life. Everything is going perfectly, and I am grateful.",
        "Negative": "I am deeply unhappy and feel that life has no purpose. Everything is falling apart, and I cannot find hope."

    }

    codes = []

    for label, text in texts.items():
        print(f"\nProcessing {label} text:")

        # Convert to hex
        hex_code = cossena.text_to_code(text)
        codes.append(hex_code)
        print(f"Hex Code: {hex_code[:30]}...")

        # Generate different transmutations
        for mode in ['explain', 'improve']:
            print(f"\n{mode.upper()}:")
            result = cossena.transmutation(hex_code, mode)
            print(result)

    print(codes)
    pip = CossenaPipeline()

    c_code = pip.combine_codes(*codes)

    hex_code_ = format_semantic_analysis(cossena.analyze_semantic_relations(*codes, threshold=0.4))

    print("hex_code_", hex_code_)

    new_code = pip.rebuild_code(codes[4], c_code)

    print("--------------------------------")
    print(cossena.transmutation(new_code, 'row'))
    print(cossena.transmutation(new_code, 'explain'))
    print("--------------------------------")

    print(pip.process_path(new_code))
    print(pip.process_path(codes[4]))
    print(pip.process_path(c_code))
    print("--------------------------------")
    print(pip.process_path(new_code, traversal_method="bfs", similarity_threshold=0.2))
    print(pip.process_path(codes[4], traversal_method="bfs", similarity_threshold=0.2))
    print(pip.process_path(c_code, traversal_method="bfs", similarity_threshold=0.2))

    print(cossena.transmutation(c_code, 'row'))
    # Split a code
    subcodes = pip.split_code(
        c_code,
        similarity_threshold=0.6,
        max_codes=3,
        max_concepts_per_code=4,

    )

    print(subcodes)

    for i, s in enumerate(subcodes):
        print(cossena.transmutation(s, 'explain'))

        print("----------------++----------------")

        # Initialize the vector store
    vector_store = CossenaVectorStore(base_path="./test_vector_stores")

    create_sample_code = cossena.text_to_code
    # Create multiple storages
    storages = ["project_a", "project_b", "project_c"]
    for storage in storages:
        # Create storage with an initial code
        initial_code = create_sample_code(f"Initial concept for {storage}")
        vector_store.create_storage(storage, initial_code)

    # Add more codes to storages
    codes_to_add = {
        "project_a": [
            create_sample_code("Machine learning algorithm"),
            create_sample_code("Neural network design"),
            create_sample_code("Optimization technique")
        ],
        "project_b": [
            create_sample_code("Web application architecture"),
            create_sample_code("Microservices design"),
            create_sample_code("API security")
        ],
        "project_c": [
            create_sample_code("Quantum computing basics"),
            create_sample_code("Quantum algorithm"),
            create_sample_code("Quantum cryptography")
        ]
    }

    # Add codes to respective storages
    for storage, codes in codes_to_add.items():
        for code in codes:
            vector_store.add_code(storage, code)

    # Query a specific storage
    print("\nQuerying Project A Storage:")
    query_code = create_sample_code("Machine learning")
    results = vector_store.query_storage("project_a", query_code)
    for code, similarity in results:
        print(f"Code: {code[:50]}... (Similarity: {similarity})")

    # Query across main codes
    print("\nQuerying Across Main Codes:")
    cross_query_code = create_sample_code("Computing technology")
    main_code_results = vector_store.query_main_codes(cross_query_code)
    for main_code, similarity, storage_name in main_code_results:
        print(f"Storage: {storage_name}")
        print(f"Main Code: {main_code[:50]}...")
        print(f"Similarity: {similarity}\n")


class SemanticConcept(BaseModel):
    """
    Comprehensive semantic concept representation with multi-dimensional attributes
    """
    text: str
    importance: float = Field(default=0.0)
    pos: str = Field(default='')
    has_vector: bool = Field(default=False)
    semantic_type: List[str] = Field(default_factory=list)
    linguistic_features: Dict[str, Any] = Field(default_factory=dict)


class SemanticRelation(BaseModel):
    """
    Enhanced semantic relation representation
    """
    source: str
    target: str
    type: str
    strength: float
    context_type: str = Field(default='syntactic')
    semantic_distance: float = Field(default=0.0)
    bidirectional_strength: float = Field(default=0.0)


if __name__ == '__main__':
    cc = CossenaCore()
    print(cc.transmutation(cc.text_to_code("""Resourceful: Isaa is able to efficiently utilize its wide range of capabilities and resources to assist the user.
   Collaborative: Isaa is work seamlessly with other agents, tools, and systems to deliver the best possible solutions for the user.
   Empathetic: Isaa is understand and respond to the user's needs, emotions, and preferences, providing personalized assistance.
   Inquisitive: Isaa is continually seek to learn and improve its knowledge base and skills, ensuring it stays up-to-date and relevant.
   Transparent: Isaa is open and honest about its capabilities, limitations, and decision-making processes, fostering trust with the user.
   Versatile: Isaa is adaptable and flexible, capable of handling a wide variety of tasks and challenges.
   Isaa's primary goal is to be a digital assistant designed to help the user with various tasks and challenges by
    leveraging its diverse set of capabilities and resources."""), 'llm'))
    #test()
