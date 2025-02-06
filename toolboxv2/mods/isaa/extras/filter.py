# Import fuzzy matching library:
from typing import List

from rapidfuzz import fuzz

# Import semantic search libraries:
from sentence_transformers import SentenceTransformer, util


# --- Helper function for filtering pages ---
def filter_relevant_texts(query: str,
                          texts: List[str],
                          fuzzy_threshold: int = 70,
                          semantic_threshold: float = 0.75,
                          model: SentenceTransformer = None) -> List[str]:
    """
    Filters a list of texts based on their relevance to the query.
    It first uses a fuzzy matching score and, if that score is below the threshold,
    it then checks the semantic similarity.

    :param query: The query string.
    :param texts: List of page texts.
    :param fuzzy_threshold: Fuzzy matching score threshold (0-100).
    :param semantic_threshold: Semantic similarity threshold (0.0-1.0).
    :param model: A preloaded SentenceTransformer model (if None, one will be loaded).
    :return: Filtered list of texts deemed relevant.
    """
    if model is None:
        # For efficiency, consider pre-loading this model outside the function.
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Pre-compute query embedding for the semantic check:
    query_embedding = model.encode(query, convert_to_tensor=True)

    relevant_texts = []
    for text in texts:
        # --- Fuzzy Keyword Filtering ---
        fuzzy_score = fuzz.partial_ratio(query.lower(), text.lower())
        if fuzzy_score >= fuzzy_threshold:
            relevant_texts.append(text)
        else:
            # --- Semantic Similarity Filtering ---
            text_embedding = model.encode(text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, text_embedding).item()
            if similarity >= semantic_threshold:
                relevant_texts.append(text)
    return relevant_texts
