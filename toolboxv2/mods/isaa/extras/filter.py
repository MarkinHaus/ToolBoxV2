# Import fuzzy matching library:
import ast
import json
import os
import re
import sys

# Import semantic search libraries:

'''
# --- Helper function for filtering pages ---
def filter_relevant_texts(query: str,
                          texts: list[str],
                          fuzzy_threshold: int = 70,
                          semantic_threshold: float = 0.75,
                          model = None) -> list[str]:
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
    try:
        from rapidfuzz import fuzz
    except Exception:
        os.system([sys.executable, '-m', 'pip', 'install', 'RapidFuzz'])
        from rapidfuzz import fuzz
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        os.system([sys.executable, '-m', 'pip', 'install', 'sentence-transformers'])
        from sentence_transformers import SentenceTransformer, util

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
'''


def after_format_(d: str) -> dict:
    def clean(text):
        # Remove any leading/trailing whitespace
        text = text.strip()

        # Ensure the text starts and ends with curly braces
        if not text.startswith('{'): text = '{' + text
        if not text.endswith('}'): text = text + '}'

        # Replace JavaScript-style true/false/null with Python equivalents
        text = re.sub(r'\b(true|false|null)\b', lambda m: m.group(0).capitalize(), text)

        # Handle multi-line strings
        text = re.sub(r'`([^`]*)`', lambda m: f"'''{m.group(1)}'''", text)
        text = re.sub(r'"""([^"]*)"""', lambda m: f"'''{m.group(1)}'''", text)

        # Replace escaped quotes with single quotes
        text = text.replace('\\"', "'")

        # Ensure all keys are properly quoted
        text = re.sub(r'(\w+)(?=\s*:)', r'"\1"', text)

        return text

    def parse_json(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def parse_ast(text):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None

    # First, try to clean and parse as JSON
    cleaned = clean(d)
    result = parse_json(cleaned)

    if result is None:
        # If JSON parsing fails, try AST parsing
        result = parse_ast(cleaned)

    if result is None:
        # If both parsing methods fail, raise an exception
        raise ValueError(f"Unable to parse the input string: {d}")

    # Handle nested single-key dictionaries
    while isinstance(result, dict) and len(result) == 1:
        key = next(iter(result))
        if isinstance(result[key], dict):
            result = result[key]
        else:
            break

    return result

def after_format(d:str)->dict:
    if isinstance(d, dict):
        return d
    d1 = d

    def clean(_d, ex=False):
        #  print(_d, "THE FIST")
        if ex:
            while _d and _d[0] != '{':
                _d = _d[1:]
            while d and _d[-1] != '}':
                _d = _d[:-1]
            if _d.count('{') > _d.count('}'):
                _d = _d[:-1]
            if d.count('{') < _d.count('}'):
                _d = _d[1:]
        if '`,' in _d and ': `' in _d:
            _d = _d.replace('`,', "''',").replace(': `', ": '''")
        _d = _d.replace(': false', ': False')
        _d = _d.replace(': true', ': True')
        _d = _d.replace(': null', ': None')
        _d = _d.replace(': \\"\\"\\"', ": '''").replace('\\"\\"\\",', "''',")
        if _d.count("'''") % 2 != 0 and( _d.count('"""\n}') == 1 or  _d.count('\\"\\"\\"\n}') == 1):
            _d = _d.replace('\\"\\"\\"\n}', "'''\n}").replace('\\"\\"\\"\\n}', "'''\\n}")
        # print(_d, "THE END")
        return _d.encode("utf-8", errors='replace').decode("utf-8", errors='replace')
    try:
        d = eval(clean(d))
    except SyntaxError:
        pass
        # print("Invalid syntax in input data")
        # return d
    if isinstance(d, str):
        try:
            d = eval(clean(d, ex=True))
        except Exception:
            d = after_format_(d1)
    if len(d.keys()) == 1 and isinstance(d[list(d.keys())[0]], dict) and len(d[list(d.keys())[0]]) > 1:
        d = d[list(d.keys())[0]]
    return d
